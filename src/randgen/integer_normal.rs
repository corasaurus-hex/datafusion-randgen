use dashu::{
    base::{BitTest, Sign},
    integer::{IBig, UBig},
    rational::RBig,
};
use datafusion_common::{Result, exec_err};
use rand::Rng;
use rand_distr::StandardNormal;

const F64_EXACT_INTEGER_LIMIT: i128 = 1_i128 << 53;
const F64_SIGNIFICAND_STORED_BITS: i32 = 52;
const DITHER_GUARD_BITS: i32 = 3;
const MAX_DITHER_BITS: u32 = 64;
const MAX_REJECTION_ATTEMPTS: usize = 4_096;
// Below 5% acceptance, fast-path rejection needs at least 20 draws on average.
// Exact bounded/tail proposals avoid that wasted work and avoid retry-cap risk.
const MIN_FAST_REJECTION_ACCEPTANCE_PROBABILITY: f64 = 0.05;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

pub(crate) enum IntegerNormalSampler {
    Constant {
        offset: i128,
    },
    F64 {
        stddev: f64,
        min_offset: i128,
        max_offset: i128,
    },
    F64Dither {
        stddev: f64,
        dither_bits: u32,
        min_offset: i128,
        max_offset: i128,
    },
    Exact {
        sigma: PositiveRational,
        strategy: ExactStrategy,
        min_offset: i128,
        max_offset: i128,
    },
}

#[derive(Clone)]
pub(crate) struct PositiveRational {
    numer: UBig,
    denom: UBig,
    ceil: UBig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExactStrategy {
    UnboundedKarney,
    UniformRange,
    LeftTail,
    RightTail,
}

impl IntegerNormalSampler {
    pub(crate) fn for_offset_range(
        min_offset: i128,
        max_offset: i128,
        stddev: f64,
        name: &str,
    ) -> Result<Self> {
        validate_stddev(stddev, name)?;
        if min_offset > max_offset {
            return exec_err!("{name} requires min <= max");
        }
        if min_offset == max_offset {
            return Ok(Self::Constant { offset: min_offset });
        }

        if let Some(exact_strategy) = low_acceptance_exact_strategy(min_offset, max_offset, stddev)
        {
            return Ok(Self::Exact {
                sigma: PositiveRational::try_from_f64(stddev, name)?,
                strategy: exact_strategy,
                min_offset,
                max_offset,
            });
        }

        let dither_bits = dither_bits(stddev);
        if dither_bits == 0 && offset_range_fits_f64(min_offset, max_offset) {
            return Ok(Self::F64 {
                stddev,
                min_offset,
                max_offset,
            });
        }

        if (1..=MAX_DITHER_BITS).contains(&dither_bits) {
            return Ok(Self::F64Dither {
                stddev,
                dither_bits,
                min_offset,
                max_offset,
            });
        }

        Ok(Self::Exact {
            sigma: PositiveRational::try_from_f64(stddev, name)?,
            strategy: ExactStrategy::UnboundedKarney,
            min_offset,
            max_offset,
        })
    }

    pub(crate) fn sample_offset<R: Rng + ?Sized>(&self, rng: &mut R, name: &str) -> Result<i128> {
        match self {
            Self::Constant { offset } => Ok(*offset),
            Self::F64 {
                stddev,
                min_offset,
                max_offset,
            } => sample_f64_offset(rng, *stddev, *min_offset, *max_offset, name),
            Self::F64Dither {
                stddev,
                dither_bits,
                min_offset,
                max_offset,
            } => {
                sample_f64_dither_offset(rng, *stddev, *dither_bits, *min_offset, *max_offset, name)
            }
            Self::Exact {
                sigma,
                strategy,
                min_offset,
                max_offset,
            } => sample_exact_offset(rng, sigma, *strategy, *min_offset, *max_offset, name),
        }
    }

    #[cfg(test)]
    pub(crate) fn uses_f64(&self) -> bool {
        matches!(self, Self::F64 { .. } | Self::F64Dither { .. })
    }

    #[cfg(test)]
    pub(crate) fn uses_dither(&self) -> bool {
        matches!(self, Self::F64Dither { .. })
    }

    #[cfg(test)]
    pub(crate) fn uses_integer_domain(&self) -> bool {
        matches!(self, Self::Exact { .. })
    }

    #[cfg(test)]
    pub(crate) fn uses_constant(&self) -> bool {
        matches!(self, Self::Constant { .. })
    }

    #[cfg(test)]
    pub(crate) fn uses_uniform_range(&self) -> bool {
        matches!(
            self,
            Self::Exact {
                strategy: ExactStrategy::UniformRange,
                ..
            }
        )
    }

    #[cfg(test)]
    pub(crate) fn uses_tail(&self) -> bool {
        matches!(
            self,
            Self::Exact {
                strategy: ExactStrategy::LeftTail | ExactStrategy::RightTail,
                ..
            }
        )
    }

    #[cfg(test)]
    pub(crate) fn uses_unbounded_karney(&self) -> bool {
        matches!(
            self,
            Self::Exact {
                strategy: ExactStrategy::UnboundedKarney,
                ..
            }
        )
    }
}

impl PositiveRational {
    fn try_from_f64(value: f64, name: &str) -> Result<Self> {
        let rational = match RBig::try_from(value) {
            Ok(rational) => rational,
            Err(error) => return exec_err!("{name} could not convert stddev exactly: {error:?}"),
        };
        let (numer, denom) = rational.into_parts();
        let (Sign::Positive, numer) = numer.into_parts() else {
            return exec_err!("{name} requires stddev > 0");
        };
        if numer.is_zero() {
            return exec_err!("{name} requires stddev > 0");
        }

        let ceil = ceil_ubig_ratio(&numer, &denom);
        Ok(Self { numer, denom, ceil })
    }

    fn ceil_mul_ubig(&self, multiplier: &UBig) -> UBig {
        ceil_ubig_ratio(&(&self.numer * multiplier), &self.denom)
    }

    fn gaussian_exponent_from_squared_delta(&self, squared_delta: UBig) -> RBig {
        if squared_delta.is_zero() {
            return RBig::ZERO;
        }

        let numer = squared_delta * &self.denom * &self.denom;
        let denom = UBig::from(2u8) * &self.numer * &self.numer;
        RBig::from_parts(numer.into(), denom)
    }

    fn gaussian_tail_rate(&self, near_abs: &UBig) -> RBig {
        if near_abs.is_zero() {
            return RBig::ZERO;
        }

        let numer = near_abs * &self.denom * &self.denom;
        let denom = &self.numer * &self.numer;
        RBig::from_parts(numer.into(), denom)
    }
}

fn validate_stddev(stddev: f64, name: &str) -> Result<()> {
    if !stddev.is_finite() {
        return exec_err!("{name} requires finite stddev");
    }
    if stddev <= 0.0 {
        return exec_err!("{name} requires stddev > 0");
    }

    Ok(())
}

fn offset_range_fits_f64(min_offset: i128, max_offset: i128) -> bool {
    min_offset >= -F64_EXACT_INTEGER_LIMIT && max_offset <= F64_EXACT_INTEGER_LIMIT
}

fn dither_bits(stddev: f64) -> u32 {
    let exponent = stddev.log2().floor() as i32;
    let bits = exponent + DITHER_GUARD_BITS - F64_SIGNIFICAND_STORED_BITS;
    bits.max(0) as u32
}

fn low_acceptance_exact_strategy(
    min_offset: i128,
    max_offset: i128,
    stddev: f64,
) -> Option<ExactStrategy> {
    let acceptance = estimated_truncation_acceptance_probability(min_offset, max_offset, stddev);
    if acceptance >= MIN_FAST_REJECTION_ACCEPTANCE_PROBABILITY {
        return None;
    }

    if min_offset > 0 {
        Some(ExactStrategy::RightTail)
    } else if max_offset < 0 {
        Some(ExactStrategy::LeftTail)
    } else {
        Some(ExactStrategy::UniformRange)
    }
}

fn estimated_truncation_acceptance_probability(
    min_offset: i128,
    max_offset: i128,
    stddev: f64,
) -> f64 {
    let lower = (min_offset as f64 - 0.5) / stddev;
    let upper = (max_offset as f64 + 0.5) / stddev;
    (standard_normal_cdf(upper) - standard_normal_cdf(lower)).clamp(0.0, 1.0)
}

fn standard_normal_cdf(x: f64) -> f64 {
    if x <= -8.0 {
        return 0.0;
    }
    if x >= 8.0 {
        return 1.0;
    }

    0.5 * (1.0 + erf_approx(x / SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();

    sign * y
}

fn sample_f64_offset<R: Rng + ?Sized>(
    rng: &mut R,
    stddev: f64,
    min_offset: i128,
    max_offset: i128,
    name: &str,
) -> Result<i128> {
    let min_offset = min_offset as f64;
    let max_offset = max_offset as f64;

    for _ in 0..MAX_REJECTION_ATTEMPTS {
        let z: f64 = rng.sample(StandardNormal);
        let offset = (z * stddev).round();
        if offset.is_finite() && min_offset <= offset && offset <= max_offset {
            return Ok(offset as i128);
        }
    }

    rejection_err(name)
}

fn sample_f64_dither_offset<R: Rng + ?Sized>(
    rng: &mut R,
    stddev: f64,
    dither_bits: u32,
    min_offset: i128,
    max_offset: i128,
    name: &str,
) -> Result<i128> {
    for _ in 0..MAX_REJECTION_ATTEMPTS {
        let z: f64 = rng.sample(StandardNormal);
        let offset = (z * stddev).round();
        if !offset.is_finite() || offset < i128::MIN as f64 || offset > i128::MAX as f64 {
            continue;
        }

        let Some(offset) = (offset as i128).checked_add(sample_centered_dither(rng, dither_bits))
        else {
            continue;
        };

        if min_offset <= offset && offset <= max_offset {
            return Ok(offset);
        }
    }

    rejection_err(name)
}

fn sample_centered_dither<R: Rng + ?Sized>(rng: &mut R, dither_bits: u32) -> i128 {
    if dither_bits == 0 {
        return 0;
    }

    let half_width = 1_i128 << (dither_bits - 1);
    rng.random_range(-half_width..=half_width)
}

fn sample_exact_offset<R: Rng + ?Sized>(
    rng: &mut R,
    sigma: &PositiveRational,
    strategy: ExactStrategy,
    min_offset: i128,
    max_offset: i128,
    name: &str,
) -> Result<i128> {
    match strategy {
        ExactStrategy::UnboundedKarney => {
            sample_unbounded_karney_offset(rng, sigma, min_offset, max_offset, name)
        }
        ExactStrategy::UniformRange => {
            sample_uniform_range_offset(rng, sigma, min_offset, max_offset, name)
        }
        ExactStrategy::LeftTail | ExactStrategy::RightTail => {
            sample_tail_offset(rng, sigma, strategy, min_offset, max_offset, name)
        }
    }
}

// Exact fallback paths sample the integer-domain distribution directly. The
// unbounded route follows Karney Algorithm D for a centered discrete normal;
// bounded uniform and geometric-tail proposals sample the already-truncated
// distribution when ordinary rejection would waste most draws.
fn sample_unbounded_karney_offset<R: Rng + ?Sized>(
    rng: &mut R,
    sigma: &PositiveRational,
    min_offset: i128,
    max_offset: i128,
    name: &str,
) -> Result<i128> {
    for _ in 0..MAX_REJECTION_ATTEMPTS {
        let Some(offset) = sample_karney_discrete_gaussian_i128(rng, sigma)? else {
            continue;
        };
        if min_offset <= offset && offset <= max_offset {
            return Ok(offset);
        }
    }

    if min_offset > 0 {
        sample_tail_offset(
            rng,
            sigma,
            ExactStrategy::RightTail,
            min_offset,
            max_offset,
            name,
        )
    } else if max_offset < 0 {
        sample_tail_offset(
            rng,
            sigma,
            ExactStrategy::LeftTail,
            min_offset,
            max_offset,
            name,
        )
    } else {
        sample_uniform_range_offset(rng, sigma, min_offset, max_offset, name)
    }
}

fn sample_uniform_range_offset<R: Rng + ?Sized>(
    rng: &mut R,
    sigma: &PositiveRational,
    min_offset: i128,
    max_offset: i128,
    name: &str,
) -> Result<i128> {
    let mode_abs = nearest_range_abs(min_offset, max_offset);

    for _ in 0..MAX_REJECTION_ATTEMPTS {
        let offset = sample_uniform_i128_inclusive(rng, min_offset, max_offset)?;
        let offset_abs = abs_i128_to_ubig(offset);
        let squared_delta = squared_difference_from_mode(offset_abs, &mode_abs);
        if sample_bernoulli_exp(
            rng,
            sigma.gaussian_exponent_from_squared_delta(squared_delta),
        )? {
            return Ok(offset);
        }
    }

    rejection_err(name)
}

fn sample_tail_offset<R: Rng + ?Sized>(
    rng: &mut R,
    sigma: &PositiveRational,
    strategy: ExactStrategy,
    min_offset: i128,
    max_offset: i128,
    name: &str,
) -> Result<i128> {
    let (near_offset, width) = match strategy {
        ExactStrategy::RightTail => (min_offset, (max_offset - min_offset) as u128),
        ExactStrategy::LeftTail => (max_offset, (max_offset - min_offset) as u128),
        ExactStrategy::UnboundedKarney | ExactStrategy::UniformRange => {
            return exec_err!("tail sampler requires a one-sided strategy");
        }
    };
    let near_abs = abs_i128_to_ubig(near_offset);
    let width = UBig::from(width);
    let tail_rate = sigma.gaussian_tail_rate(&near_abs);

    for _ in 0..MAX_REJECTION_ATTEMPTS {
        let distance = sample_geometric_exp_fast(rng, tail_rate.clone())?;
        if distance > width {
            continue;
        }

        let squared_distance = &distance * &distance;
        if !sample_bernoulli_exp(
            rng,
            sigma.gaussian_exponent_from_squared_delta(squared_distance),
        )? {
            continue;
        }

        let Some(distance) = ubig_to_u128(&distance) else {
            continue;
        };
        let Some(offset) = tail_candidate(near_offset, distance, strategy) else {
            continue;
        };
        return Ok(offset);
    }

    sample_uniform_range_offset(rng, sigma, min_offset, max_offset, name)
}

fn tail_candidate(near_offset: i128, distance: u128, strategy: ExactStrategy) -> Option<i128> {
    let distance = i128::try_from(distance).ok()?;
    match strategy {
        ExactStrategy::RightTail => near_offset.checked_add(distance),
        ExactStrategy::LeftTail => near_offset.checked_sub(distance),
        ExactStrategy::UnboundedKarney | ExactStrategy::UniformRange => None,
    }
}

fn sample_karney_discrete_gaussian_i128<R: Rng + ?Sized>(
    rng: &mut R,
    sigma: &PositiveRational,
) -> Result<Option<i128>> {
    loop {
        let k = sample_karney_integer_part(rng)?;
        let positive = sample_standard_bernoulli(rng);
        let j = sample_uniform_ubig_below(rng, sigma.ceil.clone())?;
        let i0 = sigma.ceil_mul_ubig(&k);
        let candidate_magnitude = &i0 + &j;

        let scaled_integer = &sigma.denom * &candidate_magnitude;
        let scaled_base = &sigma.numer * &k;
        if scaled_integer < scaled_base {
            return exec_err!("internal error: Karney fractional part was negative");
        }
        let x_num = scaled_integer - scaled_base;

        if x_num >= sigma.numer {
            continue;
        }
        if k.is_zero() && x_num.is_zero() && !positive {
            continue;
        }

        if accept_karney_fractional_part(rng, sigma, &k, &x_num)? {
            return Ok(signed_ubig_to_i128(&candidate_magnitude, positive));
        }
    }
}

fn sample_karney_integer_part<R: Rng + ?Sized>(rng: &mut R) -> Result<UBig> {
    loop {
        let mut k = UBig::ZERO;
        while sample_half_exp_bernoulli(rng)? {
            k += UBig::ONE;
        }

        let mut correction_trials = if k.is_zero() {
            UBig::ZERO
        } else {
            &k * (&k - UBig::ONE)
        };
        let mut accepted = true;
        while !correction_trials.is_zero() {
            if !sample_half_exp_bernoulli(rng)? {
                accepted = false;
                break;
            }
            correction_trials -= UBig::ONE;
        }

        if accepted {
            return Ok(k);
        }
    }
}

fn accept_karney_fractional_part<R: Rng + ?Sized>(
    rng: &mut R,
    sigma: &PositiveRational,
    k: &UBig,
    x_num: &UBig,
) -> Result<bool> {
    if x_num.is_zero() {
        return Ok(true);
    }

    let numer = x_num * (UBig::from(2u8) * k * &sigma.numer + x_num);
    let denom = UBig::from(2u8) * &sigma.numer * &sigma.numer;
    sample_bernoulli_exp(rng, RBig::from_parts(numer.into(), denom))
}

fn sample_half_exp_bernoulli<R: Rng + ?Sized>(rng: &mut R) -> Result<bool> {
    sample_bernoulli_exp1(rng, RBig::from_parts(IBig::from(1), UBig::from(2u8)))
}

fn nearest_range_abs(min_offset: i128, max_offset: i128) -> UBig {
    if min_offset <= 0 && max_offset >= 0 {
        UBig::ZERO
    } else if min_offset > 0 {
        UBig::from(min_offset as u128)
    } else {
        UBig::from(max_offset.unsigned_abs())
    }
}

fn squared_difference_from_mode(offset_abs: UBig, mode_abs: &UBig) -> UBig {
    let offset_square = &offset_abs * &offset_abs;
    let mode_square = mode_abs * mode_abs;
    offset_square - mode_square
}

fn abs_i128_to_ubig(value: i128) -> UBig {
    if value < 0 {
        UBig::from(value.unsigned_abs())
    } else {
        UBig::from(value as u128)
    }
}

fn signed_ubig_to_i128(value: &UBig, positive: bool) -> Option<i128> {
    let value = ubig_to_u128(value)?;
    if positive {
        i128::try_from(value).ok()
    } else if value == (i128::MAX as u128) + 1 {
        Some(i128::MIN)
    } else {
        i128::try_from(value).ok().map(|value| -value)
    }
}

fn ubig_to_u128(value: &UBig) -> Option<u128> {
    u128::try_from(value).ok()
}

fn sample_uniform_i128_inclusive<R: Rng + ?Sized>(
    rng: &mut R,
    min: i128,
    max: i128,
) -> Result<i128> {
    if min > max {
        return exec_err!("uniform range requires min <= max");
    }

    let span = max
        .checked_sub(min)
        .and_then(|span| u128::try_from(span).ok())
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| datafusion_common::DataFusionError::Internal("range is too large".into()))?;
    let offset = sample_uniform_u128_below(rng, span);
    Ok(min + offset as i128)
}

fn sample_uniform_u128_below<R: Rng + ?Sized>(rng: &mut R, upper: u128) -> u128 {
    debug_assert!(upper > 0);

    if upper <= u64::MAX as u128 {
        let upper = upper as u64;
        let threshold = u64::MAX - u64::MAX % upper;
        loop {
            let sample = rng.random::<u64>();
            if sample < threshold {
                return (sample % upper) as u128;
            }
        }
    }
    if upper == (u64::MAX as u128) + 1 {
        return rng.random::<u64>() as u128;
    }

    let threshold = u128::MAX - u128::MAX % upper;
    loop {
        let sample = rng.random::<u128>();
        if sample < threshold {
            return sample % upper;
        }
    }
}

fn ceil_ubig_ratio(numer: &UBig, denom: &UBig) -> UBig {
    debug_assert!(!denom.is_zero());

    let quotient = numer / denom;
    if numer % denom == UBig::ZERO {
        quotient
    } else {
        quotient + UBig::ONE
    }
}

fn rejection_err<T>(name: &str) -> Result<T> {
    exec_err!(
        "{name} could not sample an in-range normal value after {MAX_REJECTION_ATTEMPTS} attempts; widen the range or reduce stddev"
    )
}

fn gcd_ubig(mut a: UBig, mut b: UBig) -> UBig {
    while !b.is_zero() {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a
}

fn div_rbig_by_ubig_exact(numer: &UBig, denom: &UBig, k: &UBig) -> RBig {
    assert!(!k.is_zero(), "division by zero");

    if numer.is_zero() {
        return RBig::ZERO;
    }

    let g = gcd_ubig(numer.clone(), k.clone());
    let n_red = numer / &g;
    let k_red = k / g;

    RBig::from_parts(n_red.into(), denom * k_red)
}

fn sample_standard_bernoulli<R: Rng + ?Sized>(rng: &mut R) -> bool {
    rng.random()
}

fn sample_uniform_ubig_below<R: Rng + ?Sized>(rng: &mut R, upper: UBig) -> Result<UBig> {
    if upper.is_zero() {
        return exec_err!("upper must be positive");
    }

    let byte_len = upper.bit_len().div_ceil(8);
    let radix = UBig::ONE << (byte_len * 8);
    let threshold = &radix - &radix % &upper;
    let mut buffer = vec![0; byte_len];

    loop {
        rng.fill(&mut buffer[..]);

        let sample = UBig::from_be_bytes(&buffer);
        if sample < threshold {
            return Ok(sample % &upper);
        }
    }
}

fn sample_bernoulli_rational<R: Rng + ?Sized>(rng: &mut R, prob: RBig) -> Result<bool> {
    let (numer, denom) = prob.into_parts();
    let (Sign::Positive, numer) = numer.into_parts() else {
        return exec_err!("prob must be in [0, 1]");
    };
    if numer > denom {
        return exec_err!("prob must not be greater than one");
    }

    sample_uniform_ubig_below(rng, denom).map(|sample| numer > sample)
}

fn sample_bernoulli_exp1<R: Rng + ?Sized>(rng: &mut R, x: RBig) -> Result<bool> {
    let (numer_signed, denom) = x.into_parts();
    let (Sign::Positive, numer) = numer_signed.into_parts() else {
        return exec_err!("x must be in [0, 1]");
    };
    if numer > denom {
        return exec_err!("x must be in [0, 1]");
    }

    let mut k = UBig::ONE;
    loop {
        let x_div_k = div_rbig_by_ubig_exact(&numer, &denom, &k);

        if sample_bernoulli_rational(rng, x_div_k)? {
            k += UBig::ONE;
        } else {
            return Ok(k % 2u8 == 1);
        }
    }
}

fn sample_bernoulli_exp<R: Rng + ?Sized>(rng: &mut R, mut x: RBig) -> Result<bool> {
    while x > RBig::ONE {
        if sample_bernoulli_exp1(rng, RBig::ONE)? {
            x -= RBig::ONE;
        } else {
            return Ok(false);
        }
    }

    sample_bernoulli_exp1(rng, x)
}

fn sample_geometric_exp_slow<R: Rng + ?Sized>(rng: &mut R, x: RBig) -> Result<UBig> {
    let mut k = UBig::ZERO;
    loop {
        if sample_bernoulli_exp(rng, x.clone())? {
            k += UBig::ONE;
        } else {
            return Ok(k);
        }
    }
}

fn sample_geometric_exp_fast<R: Rng + ?Sized>(rng: &mut R, x: RBig) -> Result<UBig> {
    if x.is_zero() {
        return Ok(UBig::ZERO);
    }

    let (numer, denom) = x.into_parts();
    let (Sign::Positive, numer) = numer.into_parts() else {
        return exec_err!("x must be non-negative");
    };

    let mut u = sample_uniform_ubig_below(rng, denom.clone())?;
    while !sample_bernoulli_exp(rng, RBig::from_parts(u.as_ibig().clone(), denom.clone()))? {
        u = sample_uniform_ubig_below(rng, denom.clone())?;
    }
    let v2 = sample_geometric_exp_slow(rng, RBig::ONE)?;

    Ok((v2 * denom + u) / numer)
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    const DISTRIBUTION_CHI_SQUARE_Z_LIMIT: f64 = 4.0;
    const DISTRIBUTION_MAX_RELATIVE_ERROR: f64 = 0.20;

    #[test]
    fn chooses_f64_when_offset_range_is_exactly_representable() {
        let sampler =
            IntegerNormalSampler::for_offset_range(-(1_i128 << 53), 1_i128 << 53, 1.0, "test")
                .unwrap();

        assert!(sampler.uses_f64());
        assert!(!sampler.uses_dither());
    }

    #[test]
    fn chooses_unbounded_karney_when_offset_range_exceeds_f64_integer_precision() {
        let sampler =
            IntegerNormalSampler::for_offset_range(-(1_i128 << 53) - 1, 1, 1.0, "test").unwrap();

        assert!(sampler.uses_integer_domain());
        assert!(sampler.uses_unbounded_karney());
    }

    #[test]
    fn chooses_dither_when_stddev_can_skip_integer_low_bits() {
        let sampler = IntegerNormalSampler::for_offset_range(
            -(1_i128 << 53),
            1_i128 << 53,
            (1_u64 << 53) as f64,
            "test",
        )
        .unwrap();

        assert!(sampler.uses_f64());
        assert!(sampler.uses_dither());
    }

    #[test]
    fn chooses_constant_for_single_value_range() {
        let sampler = IntegerNormalSampler::for_offset_range(7, 7, 1.0, "test").unwrap();

        assert!(sampler.uses_constant());
    }

    #[test]
    fn chooses_uniform_range_when_range_is_narrow_relative_to_stddev() {
        let sampler = IntegerNormalSampler::for_offset_range(-5, 5, 1_000_000.0, "test").unwrap();

        assert!(sampler.uses_uniform_range());
    }

    #[test]
    fn keeps_fast_path_when_truncation_acceptance_is_high() {
        let sampler = IntegerNormalSampler::for_offset_range(1, 20, 10.0, "test").unwrap();

        assert!(sampler.uses_f64());
    }

    #[test]
    fn chooses_tail_when_range_is_far_from_mean() {
        let sampler = IntegerNormalSampler::for_offset_range(1_000, 10_000, 10.0, "test").unwrap();

        assert!(sampler.uses_tail());
    }

    #[test]
    fn estimates_truncation_acceptance_probability() {
        let centered = estimated_truncation_acceptance_probability(-10, 10, 10.0);
        let tiny = estimated_truncation_acceptance_probability(0, 4, 100.0);
        let tail = estimated_truncation_acceptance_probability(30, 44, 10.0);

        assert!((0.70..=0.72).contains(&centered));
        assert!((0.019..=0.021).contains(&tiny));
        assert!((0.0007..=0.0016).contains(&tail));
    }

    #[test]
    fn centered_dither_can_fill_low_bits() {
        let mut rng = rand::rng();
        let mut residues = [false; 4];

        for _ in 0..1_000 {
            let dither = sample_centered_dither(&mut rng, 4);
            residues[dither.rem_euclid(4) as usize] = true;
        }

        assert!(residues.iter().all(|seen| *seen));
    }

    #[test]
    fn uniform_ubig_handles_power_of_two_upper_bound() {
        let mut rng = rand::rng();
        let upper = UBig::from(256u16);

        for _ in 0..100 {
            let value = sample_uniform_ubig_below(&mut rng, upper.clone()).unwrap();
            assert!(value < upper);
        }
    }

    #[test]
    fn constant_sampler_returns_single_allowed_offset() {
        let sampler =
            IntegerNormalSampler::for_offset_range(0, 0, (1_u64 << 53) as f64, "test").unwrap();
        let mut rng = rand::rng();

        assert_eq!(sampler.sample_offset(&mut rng, "test").unwrap(), 0);
    }

    #[test]
    fn uniform_range_sampler_can_reach_each_integer_in_range() {
        let sampler = IntegerNormalSampler::for_offset_range(-2, 2, 1_000_000.0, "test").unwrap();
        let mut rng = rand::rng();
        let mut seen = [false; 5];

        for _ in 0..2_000 {
            let offset = sampler.sample_offset(&mut rng, "test").unwrap();
            seen[(offset + 2) as usize] = true;
        }

        assert!(seen.iter().all(|seen| *seen));
    }

    #[test]
    fn karney_sampler_outputs_values_in_full_i64_range() {
        let sampler =
            IntegerNormalSampler::for_offset_range(i64::MIN as i128, i64::MAX as i128, 1.0, "test")
                .unwrap();
        let mut rng = rand::rng();

        for _ in 0..100 {
            let offset = sampler.sample_offset(&mut rng, "test").unwrap();
            assert!((i64::MIN as i128..=i64::MAX as i128).contains(&offset));
        }
    }

    #[test]
    fn tail_sampler_outputs_values_in_far_range() {
        let sampler = IntegerNormalSampler::for_offset_range(1_000, 10_000, 10.0, "test").unwrap();
        let mut rng = rand::rng();

        for _ in 0..100 {
            let offset = sampler.sample_offset(&mut rng, "test").unwrap();
            assert!((1_000..=10_000).contains(&offset));
        }
    }

    #[test]
    fn karney_sampler_matches_discrete_normal_distribution() {
        let sigma = PositiveRational::try_from_f64(2.0, "test").unwrap();
        let mut rng = StdRng::seed_from_u64(0xA11C_E5A7_5EED);
        let min = -6;
        let max = 6;
        let samples = 100_000;
        let mut counts = vec![0; (max - min + 1) as usize];
        let mut accepted = 0;

        while accepted < samples {
            let Some(offset) = sample_karney_discrete_gaussian_i128(&mut rng, &sigma).unwrap()
            else {
                continue;
            };
            if (min..=max).contains(&offset) {
                counts[(offset - min) as usize] += 1;
                accepted += 1;
            }
        }

        assert_discrete_normal_counts(&counts, min, 2.0, samples);
    }

    #[test]
    fn uniform_range_sampler_matches_truncated_distribution() {
        let sampler = IntegerNormalSampler::for_offset_range(0, 4, 100.0, "test").unwrap();
        assert!(sampler.uses_uniform_range());

        let mut rng = StdRng::seed_from_u64(0xB017_D15A_5EED);
        let samples = 150_000;
        let mut counts = [0; 5];

        for _ in 0..samples {
            let offset = sampler.sample_offset(&mut rng, "test").unwrap();
            counts[offset as usize] += 1;
        }

        assert_discrete_normal_counts(&counts, 0, 100.0, samples);
    }

    #[test]
    fn tail_sampler_matches_truncated_distribution() {
        let sampler = IntegerNormalSampler::for_offset_range(30, 44, 10.0, "test").unwrap();
        assert!(sampler.uses_tail());

        let mut rng = StdRng::seed_from_u64(0x7A11_D15A_5EED);
        let samples = 150_000;
        let mut counts = [0; 15];

        for _ in 0..samples {
            let offset = sampler.sample_offset(&mut rng, "test").unwrap();
            counts[(offset - 30) as usize] += 1;
        }

        assert_discrete_normal_counts(&counts, 30, 10.0, samples);
    }

    fn assert_discrete_normal_counts(
        counts: &[usize],
        min_offset: i128,
        stddev: f64,
        samples: usize,
    ) {
        let weights = discrete_normal_weights(counts.len(), min_offset, stddev);
        let weight_sum = weights.iter().sum::<f64>();
        let mut chi_square = 0.0;

        for (index, (&observed, weight)) in counts.iter().zip(weights.iter()).enumerate() {
            let expected = samples as f64 * weight / weight_sum;
            assert!(
                expected >= 20.0,
                "bin {index} expected count {expected} is too small for a stable statistical test"
            );

            let observed = observed as f64;
            let delta = observed - expected;
            chi_square += delta * delta / expected;

            let relative_error = delta.abs() / expected;
            assert!(
                relative_error <= DISTRIBUTION_MAX_RELATIVE_ERROR,
                "bin {index} observed {observed}, expected {expected:.2}, relative error {relative_error:.3}"
            );
        }

        let degrees_of_freedom = counts.len() as f64 - 1.0;
        let z_score = (chi_square - degrees_of_freedom) / (2.0 * degrees_of_freedom).sqrt();
        assert!(
            z_score <= DISTRIBUTION_CHI_SQUARE_Z_LIMIT,
            "chi-square {chi_square:.3} with {degrees_of_freedom} degrees of freedom has z-score {z_score:.3}"
        );
    }

    fn discrete_normal_weights(count: usize, min_offset: i128, stddev: f64) -> Vec<f64> {
        (0..count)
            .map(|index| {
                let offset = min_offset as f64 + index as f64;
                (-0.5 * (offset / stddev).powi(2)).exp()
            })
            .collect()
    }
}
