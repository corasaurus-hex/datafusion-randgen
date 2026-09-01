use datafusion_common::{Result, exec_err};
use rand::{Rng, RngExt};
use rand_distr::Normal;

const MAX_REJECTION_ATTEMPTS: usize = 4_096;

pub(crate) enum RoundedIntegerNormalSampler {
    Constant(i128),
    Rejection {
        min: i128,
        max: i128,
        normal: Normal<f64>,
    },
}

impl RoundedIntegerNormalSampler {
    pub(crate) fn new(min: i128, max: i128, mean: i128, stddev: f64, name: &str) -> Result<Self> {
        if min > max {
            return exec_err!("{name} requires min <= max");
        }
        if !stddev.is_finite() || stddev <= 0.0 {
            return exec_err!("{name} requires finite stddev > 0");
        }
        if min == max {
            return Ok(Self::Constant(min));
        }

        let normal = Normal::new(mean as f64, stddev).map_err(|error| {
            datafusion_common::DataFusionError::Execution(format!(
                "{name} invalid normal distribution parameters: {error}"
            ))
        })?;

        Ok(Self::Rejection { min, max, normal })
    }

    pub(crate) fn sample<R: Rng + ?Sized>(&self, rng: &mut R, name: &str) -> Result<i128> {
        match self {
            Self::Constant(value) => Ok(*value),
            Self::Rejection { min, max, normal } => {
                for _ in 0..MAX_REJECTION_ATTEMPTS {
                    let value = rng.sample(*normal).round();
                    if !value.is_finite() {
                        continue;
                    }

                    let value = value as i128;
                    if (*min..=*max).contains(&value) {
                        return Ok(value);
                    }
                }

                exec_err!(
                    "{name} could not sample an in-range rounded normal value after {MAX_REJECTION_ATTEMPTS} attempts; widen the range or adjust mean/stddev"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_range_returns_the_only_value() {
        let mut rng = rand::rng();
        let sampler = RoundedIntegerNormalSampler::new(7, 7, 100, 1.0, "test").unwrap();

        assert_eq!(sampler.sample(&mut rng, "test").unwrap(), 7);
    }

    #[test]
    fn invalid_stddev_errors() {
        let result = RoundedIntegerNormalSampler::new(0, 10, 5, 0.0, "test");

        assert!(result.is_err());
    }

    #[test]
    fn far_tail_errors_after_retry_cap() {
        let mut rng = rand::rng();
        let sampler = RoundedIntegerNormalSampler::new(0, 1, 1_000, 1.0, "test").unwrap();

        assert!(sampler.sample(&mut rng, "test").is_err());
    }
}
