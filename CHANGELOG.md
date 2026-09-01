# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/corasaurus-hex/datafusion-randgen/releases/tag/v0.1.0) - 2026-09-01

### Other

- Upgrade dependencies and configure releases
- Configure dprint formatting
- Remove proofs and update dependencies
- Prepare release validation
- Clean up documentation
- Pivot column choice to roaring aggregate
- Add Arrow IPC column choice support
- Make nullable generation safe
- Add Parquet column choice UDF
- Refactor UDF helpers
- Improve documentation
- Improve SQL literal ergonomics
- Simplify integer normal sampling
- Add UDF property stress and soak tests
- Dual license crate
- Clean release package checks
- Implement hybrid integer normal sampling
- Preserve integer means in normal UDFs
- Add uint64 and integer normal UDFs
- Format README table
- Improve docs and UDF coverage
- Configure cargo-deny advisories
- Document public API
- Remove global UTF8 alphabet cache
- Hoist scalar float uniform validation
- Specialize numeric randgen choice outputs
- Use non-null output buffers in randgen UDFs
- Specialize scalar randgen UDF arguments
- Simplify randgen UDF plumbing
- Refine UDF APIs and performance
- Fix an error condition in choice
- Reject overflowing float ranges
- Update rand dependencies
- Remove missing readme manifest entry
- Remove empty macros module
- Use just and nextest in CI
- Register all generators
- Add timestamp millisecond generator
- Add date32 generator
- Add choice generator
- Add UTF-8 generator
- Add boolean generator
- Add float64 generators
- Harden int64 uniform generation
- Upgrade to DataFusion 53
- Add nextest tooling
- Ignore temporary planning files
- Handle nulls when generating int64
- add default impl
- Refactor test helpers
- randgen_int64_uniform
- Initial commit
