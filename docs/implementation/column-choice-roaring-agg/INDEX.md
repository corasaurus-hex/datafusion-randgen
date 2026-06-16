# Column Choice Roaring Aggregate Plan

## Design

- [Design](../../column-choice-roaring-agg-design.md)
- [How To Work](HOW_TO_WORK.md)
- [Context](CONTEXT.md)

## Steps

1. [API and Registration Contracts](steps/01-api-registration-contracts.md)
2. [Roaring Aggregate Accumulator](steps/02-roaring-aggregate-accumulator.md)
3. [Aggregate-Value Sampler](steps/03-aggregate-value-sampler.md)
4. [Feature Integration Tests](steps/04-feature-integration-tests.md)
5. [Remove Cache Surface](steps/05-remove-cache-surface.md)
6. [Docs and Examples](steps/06-docs-and-examples.md)
7. [Final Hardening](steps/07-final-hardening.md)

## Execution

Execute steps serially. Each step must use tests-first red/green/refactor and
must update `CONTEXT.md` before completion if behavior, assumptions, commands, or
risks change.

