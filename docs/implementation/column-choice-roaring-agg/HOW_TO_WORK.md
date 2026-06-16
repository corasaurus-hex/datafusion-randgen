# Column Choice Roaring Aggregate HOW TO WORK

## Workflow Rules

1. Follow tests-first TDD with red/green/refactor for every step.
2. Start each step by defining or reviewing the abstractions and types needed for
   that step.
3. Write failing tests, compile checks, or source checks before implementation.
4. Run the targeted red command and confirm it fails for the expected reason.
5. Implement the minimum code needed to pass the red tests.
6. Refactor only after green.
7. Run the step's required verification commands before marking it done.
8. Update `CONTEXT.md` and all later step files when assumptions, APIs,
   commands, or constraints change.
9. Update user-facing docs and examples when behavior changes.
10. Use as many applicable test layers as practical: unit tests, integration
    tests, feature-gated scenario tests, source scans, rustdoc/API checks, and
    regression tests.
11. Do not create a git commit unless the user explicitly asks for commits. If
    commits are requested later, use one single-line commit message per completed
    step and no co-author or attribution footers.
12. Do not write continuation prompts into `CONTEXT.md` or step files; show them
    only in the final response for the step.
13. If implementation execution is requested and fresh worker agents are
    available, use fresh-agent sequential execution by default: exactly one
    worker per step, serially, with review between steps. If fresh agents are
    unavailable, use the same TDD/RGR loop locally.

## Feature Constraints

- The roaring aggregate is scoped to the column-choice feature area.
- Do not keep or rebuild the old file-path
  `randgen_column_choice(source_path, column_name[, null_probability])` API.
- Do not add shared cache state, SQL cache controls, or Tokio background cleanup.
- `randgen_roaring_agg(UInt32)` returns `Binary`.
- `randgen_roaring_agg(UInt64)` returns `LargeBinary`.
- `randgen_column_choice(Binary[, Float64])` returns `UInt32`.
- `randgen_column_choice(LargeBinary[, Float64])` returns `UInt64`.
- `randgen_roaring_agg` may return a valid serialized empty roaring value.
- `randgen_column_choice` must error when sampling from an empty non-null roaring
  value.

## Standard Step Loop

For each step:

1. Read this file, `CONTEXT.md`, the design doc, and the current step file.
2. Review the target API and types before editing.
3. Add tests/checks first.
4. Run the red command.
5. Implement the minimal green change.
6. Refactor.
7. Run targeted verification and any broader verification listed in the step.
8. Update `CONTEXT.md`, future step files, and user-facing docs if needed.
9. Report the changed files, verification, and next prompt.

