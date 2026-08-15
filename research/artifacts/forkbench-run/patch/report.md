# ForkBench selective extraction report

Validation outcome: PASS
Sealed holdout outcome: PASS
Selected target retention: 1.000
Unselected behavior preservation: 1.000

The claims above cover only the executed probes, generation policy, and search budget.

## Negative findings

- The initial extracted candidate exposed 2 target and 7 guard counterexample(s) before recompilation.
- Worst observed prompt-level base KL was 1.956121; exact generated controls passed, but this is measurable distributional drift.
