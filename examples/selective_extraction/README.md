# Selective extraction configuration

Selective extraction requires real models and a real behavioral-diff bundle, so
this directory does not ship synthetic witnesses or an edited transcript. The
configuration records the validation thresholds used by the typed
`extract_behavior_cluster` API.

The caller must load a trusted adapter, base and multi-change target models,
then pass selected and nonselected `DifferenceWitness` objects from an executed
diff:

```python
from modelpact.compiler.extract import extract_behavior_cluster

evidence = extract_behavior_cluster(
    adapter,
    base_model,
    target_model,
    selected_witnesses,
    nonselected_witnesses,
    maximum_selected_kl=0.05,
    maximum_nonselected_base_kl=0.02,
)
if not evidence.validation_passed:
    raise SystemExit(2)
```

`selected_teacher_kl` measures transfer on the selected empirical domain;
`nonselected_base_kl` measures whether nonselected target changes leaked into
the patch. Neither metric proves that a cluster is a complete semantic concept.
