# Seven-part reproducible demo

After setup and `python -m experiments.run_all --seed 0`, run
`python -m scripts.demo --seed 0` or `bash scripts/demo.sh --seed 0`.
The demo reads saved results and prints the relevant plot path for every part.
It does not silently retrain or replace an experiment.

| Part | Inspect | Regenerate individually |
|---|---|---|
| Attention walkthrough | src/wdam/attention/core.py, printed shape table | python -m scripts.demo --seed 0 |
| Gradient verification | gradcheck.json, gradcheck_eps_sweep.png | python -m experiments.gradcheck --seed 0 |
| Toy learning | toy_task.json, toy_curves.png, toy_attention.png | python -m experiments.toy_task --seed 0 |
| Warehouse vs baselines | warehouse.json, warehouse_comparison.png | python -m experiments.warehouse_train_eval --seed 0 |
| Predicted vs observed ablation | hypotheses.json, ablation_scaling.png | python -m experiments.ablation_scaling --seed 0 |
| Distribution shift | generalization_shift.json/png | python -m experiments.generalization_shift --seed 0 |
| Failure intervention | failure_analysis.json, failure_counterfactual.png | python -m experiments.failure_analysis --seed 0 |

Shift, failure, attention visualization and adaptation require the trained checkpoint
from warehouse_train_eval. Table filling requires the full result set:
`python -m scripts.fill_tables --seed 0`. Explain what each tested gradient means,
which predeclared conditions failed, and why attention weights do not establish
causality. The candidate should complete the reflection and AI log personally.
