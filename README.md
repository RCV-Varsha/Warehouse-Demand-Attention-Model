# From First Principles: Warehouse Demand Attention Model

A CPU-only, single-head investigation of associative recall and next-hour synthetic
warehouse demand. Attention uses raw tensor projections, matrix products and a
manually normalized exponential. No high-level attention implementation is used.
All comparisons include unfavorable outcomes and retain the pre-experiment design.

## Quickstart

Python **3.11** is required. Exact direct dependency versions are in
`requirements.txt` and `requirements-jax.txt`; the executed environment is recorded
in `results/environment.json`. Install both files to reproduce all extensions.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-jax.txt
python -m pip install --no-build-isolation -e .
python -m pytest -q
python -m experiments.run_all --seed 0
python -m scripts.demo --seed 0
```

The requested direct script command is also supported:
`python experiments/run_all.py --seed 0`. Run commands from this repository root.
Every entrypoint accepts `--seed`. Runner seed is the first of the five consecutive
seeds specified by offsets in `configs/experiments.yaml`; demo uses saved results
and does not retrain. `make all`, `make test`, `make data`, `make demo`, `make clean`
and `make reset` are convenience targets. Reset removes generated data/results and
regenerates them; it never removes source, the virtual environment or Git history.
On Windows without GNU Make use `python -m scripts.manage reset --seed 0` and then
`python -m experiments.run_all --seed 0`.
Set `WDAM_MAKE` to a native Make executable if the reproduction checker should use
a particular Windows build. It records the actual executed commands.

<!-- AUTO:RUNTIME -->
Measured experiment-stage CPU wall time: 4.74 minutes on the environment in `results/environment.json`. Source: `results/runtime.json`. Download/setup and final test/report overhead are excluded; timings vary.
<!-- /AUTO:RUNTIME -->

## Results
<!-- AUTO:SUMMARY -->
| Model | MAE mean ± sample std | RMSE mean ± std | MASE mean ± std | Attention minus model MAE | Difference % ± std |
| --- | --- | --- | --- | --- | --- |
| attention | 9.212 ± 0.364 | 13.724 ± 1.178 | 0.365 ± 0.016 | 0.000 | 0.00 ± 0.00 |
| last_value | 12.889 ± 0.271 | 17.641 ± 0.912 | 0.510 ± 0.011 | -3.677 | -28.54 ± 2.00 |
| learned_pool | 9.365 ± 0.474 | 13.805 ± 1.355 | 0.371 ± 0.017 | -0.153 | -1.54 ± 3.88 |
| moving_average | 26.656 ± 0.885 | 34.166 ± 1.744 | 1.055 ± 0.029 | -17.444 | -65.43 ± 1.14 |
| ridge | 10.370 ± 0.323 | 14.776 ± 1.073 | 0.411 ± 0.011 | -1.158 | -11.16 ± 2.74 |
| seasonal_naive | 27.749 ± 1.926 | 38.211 ± 3.067 | 1.098 ± 0.064 | -18.537 | -66.72 ± 1.72 |
| unscaled | 9.292 ± 0.261 | 13.818 ± 1.192 | 0.368 ± 0.009 | -0.081 | -0.87 ± 2.63 |

Source: `results/warehouse.json`. Negative differences favor attention.
<!-- /AUTO:SUMMARY -->

See [all experiments](docs/EXPERIMENTS.md) for seed variability, hypotheses that failed,
scaling/aggregation controls, shift attribution, manual gradients and extensions.
The deterministic `results/summary.json` excludes wall timings, allocator traces,
and environment strings. Full precision measured values remain in result JSONs;
generated Markdown rounds for readability. `scripts.verify_reproduction` runs reset
and a second full regeneration and checks byte-identical summary files.

## Configuration and data

- `configs/data_train.yaml`: all generator parameters; trend, two daily harmonics,
  weekday multipliers, heteroscedastic Gaussian noise, Poisson/lognormal spikes,
  multiplicative drops, and event durations.
- `configs/data_shift.yaml`: same seasonal structure, doubled noise and spike
  magnitude, tripled spike rate. Single-factor variants derive from these files.
- `configs/toy.yaml`: key/value task, training steps, batch size, validation size,
  optimizer rate, embedding size and accuracy threshold.
- `configs/warehouse_model.yaml`: model size, optimizer, patience, batch/epochs,
  ridge penalty, adapter rank and adaptation slice size.
- `configs/experiments.yaml`: seed offsets, key widths, LR/init grids, profiling
  lengths/repetitions, variance reproduction sample count.

Generate independently with:
`python -m wdam.data.generator --config configs/data_train.yaml --seed 0 --out data/`.
Reset/regenerate overwrites deterministic files for that config/seed. Splits are
chronological, with raw gaps and no cross-block windows. Demand and ridge scalers
use training data only. Checkpoints and raw arrays are regenerated in ignored
`data/`; small reviewer-facing results are committed.

## Repository and documentation

`src/wdam/attention/`: core, softmax, manual backward, LoRA and JAX.
`src/wdam/data/`: generator, scaler and windows.
`src/wdam/models/`: toy and forecasting modules.
`experiments/`: standalone experiments, automated runner.
`tests/`: mathematical, data, baseline and deliberate-mutation checks.
`scripts/`: table filling, reset/reproduction checks and demo.

- [Immutable pre-experiment design](docs/PHASE0_DESIGN.md)
- [Math and shape derivation](docs/MATH_DERIVATION.md)
- [Dataset](docs/DATASET.md)
- [Experiments and hypothesis outcomes](docs/EXPERIMENTS.md)
- [Failure investigation](docs/FAILURE_ANALYSIS.md)
- [Real debugging record](docs/DEBUGGING_LOG.md)
- [Candidate reflection draft](docs/REFLECTION.md)
- [AI assistance disclosure](docs/AI_ASSISTANCE_LOG.md)
- [Seven-part demo](docs/DEMO.md)

## Requirement traceability

| Requirement / phase | Implementation and evidence |
|---|---|
| 0: design before experiments, equations, assumptions, hypotheses, failures, evaluation, baseline expectations, shift plan | docs/PHASE0_DESIGN.md; first Git commit |
| 1: synthetic environment, reset, components, ACF, event/histogram plots | data/generator.py; data/windows.py; experiments/dataset.py; configs/data_*.yaml; docs/DATASET.md |
| 2: Q/K/V, scores, scaling, stable softmax, shapes, causal mask, temperature | attention/core.py; attention/softmax.py; tests/test_attention.py |
| 3: interpretation, variance, Jacobian, convexity, shapes, backward | docs/MATH_DERIVATION.md |
| 4: central differences, epsilon sweep, float32/64, autograd and manual checks | experiments/gradcheck.py; attention/manual_backward.py; results/gradcheck.json |
| 5: overflow, all-masked rows, saturation, precision | experiments/numerical_stability.py; results/numerical_stability.json |
| 6: associative recall, training/validation, accuracy, mean-pool control, heatmap | models/toy_model.py; experiments/toy_task.py; results/toy_task.json |
| 7: learning-rate/init sweep, gradient/entropy/weight traces | training.py; experiments/training_dynamics.py; results/training_dynamics.json |
| 8: key-width scaling ablation, seeds, thresholds, variance, mathematical analysis | experiments/ablation_scaling.py; docs/EXPERIMENTS.md; results/ablation_scaling.json |
| 9: embedding, positions, raw attention, aggregation ablation, head, early stopping | models/warehouse_model.py; experiments/warehouse_train_eval.py; results/warehouse.json |
| 10: last value, MA, seasonal naive, ridge, paired comparisons | baselines.py; results/warehouse.json |
| 11: chronological gaps, train-only scaling, MAE/RMSE/MASE and slices | data/windows.py; metrics.py; tests/test_data_metrics.py |
| 12: combined and single-factor shifts, frozen scaler, all baseline comparisons | experiments/generalization_shift.py; results/generalization_shift.json |
| 13: genuine failure, event-relative errors, spike weights, counterfactual | experiments/failure_analysis.py; docs/FAILURE_ANALYSIS.md |
| 14: individual, lag-average, target-hour weights, explanation caveat | experiments/attention_viz.py; results/attention_viz.json |
| 15: shared core, toy-to-warehouse interpretation | docs/EXPERIMENTS.md, Cross-problem integration |
| 16: shape/softmax/convexity/equivariance/causal/gradient/data/leakage/baseline/metric/mutation tests | tests/; docs/DEBUGGING_LOG.md; results/verification.json |
| 17: automated regeneration, seeds, environment, hashes, tables, reset | experiments/run_all.py; scripts/fill_tables.py; scripts/verify_reproduction.py; Makefile |
| 18: manual backward, LoRA, JAX, profiling, paper, temperature/rank | attention/manual_backward.py, lora.py, jax_core.py; experiments/ext_*.py; numerical_stability.py |
| 19: README, debugging, reflection, AI log, reproducible demo | docs/; scripts/demo.py; scripts/demo.sh |
| No fabricated results, no forbidden APIs, CPU determinism, small models | results/; scripts/audit.py; utils.py; tests/ |
| All MUST/SHOULD/NICE outcomes, per-phase commits, safe GitHub delivery | phase Git history; results/verification.json; results/reproduction.json |

Source paths in the matrix are relative to `src/wdam` unless prefixed with
`experiments`, `scripts`, `tests`, `docs`, `configs` or `results`.

## Limits and candidate responsibility

No real warehouse evidence is used. Random onset prediction is information-limited;
attention weights alone are not causal explanations. This is an AI-assisted
submission draft. Candidate-personal reflection and understanding columns remain
explicit TODOs rather than fabricated personal claims.
