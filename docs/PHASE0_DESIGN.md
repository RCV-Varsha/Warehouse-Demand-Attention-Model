# Phase 0: design and hypotheses (immutable pre-experiment record)

Title: From First Principles: Warehouse Demand Attention Model.
This document specifies planned constants, not measured results. Outcomes belong in
EXPERIMENTS.md. The pasted user specification takes precedence over the broader PDF.

## Formulation
Let y_t be hourly orders, with zero-based t starting at Monday midnight.
Given x_{t-23:t} in R^{24 x 5}, predict y_{t+1}. Features are standardized demand,
sin/cos of hour and sin/cos of weekday. Minimize mean squared standardized error
on training targets; select checkpoints by validation MAE in orders.

## Generator and assumptions
T=8760, base=100, g=0.10. level(t)=100(1+0.10t/T).
daily(t)=0.30 cos(2 pi (h-12)/24)-0.12 cos(4 pi (h-12)/24):
a broad working-day peak with two shoulders, not a claim of exact peak hours.
weekday multipliers (Monday first)=[1.10,1.05,1.05,1.10,1.00,0.70,0.50].
mu(t)=level(t)(1+daily(t))m[d(t)]. Each hourly spike count is Poisson(0.12/24),
each independent magnitude is LogNormal(log(0.8),0.4); contributions sum and
decay as exp(-age/3), truncated after 24 hours. Drop counts are Poisson(0.04/24);
each reduces demand to Uniform(0.2,0.5) for an integer Uniform{2,...,5} hours.
Overlapping drops multiply. y=max(0,mu*(1+spike)*drop+noise), with independent
noise Normal(0,(3+0.06mu)^2). Store every component and onset/active/post flags.
Post-spike means the first three hours after an onset, and can overlap active spikes.

Daily/weekly phases are stationary; the level drifts mildly, so the full process is
not stationary. Events are independent of calendar and noise. Unannounced onsets
cannot be predicted from the window. Calendar encodings identify weekday even
though a 24-hour window cannot directly observe a complete week. Lag 24 and lag 168
should correlate; the latter is unavailable to this model. No oracle event feature
enters training. Ground-truth components are diagnostics only.

## Model and controls
Single head, embedding dimension 16, learned positional vectors, last-position
query and a small MLP head. Predict standardized demand directly, without a
last-value residual. Positions distinguish lag 24 from recency. Ablate a learned
pooling query by appending it to the sequence, masking its value out of retrieval.
Toy associative recall: four unique keys sampled from eight keys, independent
four-class values; a final query asks for one stored value. Token features separate
key, value and query identity; a mean-pool linear control cannot bind values to keys.
Use fresh training batches and a fixed independently generated validation set.

## Falsifiable hypotheses (do not edit after commit)
- H1: For d_k >= 64, unscaled toy attention has lower mean initial entropy and lower final accuracy than scaled attention; independent unit-variance dot products have scaled variance within 10% of 1.
- H2: Warehouse attention improves test MAE by at least 10% over last value and MA-24, and is within 15% of both seasonal naive and ridge MAE.
- H3: Mean attention mass on lag 24 plus lags 1, 2 and 3 exceeds 25% on warehouse test windows.
- H4: Combined shift increases every model's MAE by at least 20%; attention's relative degradation on spike hours exceeds its degradation on normal hours.
- H5: Spike-onset MAE is at least twice normal-hour MAE, and mean signed error in hours 1 through 3 after onset is positive.
- H6: Toy attention exceeds 90% validation accuracy and the mean-pool control stays below 60%.
- H7: Rank-2 Q/V adapters use fewer trainable parameters than full fine-tuning and improve shifted-test MAE over no adaptation.

## Failure modes
Unpredictable onset misses; post-spike overprediction; multiplicative drops confused
with weekends; Friday-to-Saturday discontinuities; unscaled-logit saturation;
train/test scaler leakage; nonfinite all-masked softmax rows. Explicitly test the
last two rather than relying on good-looking forecasts.

## Evaluation fixed in advance
Chronological raw blocks: 70/15/15, with a 24-hour unused raw gap before validation
and before test. Windows and targets stay entirely within their block. Fit mean/std
and ridge feature scaling only on the training block. MAE is primary (staffing
units), RMSE penalizes large misses, MASE divides MAE by training seasonal-naive
absolute error at lag 24. Report normal, spike, post-spike, drop, weekday, weekend
slices with counts (overlap is explicit). Seeds 0..4 vary both generator and training;
paired models share data and seeds. Report mean and sample std across five seeds;
this is descriptive uncertainty, not a formal significance claim. A difference
smaller than its paired-seed standard deviation is inconclusive.

Last value should track short continuity but miss calendar transitions; MA-24
smooths noise and loses intraday structure; seasonal naive uses daily structure but
misses weekday changes; ridge on standardized lags plus target calendar features
should be strong for smooth periodic demand but cannot foresee random onsets.

## Planned experiments and resource budget
Float64 central differences over every element of small X/W matrices, epsilon
10^-2 through 10^-8, plus float32 and independent manual backprop. Stable softmax
returns zero for all-masked rows (there is no probability distribution there).
Toy LR and Q/K initialization sweeps; scaled/unscaled d_k={4,16,64,256}, five seeds.
d_model remains <=64; only the explicit key-width ablation uses d_k=256.
Warehouse scaled/unscaled and last/learned pooling comparisons use five seeds.
Training budgets and optimizer settings will be recorded in YAML before each run;
early stopping uses validation data only. No tuning on test outcomes.

Shift doubles both noise coefficients, doubles spike magnitudes and triples spike
rate, keeping seasonality unchanged. Test combined and each single-factor change
with paired random streams. Freeze training scalers. Larger inputs can increase
logit magnitude and saturation. Adaptation uses only an early shifted-data slice,
with disjoint shifted validation/test blocks; compare no adaptation, full update,
rank-2 Q/V adapters. Investigate the largest absolute test error and separately the
largest error with an input spike; replace input spike demand by de-spiked demand
with the same noise/drop components, measure prediction and attention changes.
This intervention tests model sensitivity, not real-world causal identification.

JAX compares outputs/gradients in float64; profiling records actual CPU time and
allocator peak memory separately from deterministic summary statistics. Paper
reproduction tests the independent q/k variance premise, not the entire paper.
All numerical observations must be saved under results. No outcome is assumed.
