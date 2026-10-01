# Synthetic warehouse

The exact equations and fixed constants are in the immutable [design](PHASE0_DESIGN.md)
and `configs/data_train.yaml`. The generator stores additive seasonal mean,
multiplicative event contributions and sampled noise separately. A linear level drift
represents modest growth; two harmonics represent working-day structure. Weekday
multipliers create weekly dependence. Nonnegative clipping represents nonnegative
order counts, although observations are continuous demand intensities, not integers.
Gaussian noise increases with the seasonal mean; lognormal spikes are positive and
right-skewed; multiplicative drops represent reduced capacity or demand.

Five independent random streams isolate onset counts, magnitudes, drop counts,
drop details and noise. Changing spike rate can change which sampled magnitudes
map to hours, but leaves noise/drop streams identical. Shift doubles noise
coefficients and spike median, triples onset rate. Individual-factor experiments
separate these interventions. This synthetic process does not validate real demand.

Raw chronological blocks have unused gaps, and each window remains within its
block. The demand scaler is fit only on the training block. ACF uses a full-series
mean and denominator for descriptive visualization only; it is not a model feature.

Generated evidence: `results/dataset.json`, `dataset_components.png`,
`dataset_acf.png`, `dataset_histogram.png`, `dataset_events.png`.
