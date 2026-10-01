# Genuine failure investigation

## Scenario and expected behavior
Select the maximum absolute held-out error of the primary model, then separately
select the maximum error whose historical window includes an active spike. This
second selection makes the counterfactual test meaningful even if the largest
overall error has an unannounced onset outside the input. Selection uses the
fixed seed's held-out predictions and is descriptive, not a new tuning dataset.
We expect calendar demand plus predictable event decay, but cannot foresee
independent random onset hours from past demand alone.

## Actual behavior and evidence
<!-- AUTO:FAILURE -->
| Case | Hour | Target is onset | Actual | Predicted | Error | Input spike attention mass | Counterfactual delta prediction | Attention L1 delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| worst_overall | 7816 | True | 261.8182520710774 | 130.6433868408203 | -131.17486523025707 | 0.9014344811439514 | 4.986151695251465 | 0.10188737511634827 |
| worst_with_input_spike | 7816 | True | 261.8182520710774 | 130.6433868408203 | -131.17486523025707 | 0.9014344811439514 | 4.986151695251465 | 0.10188737511634827 |

For the selected historical-spike case, removing the input spike raises the forecast by 4.986 orders, while the original absolute error is 131.175. This is inconsistent with a simple story that the past spike only pushes the prediction upward. The target is a new unpredictable onset; the counterfactual does not remove that target event. Across seeds, post-onset mean signed bias is -22.685 orders (results/hypotheses.json); underprediction contradicts the positive-bias component of H5.
<!-- /AUTO:FAILURE -->

Source: `results/failure_analysis.json`. The signed error is prediction minus truth.
The JSON contains original/de-spiked input demand, all attention weights, onset
flags and selected input spike positions. `failure_event_error.png` reports errors
by time since onset and the JSON supplies counts. `failure_counterfactual.png`
shows the historical spike and the corresponding weight changes.

## Explanation and intervention
The generator creates statistically independent onset events, so an onset miss is
an information limitation rather than necessarily an optimization defect. Historical
spikes can bias a model trained on mostly normal hours. To test whether the observed
spike affects the forecast, replace only historical spike demand with
max(0, seasonal mean * drop multiplier + the original noise). Keep calendars,
positions and the scaler fixed. Report the prediction delta and attention L1 delta.
A large spike weight alone is insufficient evidence: the perturbation can change
keys, queries and values simultaneously. The intervention is oracle-assisted
diagnostics, never a training or deployable feature.

## Potential improvement and limitations
Consider robust losses or a learned event-decay state to reduce post-event bias;
evaluate against the existing fixed split and a new held-out generator seed.
Known advance operational event flags could make some onsets predictable, but
the current generator intentionally provides no such information. No fix attempt
has been run, and none is claimed to work. Reported failure selection is one seed;
the aggregate onset/post-spike slices across seeds provide broader context.
