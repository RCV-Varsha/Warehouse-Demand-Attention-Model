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
Not run yet.
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
