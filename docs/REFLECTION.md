# Reflection: evidence-backed draft for candidate review

<!-- CANDIDATE: review and rewrite in your own words -->
This is an AI-authored technical draft. It does not claim the candidate has performed
or understood the work. Personal answers below require review and replacement.

1. **Initial expectations.** The committed design predicted useful daily-lag
   retrieval, scaled-attention robustness, and event-related failures. These are
   recorded before experimental code/results in `PHASE0_DESIGN.md`.
2. **Correct hypotheses.** Use the generated decisions below and inspect each
   conjunction in `results/hypotheses.json`; describe why the evidence supports it.
3. **Wrong hypotheses.** Any "Not supported in full" entry below must remain visible.
   A partially correct mechanistic premise does not rescue a failed quantitative
   prediction. TODO (candidate): explain which failed prediction changed your view.
4. **Surprises.** TODO (candidate): select a specific observed outcome from
   `results/ablation_scaling.json` or `warehouse.json`; do not invent an emotional
   response on behalf of the candidate.
5. **Hardest implementation problem.** TODO (candidate): review the actual debugging
   log and identify what you personally found difficult. Technical pressure points
   include all-masked gradients and split boundary indexing.
6. **Failure investigated.** The largest held-out error and the largest error with
   an input spike are selected in `failure_analysis.json`; a de-spiked intervention
   tests prediction/attention sensitivity. See `FAILURE_ANALYSIS.md`.
7. **What is understood better.** TODO (candidate): explain the distinction between
   max-subtraction stability and softmax saturation using `numerical_stability.json`.
8. **Remaining uncertainty.** Synthetic demand cannot establish real warehouse
   usefulness. Mean attention does not identify causality; limited seeds and rare
   onset counts limit inference. See sliced counts in `warehouse.json`.
9. **Where AI helped.** Codex generated the implementation, experiments, test suite
   and documentation. See `AI_ASSISTANCE_LOG.md` for verification responsibilities.
10. **AI suggestions corrected/rejected.** TODO (candidate): describe your own review.
    Agent-discovered corrections are in `DEBUGGING_LOG.md`, and are not candidate
    modifications. No unrecorded candidate correction is asserted.
11. **One more day.** Proposed, not run: evaluate a robust loss and event-decay model
    on additional independent seeds and test whether counterfactual sensitivity
    predicts error reduction. TODO (candidate): choose and justify a priority.

<!-- AUTO:OUTCOMES -->
| Hypothesis | Decision |
| --- | --- |
| H1 | Not supported in full |
| H2 | Not supported in full |
| H3 | Supported |
| H4 | Supported |
| H5 | Not supported in full |
| H6 | Supported |
| H7 | Supported |
<!-- /AUTO:OUTCOMES -->
