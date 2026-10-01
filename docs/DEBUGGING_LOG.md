# Actual development debugging record

This log separates observed defects from deliberately injected mutations. No
candidate participation or unobserved model-training bug is implied.

## Generated-data ignore rule hid source code
Initial implementation: `.gitignore` used the requested `data/` pattern.
Unexpected result: phase verification passed, but `git add src/wdam/data` failed
because Git treated the source package as ignored generated data.
Hypothesis: the unanchored ignore pattern applies to every directory named data.
Diagnostic: Git explicitly reported `src/wdam/data` as ignored during phase staging.
Correction: anchor generated data to repository root with `/data/`.
Verification: the phase-one commit includes generator.py and windows.py; later
commits preserve only generated artifacts under the ignored root data directory.

## Runtime and repository ownership
Initial implementation: use default package TLS roots and invoke the environment
inside the sandbox. Unexpected result: PyPI certificate validation failed and the
virtual environment could not launch the interpreter in the protected user
installation. Diagnostic: errors were UnknownIssuer and inability to create the
underlying Python process. Correction: use system TLS certificates and explicitly
approved environment execution. Verification: dependency installation and executed
gradient/data experiments succeeded. These were environment issues, not numerical
bugs.

Git additionally rejected repository ownership when switching between the sandbox
and user accounts. A per-command safe.directory for this exact repository resolved
the mismatch without weakening global ownership checks.

## Source review correction before execution
The initial repository-root calculation walked one parent too far from utils.py.
Inspection of the actual path caught this before experimental execution. Corrected
to the repository parent of src; generated files then appeared under this repository.
This is a review finding, not a fabricated runtime failure.

## Deliberate mathematical mutations
`tests/test_attention.py` explicitly constructs broken alternatives and asserts
that the same mathematical contract rejects them:

- Wrong softmax axis: row sums fail normalization on nonsymmetric logits.
- Missing key transpose: a non-square sequence/key shape rejects Q @ K.
- Missing scaling: scaled-score equality fails against the expected square-root
  factor.

These are explicit local mutation probes, not a claim of exhaustive automated
mutation coverage. `results/verification.json` records each named test outcome and
the full pytest summary. The suite also checks causal masking, convex bounds,
permutation equivariance, central differences, manual gradients, train-only scaling,
chronological gaps, paired generator streams, baselines and hand-computed metrics.
