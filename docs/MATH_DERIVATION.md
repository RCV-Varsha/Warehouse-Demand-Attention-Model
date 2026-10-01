# Attention from first principles

Rows of X are token vectors. Q=XW_Q represents what each position seeks,
K=XW_K represents what each source matches, and V=XW_V represents what it returns.
Separate matrices decouple matching from returned content. In row-vector notation
$S_{ij}=x_i W_Q W_K^T x_j^T$, a learned bilinear relationship which need not be
symmetric. It is not necessarily a metric or a causal relationship.

For independent, zero-mean, unit-variance coordinates q_l and k_l,
$Var(\sum_{l=1}^{d_k}q_l k_l)=\sum_l E[q_l^2]E[k_l^2]=d_k$.
Division by $\sqrt{d_k}$ restores unit variance under those assumptions. Learned
projections of correlated inputs need not satisfy them; the paper experiment uses
independent samples while the training ablation measures actual logits.

$a_i=\exp(s_i)/\sum_j\exp(s_j)$ is unchanged after adding any common scalar c.
Choose c=-max(s) to avoid overflow. This does not undo saturation: the Jacobian
$\partial a_i/\partial s_j=a_i(\delta_{ij}-a_j)$ approaches zero as a row becomes
one-hot. Fully masked rows have no valid probability distribution; our explicit
extension returns a zero row and zero output, with zero gradient. Other rows sum
to one and $Y_i=\sum_j a_{ij}V_j$ is a convex combination inside the convex hull
of value rows. Temperature divides logits before normalization; lower values
sharpen the distribution.

| Tensor | Shape |
|---|---|
| X | B,n,d |
| W_Q,W_K | d,d_k |
| W_V | d,d_v |
| Q,K | B,n,d_k |
| V | B,n,d_v |
| S,S_scaled,A | B,n,n |
| Y | B,n,d_v |
| Warehouse input | B,24,5 |
| Embedding + learned positions | B,24,16 |
| Warehouse Q,K,V | B,24,16 |
| Last output / learned-query output | B,16 |
| MLP hidden, prediction | B,16 and B |

The learned pooling variant appends a learned query token; sequence length is then
25, with the last source column masked so the query retrieves only historical
values. No causal mask is necessary for forecasting a target after the entire
observed input window. A separate causal option is tested in the generic core.

## Full vector-Jacobian product

For scalar loss L, let G=dL/dY. For gradient checks specifically,
$L=\frac12\sum_{b,i,j}Y_{bij}^2$ so G=Y. Backpropagation gives

$$dV=A^T G,\qquad dA=G V^T,$$
$$dS'=A\odot(dA-\operatorname{rowsum}(dA\odot A)),$$
$$dS=dS'/(T\sqrt{d_k}),\qquad dQ=dS K,\qquad dK=dS^T Q,$$
$$dW_Q=\sum_b X_b^T dQ_b,\quad dW_K=\sum_b X_b^T dK_b,\quad dW_V=\sum_b X_b^T dV_b,$$
$$dX=dQW_Q^T+dKW_K^T+dVW_V^T.$$

Here dS' denotes the gradient of temperature-adjusted logits (T is temperature).
Omit sqrt(d_k) when scaling is disabled. Masked entries have zero derivative.
The manual implementation verifies the unmasked case at default temperature.

Central differences $(L(\theta+\epsilon)-L(\theta-\epsilon))/(2\epsilon)$
estimate each element of dL/dX and dL/dW, with truncation error O(epsilon^2)
until floating-point subtraction error O(machine-epsilon/epsilon) dominates.
Float32 reaches this regime sooner. The epsilon sweep should show a U-shaped
envelope, although individual coordinates and plateaus need not be monotone.
