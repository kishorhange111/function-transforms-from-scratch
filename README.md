# function-transforms-from-scratch

[![CI](https://github.com/kishorhange111/function-transforms-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/function-transforms-from-scratch/actions/workflows/ci.yml)

JAX's composable function transformations, written from scratch on numpy in about 1,200
lines:
- `jvp`;
- `vjp` and `grad`, with reverse mode derived as **linearize + transpose**;
- `vmap`;
- `jit`, with CSE, DCE and code generation;
- `jacfwd`, `jacrev` and `hessian`.

Every transform composes with every other: `grad(grad(f))`, `vmap(grad(f))`,
`jit(vmap(grad(f)))`, `jacfwd(jacrev(f))`, `grad` of a jitted function. Gradients,
Hessians, Jacobians and per-example gradients match `torch.func` to within 1e-9 relative
error on every test function.

```python
import minijax as mj
from minijax import np as jnp

loss = lambda W, x, y: jnp.sum((jnp.tanh(x @ W) - y) ** 2)
per_example = mj.jit(mj.vmap(mj.grad(loss), in_axes=(None, 0, 0)))  # (B, *W.shape)
hvp = lambda f, x, v: mj.jvp(mj.grad(f), (x,), (v,))[1]  # forward-over-reverse
```

## Design ([`src/minijax`](src/minijax))

**Primitives and traces** ([`core.py`](src/minijax/core.py)). There are 19 primitives:
- elementwise arithmetic, `exp`, `log`, the trig functions, `max` and comparisons;
- `select`;
- `reduce_sum` and `reduce_max`;
- `broadcast`, `reshape` and `transpose`;
- a two-operand `einsum`.

Each primitive has a numpy implementation and a shape rule. A transformation is a
*trace* on a stack. `bind` sends each primitive to the trace of its highest-level
argument and lifts the other arguments into that trace. So nested transforms interleave
correctly. A tracer whose transform has exited raises an error instead of computing
silently. numpy broadcasting is done in the user API ([`ops.py`](src/minijax/ops.py))
as explicit `reshape` and `broadcast` primitives, so the transforms handle it like
anything else.

**Forward mode** ([`jvp.py`](src/minijax/jvp.py)). A tracer carries (primal, tangent).
Each JVP rule is written with primitives, never raw numpy, so tangent code can be
transformed again. Ties in `max` and `reduce_max` share the gradient equally.

**Batching** ([`vmap.py`](src/minijax/vmap.py)). A tracer carries (value, batch axis).
- Elementwise rules move the batch axis to the front.
- Reductions and reshapes shift their axes.
- `einsum` gives the batch axis a fresh index letter. One batched matmul rule covers
  `vmap` of matrix-vector products, outer products and contractions.

**Reverse mode is not a second autodiff system** ([`staging.py`](src/minijax/staging.py)).
It is forward mode, partially evaluated, then transposed, as in JAX:
1. `linearize` runs `jvp` with *staged* tangent inputs, one trace level below the jvp.
   Primal operations never touch a staged value, so they run immediately. Every tangent
   operation does, so it is recorded. The result is a jaxpr that is linear in the
   tangents, with the residuals it needs as constants. A test checks that it contains no
   `exp`, `sin` or `tanh`: only linear primitives.
2. `transpose` walks that jaxpr backwards. Only linear primitives have transpose rules:
   - `add` fans the cotangent out;
   - `broadcast` and `reduce_sum` are each other's transpose;
   - `mul` multiplies by its constant factor;
   - `einsum` contracts the cotangent with the other operand, then broadcasts over
     indices only the wanted operand had.

   Cotangents accumulate with `add`. Tested with the dot-product identity
   ⟨ct, J t⟩ = ⟨Jᵀ ct, t⟩.
3. `vjp` is linearize then transpose, and `grad` is `vjp` with cotangent 1. Since the
   backward pass binds primitives too, it can be differentiated, batched or staged again.
   That is all `grad(grad)`, `jacrev` (= `vmap` over `vjp`) and `jit(grad)` need.

**jit** stages the function once per input shape and type, then optimises it:
- common-subexpression elimination merges `sin(x) · sin(x)` into one `sin`;
- dead-code elimination drops the cotangents nobody asked for.

It compiles the result to straight-line Python, one numpy call per equation. On
concrete inputs it runs that code with no tracing at all. Under an outer transform it
replays the jaxpr through `bind`, so `vmap(jit(f))` and `grad(jit(f))` still work. A
jaxpr that closes over an outer tracer is not cached, because it is only valid for that
call.

## Results ([`bench.py`](src/minijax/bench.py))

These are CPU timings on a shared machine. The benchmark ran three times (9, 9 and 25
repetitions per timing; [results/bench.md](results/bench.md) has the last), so ranges are
over those three runs.

**Per-example gradients**, for an MLP 64 → 256 → 10 in float64:

| batch | loop of `grad` | `vmap(grad)` | `jit(vmap(grad))` | max abs difference vs loop |
|---|---|---|---|---|
| 8 | 2.6 ms | 0.7 ms (3.2–3.8× faster) | 0.14 ms (14–18×) | 0 |
| 32 | 12.0 ms | 2.6 ms (4.6–7.0×) | 1.64 ms (6–13×) | 0 |
| 128 | 51.5 ms | 9.2 ms (5.6–8.3×) | 5.9 ms (9–12×) | 0 |

The times shown are from the last run.

- `vmap(grad)` gives **exactly** the loop's per-example gradients (difference 0), and
  `torch.func.vmap(grad)` agrees to 1e-12.
- The vectorised version gets relatively faster as the batch grows: in each run its
  speed-up rises from batch 8 to batch 128.
- `jit` removes Python tracing overhead. That overhead matters most at small batch, where
  it is most of the cost.

**jit on one gradient call** (∂loss/∂W1, one example):
- Eager `grad` re-runs linearize and transpose on every call: 0.43–0.53 ms.
- `jit(grad)` runs the compiled 14-equation program, 5 fewer than the 19 traced:
  0.058–0.079 ms, so 5.4–9.1× faster than eager.
- A hand-written numpy gradient took 0.039–0.130 ms on the same runs. So the compiled
  program is the same order of cost as code written by hand, and it agrees with it to
  3.4e-15.

**Second order**, for f: R²⁴ → R:
- Forward-over-reverse (`jvp` of `grad`) and reverse-over-reverse Hessian-vector products
  agree to 1.1e-16 with each other and with H·v.
- Forward-over-reverse was faster in all three runs (0.47–0.71 ms vs 0.78–1.88 ms), even
  though its jitted program is longer (39 equations vs 24). Eager reverse-over-reverse
  pays for two nested linearize/transpose passes in Python, and that cost does not show
  in the final program size.
- For the full 24 × 24 Hessian, neither `jacfwd(jacrev)` nor `jacrev(jacrev)` was
  consistently faster at this size (1.2–3.6 ms vs 1.7–3.4 ms).

## Tests

`pytest -q` runs 48 tests:
- **Gradients and Hessians of 14 functions match `torch.func`.** The functions cover
  broadcasting, softmax and logsumexp, `max` with ties, reshape and transpose, `where`,
  and einsum with an index only one operand has. Five also match central finite
  differences.
- **Jacobians:** `jacfwd` and `jacrev` of a vector function match `torch.func.jacrev`.
- **Per-example gradients:** they match `torch.func.vmap(grad)` and a Python loop.
- **Hessian-vector products:** forward-over-reverse = reverse-over-reverse = H·v.
- **Linearize and transpose:**
  - the transpose passes the dot-product test;
  - linearize evaluates the primal and stages only linear primitives.
- **vmap:** 18 cases (6 functions × 3 `in_axes` patterns) equal a Python loop, and nested
  `vmap` and `vmap(hessian)` work.
- **jit:**
  - it equals eager, traces once per shape, and composes as `vmap(grad(jit(f)))` and
    `jit(vmap(grad(jit(f))))`;
  - CSE and DCE shrink the program without changing its output.
- **Errors:**
  - nested `grad` and nested `jvp` avoid perturbation confusion: d/dx [x · d/dy(x · y)] = 2x;
  - Python control flow on a tracer raises, and so does using a tracer that escaped its
    transform.

```
pip install -e ".[dev]" && pytest -q
python -m minijax.bench --out results
```
