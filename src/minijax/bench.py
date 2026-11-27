"""What the transforms buy, measured.

    python -m minijax.bench --out results

1. Per-example gradients of an MLP: ``vmap(grad)`` against a Python loop over ``grad``.
2. ``jit``: tracing once and running compiled straight-line numpy, against eager ``grad``,
   which re-traces linearize and transpose on every call. Also how many equations CSE and
   DCE remove.
3. Second order: a Hessian-vector product forward-over-reverse vs reverse-over-reverse,
   and the full Hessian as ``jacfwd(jacrev)`` vs ``jacrev(jacrev)``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as onp

import minijax as mj
from minijax import np as jnp


def timeit(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def mlp(W1, W2, x, y):
    h = jnp.tanh(jnp.einsum("i,ij->j", x, W1))
    return jnp.sum((jnp.einsum("j,jk->k", h, W2) - y) ** 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=7)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rng = onp.random.default_rng(0)
    rows, md = [], []

    # 1. per-example gradients
    din, dh, dout = 64, 256, 10
    W1, W2 = rng.normal(size=(din, dh)) * 0.1, rng.normal(size=(dh, dout)) * 0.1
    g = mj.grad(mlp, (0, 1))
    pe = mj.vmap(g, in_axes=(None, None, 0, 0))
    pe_jit = mj.jit(lambda W1, W2, X, Y: pe(W1, W2, X, Y)[0])
    md += ["## Per-example gradients (MLP 64 → 256 → 10, float64)", "",
           "| batch | loop of grad (ms) | vmap(grad) (ms) | jit(vmap(grad)) (ms) | vmap speed-up "
           "| jit speed-up | max abs diff vs loop |", "|---" * 7 + "|"]  # fmt: skip
    for B in (8, 32, 128):
        X, Y = rng.normal(size=(B, din)), rng.normal(size=(B, dout))
        loop = lambda X=X, Y=Y: [g(W1, W2, X[i], Y[i]) for i in range(len(X))]
        t_loop = timeit(loop, a.reps)
        t_vmap = timeit(lambda X=X, Y=Y: pe(W1, W2, X, Y), a.reps)
        t_jit = timeit(lambda X=X, Y=Y: pe_jit(W1, W2, X, Y), a.reps)
        ref = onp.stack([r[0] for r in loop()])
        diff = max(float(onp.abs(pe(W1, W2, X, Y)[0] - ref).max()),
                   float(onp.abs(pe_jit(W1, W2, X, Y) - ref).max()))  # fmt: skip
        r = {"bench": "per_example", "batch": B, "loop_s": t_loop, "vmap_s": t_vmap,
             "jit_s": t_jit, "max_diff": diff}  # fmt: skip
        rows.append(r)
        md.append(f"| {B} | {t_loop * 1e3:.1f} | {t_vmap * 1e3:.1f} | {t_jit * 1e3:.2f} | "
                  f"{t_loop / t_vmap:.1f}× | {t_loop / t_jit:.0f}× | {diff:.1e} |")  # fmt: skip

    # 2. jit on one gradient call, and what CSE + DCE remove
    x, y = rng.normal(size=din), rng.normal(size=dout)
    g0 = lambda W1, W2, x, y: g(W1, W2, x, y)[0]
    gj = mj.jit(g0)
    raw = mj.make_jaxpr(g0, W1, W2, x, y)
    opt = mj.optimize(raw)
    t_eager, t_j = (
        timeit(lambda: g0(W1, W2, x, y), a.reps),
        timeit(lambda: gj(W1, W2, x, y), a.reps),
    )
    t_numpy = timeit(lambda: _numpy_grad(W1, W2, x, y), a.reps)
    _check = float(onp.abs(gj(W1, W2, x, y) - _numpy_grad(W1, W2, x, y)).max())
    rows.append({"bench": "jit", "eager_s": t_eager, "jit_s": t_j, "numpy_s": t_numpy,
                 "eqns_raw": len(raw.eqns), "eqns_opt": len(opt.eqns), "max_diff": _check})  # fmt: skip
    md += ["", "## jit on one gradient (∂loss/∂W1, one example)", "",
           "| eager grad | jit(grad) | hand-written numpy gradient | equations traced | after CSE + DCE |",
           "|---|---|---|---|---|",
           f"| {t_eager * 1e3:.2f} ms | {t_j * 1e3:.3f} ms | {t_numpy * 1e3:.3f} ms | "
           f"{len(raw.eqns)} | {len(opt.eqns)} |", "",
           f"jit(grad) differs from the hand-written gradient by at most {_check:.1e}."]  # fmt: skip

    # 3. second order
    n = 24
    A = rng.normal(size=(n, n)) / n
    f = lambda x: jnp.sum(jnp.tanh(jnp.einsum("ij,j->i", A, x)) ** 2) + jnp.sum(jnp.sin(x))
    xv, v = rng.normal(size=n), rng.normal(size=n)
    fwd_rev = lambda: mj.jvp(mj.grad(f), (xv,), (v,))[1]
    rev_rev = lambda: mj.grad(lambda x: jnp.sum(mj.grad(f)(x) * v))(xv)
    H_fr = lambda: mj.jacfwd(mj.jacrev(f))(xv)
    H_rr = lambda: mj.jacrev(mj.jacrev(f))(xv)
    H = H_fr()
    d_hvp = float(onp.abs(fwd_rev() - rev_rev()).max())
    d_H = float(max(onp.abs(H_rr() - H).max(), onp.abs(H @ v - fwd_rev()).max()))
    times = {k: timeit(fn, a.reps) for k, fn in
             (("hvp_fwd_rev", fwd_rev), ("hvp_rev_rev", rev_rev), ("hess_fwd_rev", H_fr),
              ("hess_rev_rev", H_rr))}  # fmt: skip
    eq = {k: len(mj.optimize(mj.make_jaxpr(fn, xv)).eqns) for k, fn in
          (("hvp_fwd_rev", lambda x: mj.jvp(mj.grad(f), (x,), (v,))[1]),
           ("hvp_rev_rev", lambda x: mj.grad(lambda z: jnp.sum(mj.grad(f)(z) * v))(x)))}  # fmt: skip
    rows.append({"bench": "second_order", "n": n, **times, "eqns": eq, "max_diff_hvp": d_hvp,
                 "max_diff_hessian": d_H})  # fmt: skip
    md += ["", f"## Second order (f: R^{n} → R, tanh-quadratic + sin)", "",
           "| quantity | forward-over-reverse | reverse-over-reverse |", "|---|---|---|",
           f"| Hessian-vector product | {times['hvp_fwd_rev'] * 1e3:.2f} ms, "
           f"{eq['hvp_fwd_rev']} equations | {times['hvp_rev_rev'] * 1e3:.2f} ms, "
           f"{eq['hvp_rev_rev']} equations |",
           f"| full Hessian ({n}×{n}) | {times['hess_fwd_rev'] * 1e3:.1f} ms | "
           f"{times['hess_rev_rev'] * 1e3:.1f} ms |", "",
           f"The two HVPs agree to {d_hvp:.1e}; the two Hessians and H·v agree to {d_H:.1e}."]  # fmt: skip

    with open(os.path.join(a.out, "bench.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


def _numpy_grad(W1, W2, x, y):
    """∂loss/∂W1 by hand, as the reference for what jit should approach."""
    h = onp.tanh(x @ W1)
    r = h @ W2 - y
    dh = (2 * r) @ W2.T * (1 - h * h)
    return onp.outer(x, dh)


if __name__ == "__main__":
    main()
