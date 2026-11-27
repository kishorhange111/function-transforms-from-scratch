import numpy as onp
import pytest
import torch

import minijax as mj
from minijax import np as jnp
from minijax.core import PRIMITIVES

rng = onp.random.default_rng(0)
T = lambda *a: [torch.tensor(x) for x in a]

# scalar functions of one array, each paired with its torch twin
FUNCS = {
    "poly": (lambda x: jnp.sum(x**3 - 2 * x), lambda x: (x**3 - 2 * x).sum()),
    "trig": (
        lambda x: jnp.sum(jnp.sin(x) * jnp.cos(2 * x)),
        lambda x: (x.sin() * (2 * x).cos()).sum(),
    ),
    "exp_log": (
        lambda x: jnp.sum(jnp.log(1 + jnp.exp(x))),
        lambda x: torch.log(1 + x.exp()).sum(),
    ),
    "tanh_div": (
        lambda x: jnp.sum(jnp.tanh(x) / (1 + x * x)),
        lambda x: (x.tanh() / (1 + x * x)).sum(),
    ),
    "relu": (lambda x: jnp.sum(jnp.relu(x) * x), lambda x: (torch.relu(x) * x).sum()),
    "softmax": (
        lambda x: jnp.sum(jnp.softmax(x) ** 2),
        lambda x: (torch.softmax(x, -1) ** 2).sum(),
    ),
    "logsumexp": (
        lambda x: jnp.sum(jnp.logsumexp(x)),
        lambda x: torch.logsumexp(x, -1).sum(),
    ),
    "max": (
        lambda x: jnp.sum(jnp.max(x, 0) * 3),
        lambda x: (x.max(0).values * 3).sum(),
    ),
    "mean_bcast": (
        lambda x: jnp.sum((x - jnp.mean(x, 1, keepdims=True)) ** 2),
        lambda x: ((x - x.mean(1, keepdim=True)) ** 2).sum(),
    ),  # fmt: skip
    "reshape_T": (
        lambda x: jnp.sum(jnp.reshape(x, (-1,)) * jnp.reshape(x.T, (-1,))),
        lambda x: (x.reshape(-1) * x.T.reshape(-1)).sum(),
    ),  # fmt: skip
    "einsum_outer": (
        lambda x: jnp.sum(jnp.einsum("ij,kl->ijkl", x, x) ** 2),
        lambda x: (torch.einsum("ij,kl->ijkl", x, x) ** 2).sum(),
    ),  # fmt: skip
    "einsum_only_a": (
        lambda x: jnp.einsum("ij,k->", x, jnp.sum(x, 0)),
        lambda x: torch.einsum("ij,k->", x, x.sum(0)),
    ),  # fmt: skip
    "where": (
        lambda x: jnp.sum(jnp.where(x > 0.1, x * x, jnp.sin(x))),
        lambda x: torch.where(x > 0.1, x * x, x.sin()).sum(),
    ),  # fmt: skip
    "gram": (lambda x: jnp.sum(jnp.tanh(x @ x.T)), lambda x: torch.tanh(x @ x.T).sum()),
}


@pytest.mark.parametrize("name", FUNCS)
def test_grad_and_hessian_match_torch(name):
    f, tf = FUNCS[name]
    x = rng.normal(size=(3, 4))
    (xt,) = T(x)
    onp.testing.assert_allclose(
        mj.grad(f)(x), torch.func.grad(tf)(xt).numpy(), rtol=1e-10, atol=1e-12
    )
    onp.testing.assert_allclose(
        mj.hessian(f)(x), torch.func.hessian(tf)(xt).numpy(), rtol=1e-9, atol=1e-11
    )


@pytest.mark.parametrize("name", ["poly", "trig", "softmax", "gram", "max"])
def test_grad_matches_finite_differences(name):
    f, _ = FUNCS[name]
    x = rng.normal(size=(3, 4))
    g = mj.grad(f)(x)
    eps, fd = 1e-6, onp.zeros_like(x)
    for i in onp.ndindex(x.shape):
        e = onp.zeros_like(x)
        e[i] = eps
        fd[i] = (f(x + e) - f(x - e)) / (2 * eps)
    onp.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-7)


def test_max_ties_share_the_gradient():
    g = mj.grad(lambda x: jnp.max(x))(onp.array([1.0, 3.0, 3.0, 2.0]))
    onp.testing.assert_array_equal(g, [0, 0.5, 0.5, 0])


def test_transpose_is_the_adjoint():
    """<ct, J t> = <Jᵀ ct, t> for the linear jaxpr linearize returns (the dot-product test)."""
    W = rng.normal(size=(4, 3))
    f = lambda x: jnp.tanh(jnp.einsum("bi,ij->bj", x, W)) * jnp.sum(x, 1, keepdims=True)
    x = rng.normal(size=(5, 4))
    _, lin = mj.linearize(f, x)
    for _ in range(5):
        t, ct = rng.normal(size=x.shape), rng.normal(size=(5, 3))
        (jt,) = mj.eval_jaxpr(lin, [t])
        (jtct,) = mj.transpose(lin, ct)
        assert onp.sum(ct * jt) == pytest.approx(onp.sum(jtct * t), rel=1e-12)


def test_linearize_evaluates_the_primal_and_stages_only_linear_work():
    x = rng.normal(size=(6,))
    f = lambda x: jnp.sum(jnp.exp(jnp.sin(x)) * jnp.tanh(x))
    y, lin = mj.linearize(f, x)
    assert y == pytest.approx(f(x))
    prims = {e.prim.name for e in lin.eqns}
    assert prims.isdisjoint({"exp", "sin", "cos", "tanh", "log"}), prims
    assert all(PRIMITIVES[p].transpose is not None for p in prims)


def test_jacobians_of_a_vector_function_match_torch():
    W = rng.normal(size=(4, 3))
    f = lambda x: jnp.tanh(x @ W) * jnp.sum(x)
    tf = lambda x: torch.tanh(x @ torch.tensor(W)) * x.sum()
    x = rng.normal(size=(4,))
    ref = torch.func.jacrev(tf)(torch.tensor(x)).numpy()
    onp.testing.assert_allclose(mj.jacfwd(f)(x), ref, rtol=1e-12, atol=1e-13)
    onp.testing.assert_allclose(mj.jacrev(f)(x), ref, rtol=1e-12, atol=1e-13)


def mlp_loss(params, x, y):
    W1, W2 = params
    h = jnp.tanh(jnp.einsum("i,ij->j", x, W1))
    return jnp.sum((jnp.einsum("j,jk->k", h, W2) - y) ** 2)


def torch_mlp_loss(W1, W2, x, y):
    return ((torch.tanh(x @ W1) @ W2 - y) ** 2).sum()


def test_per_example_gradients_match_torch():
    W1, W2 = rng.normal(size=(5, 8)), rng.normal(size=(8, 3))
    X, Y = rng.normal(size=(16, 5)), rng.normal(size=(16, 3))
    loss = lambda W1, W2, x, y: mlp_loss((W1, W2), x, y)
    g1, g2 = (mj.vmap(mj.grad(loss, i), in_axes=(None, None, 0, 0))(W1, W2, X, Y) for i in (0, 1))
    t1, t2 = torch.func.vmap(torch.func.grad(torch_mlp_loss, (0, 1)), in_dims=(None, None, 0, 0))(
        *T(W1, W2, X, Y)
    )
    onp.testing.assert_allclose(g1, t1.numpy(), rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(g2, t2.numpy(), rtol=1e-12, atol=1e-12)
    loop = onp.stack([mj.grad(loss, 0)(W1, W2, X[i], Y[i]) for i in range(16)])
    onp.testing.assert_allclose(g1, loop, rtol=1e-13, atol=1e-13)


def test_hvp_forward_over_reverse_equals_reverse_over_reverse():
    f, _ = FUNCS["gram"]
    x, v = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    fwd_rev = mj.jvp(mj.grad(f), (x,), (v,))[1]
    rev_rev = mj.grad(lambda x: jnp.sum(mj.grad(f)(x) * v))(x)
    H = mj.hessian(f)(x).reshape(12, 12)
    onp.testing.assert_allclose(fwd_rev, rev_rev, rtol=1e-11, atol=1e-12)
    onp.testing.assert_allclose(fwd_rev.reshape(-1), H @ v.reshape(-1), rtol=1e-11, atol=1e-12)


VMAP = {
    "elementwise": lambda x, y: jnp.sin(x) * y + 1.0,
    "reduce": lambda x, y: jnp.sum(x * y, 0) + jnp.max(x),
    "matmul": lambda x, y: jnp.einsum("ij,jk->ik", x, y.T),
    "reshape_T": lambda x, y: jnp.reshape(x.T, (-1,)) * jnp.reshape(y, (-1,)),
    "broadcast": lambda x, y: x - jnp.mean(y, 1, keepdims=True),
    "softmax": lambda x, y: jnp.softmax(x * y),
}


@pytest.mark.parametrize("name", VMAP)
@pytest.mark.parametrize("axes", [(0, 0), (0, None), (1, 0)])
def test_vmap_equals_a_loop(name, axes):
    f = VMAP[name]
    n = 7
    X = rng.normal(size=(n, 3, 4) if axes[0] == 0 else (3, n, 4))
    Y = rng.normal(size=(n, 3, 4)) if axes[1] == 0 else rng.normal(size=(3, 4))
    xi = lambda i: X[i] if axes[0] == 0 else X[:, i]
    yi = lambda i: Y[i] if axes[1] == 0 else Y
    ref = onp.stack([f(xi(i), yi(i)) for i in range(n)])
    onp.testing.assert_allclose(mj.vmap(f, in_axes=axes)(X, Y), ref, rtol=1e-13, atol=1e-13)


def test_nested_vmap_and_vmap_of_jacobian():
    f = lambda x: jnp.sum(jnp.tanh(x) ** 2)
    X = rng.normal(size=(3, 5, 4))
    out = mj.vmap(mj.vmap(mj.grad(f)))(X)
    ref = onp.stack([[mj.grad(f)(X[i, j]) for j in range(5)] for i in range(3)])
    onp.testing.assert_allclose(out, ref, rtol=1e-13)
    H = mj.vmap(mj.hessian(f))(X[0])
    onp.testing.assert_allclose(H[2], mj.hessian(f)(X[0, 2]), rtol=1e-13)


def test_jit_equals_eager_caches_and_composes():
    W1, W2 = rng.normal(size=(5, 8)), rng.normal(size=(8, 3))
    x, y = rng.normal(size=(5,)), rng.normal(size=(3,))
    loss = lambda W1, W2, x, y: mlp_loss((W1, W2), x, y)
    g = mj.grad(loss, (0, 1))
    gj = mj.jit(lambda *a: g(*a)[0])
    onp.testing.assert_allclose(gj(W1, W2, x, y), g(W1, W2, x, y)[0], rtol=1e-14)
    gj(W1, W2, x + 1, y)
    assert len(gj.cache) == 1  # same shapes: traced once
    # jit under transforms: differentiate and batch through a jitted function
    fj = mj.jit(lambda x: jnp.sum(jnp.sin(x) * x))
    xs = rng.normal(size=(4, 6))
    onp.testing.assert_allclose(
        mj.vmap(mj.grad(fj))(xs), onp.sin(xs) + xs * onp.cos(xs), rtol=1e-13
    )
    onp.testing.assert_allclose(mj.jit(mj.vmap(mj.grad(fj)))(xs), onp.sin(xs) + xs * onp.cos(xs),
                                rtol=1e-13)  # fmt: skip


def test_optimize_removes_dead_and_duplicate_work_without_changing_results():
    f = lambda x: jnp.sum(jnp.sin(x) * jnp.sin(x)) + 0.0 * jnp.sum(jnp.exp(x))
    x = rng.normal(size=(5,))
    raw = mj.make_jaxpr(mj.grad(f), x)
    opt = mj.optimize(raw)
    assert len(opt.eqns) < len(raw.eqns)
    run, src = mj.compile_jaxpr(opt)
    onp.testing.assert_allclose(run(x), mj.eval_jaxpr(raw, [x])[0], rtol=1e-15)
    names = [e.prim.name for e in opt.eqns]
    assert names.count("sin") == 1  # CSE merged sin(x) · sin(x)


def test_perturbation_confusion_is_avoided():
    # d/dx [x · d/dy (x · y)] = d/dx [x · x] = 2x  (Siskind & Pearlmutter)
    out = mj.grad(lambda x: x * mj.grad(lambda y: x * y)(onp.float64(3.0)))(onp.float64(2.0))
    assert out == pytest.approx(4.0)
    # and in forward mode: nested jvp keeps its tangents apart
    inner = lambda x: mj.jvp(lambda y: x * y, (onp.float64(3.0),), (onp.float64(1.0),))[1]
    assert mj.jvp(lambda x: x * inner(x), (onp.float64(2.0),), (onp.float64(1.0),))[1] == 4.0


def test_errors_for_control_flow_on_tracers_and_escaped_tracers():
    with pytest.raises(TypeError):
        mj.grad(lambda x: x if x > 0 else -x)(onp.float64(1.0))
    leak = []
    mj.grad(lambda x: leak.append(x) or x * x)(onp.float64(1.0))
    with pytest.raises(RuntimeError):
        jnp.add(leak[0], 1.0)
