"""The numpy-like user API, built only from primitives.

Binary primitives need equal shapes, so numpy broadcasting happens here, as explicit
``reshape`` and ``broadcast`` primitives that the transforms differentiate and batch
like any other.
"""

from __future__ import annotations

import numpy as np

from .core import PRIMITIVES as P
from .core import (
    broadcast_p,
    einsum_p,
    reduce_max_p,
    reduce_sum_p,
    reshape_p,
    select_p,
    shape_of,
    transpose_p,
)


def broadcast_to(x, shape):
    shape = tuple(shape)
    xs = shape_of(x)
    if xs == shape:
        return x
    lead = len(shape) - len(xs)
    assert lead >= 0, f"cannot broadcast {xs} to {shape}"
    keep = [i for i, d in enumerate(xs) if not (d == 1 and shape[lead + i] != 1)]
    for i in keep:
        assert xs[i] == shape[lead + i], f"cannot broadcast {xs} to {shape}"
    if len(keep) != len(xs):
        x = reshape_p.bind(x, shape=tuple(xs[i] for i in keep))
    kept = {lead + i for i in keep}
    return broadcast_p.bind(
        x, shape=shape, axes=tuple(i for i in range(len(shape)) if i not in kept)
    )


def _binary(name):
    p = P[name]

    def f(x, y):
        shape = np.broadcast_shapes(shape_of(x), shape_of(y))
        return p.bind(broadcast_to(x, shape), broadcast_to(y, shape))

    f.__name__ = name
    return f


add, mul, div, maximum = (_binary(n) for n in ("add", "mul", "div", "max"))
greater, equal = _binary("greater"), _binary("eq")


def neg(x):
    return P["neg"].bind(x)


def sub(x, y):
    return add(x, neg(y))


def exp(x):
    return P["exp"].bind(x)


def log(x):
    return P["log"].bind(x)


def sin(x):
    return P["sin"].bind(x)


def cos(x):
    return P["cos"].bind(x)


def tanh(x):
    return P["tanh"].bind(x)


def power(x, k):
    assert isinstance(k, int) and k >= 1, "only positive integer powers"
    out = x
    for _ in range(k - 1):
        out = mul(out, x)
    return out


def where(pred, a, b):
    shape = np.broadcast_shapes(shape_of(pred), shape_of(a), shape_of(b))
    return select_p.bind(broadcast_to(pred, shape), broadcast_to(a, shape), broadcast_to(b, shape))


def _axes(x, axis):
    nd = len(shape_of(x))
    if axis is None:
        return tuple(range(nd))
    axis = (axis,) if isinstance(axis, int) else axis
    return tuple(sorted(a % nd for a in axis))


def _keep(out, x, axes, keepdims):
    if not keepdims:
        return out
    return reshape(out, tuple(1 if i in axes else d for i, d in enumerate(shape_of(x))))


def sum(x, axis=None, keepdims=False):  # noqa: A001
    axes = _axes(x, axis)
    return _keep(reduce_sum_p.bind(x, axes=axes), x, axes, keepdims)


def max(x, axis=None, keepdims=False):  # noqa: A001
    axes = _axes(x, axis)
    return _keep(reduce_max_p.bind(x, axes=axes), x, axes, keepdims)


def mean(x, axis=None, keepdims=False):
    axes = _axes(x, axis)
    n = int(np.prod([shape_of(x)[a] for a in axes]))
    return div(sum(x, axis, keepdims), float(n))


def reshape(x, shape):
    shape = tuple(shape)
    if -1 in shape:
        n = int(np.prod(shape_of(x)))
        known = int(np.prod([d for d in shape if d != -1]))
        shape = tuple(n // known if d == -1 else d for d in shape)
    if shape == shape_of(x):
        return x
    return reshape_p.bind(x, shape=shape)


def transpose(x, perm=None):
    perm = tuple(reversed(range(len(shape_of(x))))) if perm is None else tuple(perm)
    if perm == tuple(range(len(perm))):
        return x
    return transpose_p.bind(x, perm=perm)


def einsum(spec, a, b):
    return einsum_p.bind(a, b, spec=spec.replace(" ", ""))


def matmul(a, b):
    na, nb = len(shape_of(a)), len(shape_of(b))
    if na == 1 and nb == 1:
        return einsum("i,i->", a, b)
    if na == 2 and nb == 2:
        return einsum("ij,jk->ik", a, b)
    if na == 2 and nb == 1:
        return einsum("ij,j->i", a, b)
    if na == 1 and nb == 2:
        return einsum("j,jk->k", a, b)
    raise NotImplementedError("matmul of rank > 2: use einsum")


def relu(x):
    return maximum(x, 0.0)


def logsumexp(x, axis=-1):
    m = max(x, axis, keepdims=True)
    return add(log(sum(exp(sub(x, m)), axis, keepdims=True)), m)


def softmax(x, axis=-1):
    return exp(sub(x, logsumexp(x, axis)))
