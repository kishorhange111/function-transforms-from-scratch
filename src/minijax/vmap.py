"""Vectorisation: a tracer carries (value, batch dim) and each primitive a batching rule.

``vmap(f)`` runs f once on whole batches. Each rule rewrites one primitive to act on the
batched value: elementwise operations move the batch axis to the front, reductions and
reshapes shift their axes, and einsum gives the batch axis a fresh index letter.
"""

from __future__ import annotations

import numpy as np

from .core import PRIMITIVES as P
from .core import ShapedArray, Trace, Tracer, aval_of, fresh_letter, new_main, parse_einsum


class BatchTracer(Tracer):
    def __init__(self, trace, val, bdim):
        self._trace, self.val, self.bdim = trace, val, bdim

    @property
    def aval(self):
        a = aval_of(self.val)
        if self.bdim is None:
            return a
        return ShapedArray(a.shape[: self.bdim] + a.shape[self.bdim + 1 :], a.dtype)

    def full_lower(self):
        return self.val if self.bdim is None else self


class BatchTrace(Trace):
    def pure(self, x):
        return BatchTracer(self, x, None)

    def process(self, prim, tracers, params):
        vals = [t.val for t in tracers]
        dims = [t.bdim for t in tracers]
        if all(d is None for d in dims):
            return BatchTracer(self, prim.bind(*vals, **params), None)
        out, d = prim.batch(self.main.data, vals, dims, **params)
        return BatchTracer(self, out, d)


def to_front(x, d, size):
    """Put the batch axis first, materialising it if x is not batched."""
    if d is None:
        a = aval_of(x)
        return P["broadcast"].bind(x, shape=(size, *a.shape), axes=(0,))
    if d == 0:
        return x
    nd = aval_of(x).ndim
    perm = (d, *[i for i in range(nd) if i != d])
    return P["transpose"].bind(x, perm=perm)


def vmap(f, in_axes=0):
    def batched(*args):
        axes = in_axes if isinstance(in_axes, (tuple, list)) else (in_axes,) * len(args)
        sizes = {np.shape(a)[d] if not isinstance(a, Tracer) else a.shape[d]
                 for a, d in zip(args, axes, strict=True) if d is not None}  # fmt: skip
        assert len(sizes) == 1, f"inconsistent batch sizes {sizes}"
        size = sizes.pop()
        with new_main(BatchTrace, size) as main:
            trace = BatchTrace(main)
            ins = [BatchTracer(trace, a, d) for a, d in zip(args, axes, strict=True)]
            out = f(*ins)

            def lower(o):
                if not (isinstance(o, BatchTracer) and o._trace.main is main):
                    o = trace.pure(o)
                return to_front(o.val, o.bdim, size)

            return tuple(map(lower, out)) if isinstance(out, (tuple, list)) else lower(out)

    return batched


# -- rules -----------------------------------------------------------------------------
def _elementwise(size, vals, dims, prim):
    return prim.bind(*[to_front(v, d, size) for v, d in zip(vals, dims, strict=True)]), 0


for _name in ("neg", "exp", "log", "sin", "cos", "tanh", "add", "mul", "div", "max",
              "greater", "eq", "select"):  # fmt: skip
    P[_name].batch = (lambda p: lambda size, vals, dims: _elementwise(size, vals, dims, p))(
        P[_name]
    )


def _reduce(prim):
    def f(size, vals, dims, axes):
        x = to_front(vals[0], dims[0], size)
        return prim.bind(x, axes=tuple(a + 1 for a in axes)), 0

    return f


P["reduce_sum"].batch = _reduce(P["reduce_sum"])
P["reduce_max"].batch = _reduce(P["reduce_max"])


def _broadcast(size, vals, dims, shape, axes):
    x = to_front(vals[0], dims[0], size)
    return P["broadcast"].bind(x, shape=(size, *shape), axes=tuple(a + 1 for a in axes)), 0


def _transpose(size, vals, dims, perm):
    x = to_front(vals[0], dims[0], size)
    return P["transpose"].bind(x, perm=(0, *[p + 1 for p in perm])), 0


def _reshape(size, vals, dims, shape):
    x = to_front(vals[0], dims[0], size)
    return P["reshape"].bind(x, shape=(size, *shape)), 0


def _einsum(size, vals, dims, spec):
    sa, sb, out = parse_einsum(spec)
    z = fresh_letter(spec)
    (a, b), (da, db) = vals, dims
    if da is not None:
        a, sa = to_front(a, da, size), z + sa
    if db is not None:
        b, sb = to_front(b, db, size), z + sb
    return P["einsum"].bind(a, b, spec=f"{sa},{sb}->{z}{out}"), 0


P["broadcast"].batch = _broadcast
P["transpose"].batch = _transpose
P["reshape"].batch = _reshape
P["einsum"].batch = _einsum
