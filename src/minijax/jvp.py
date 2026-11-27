"""Forward mode: a tracer carries (primal, tangent) and each primitive has a JVP rule.

The rules are written with ops and primitives, never with numpy directly. Tangent
computations are then themselves traceable: staged by ``linearize``, batched by ``vmap``,
or differentiated again by an outer ``jvp``.
"""

from __future__ import annotations

import numpy as np

from . import ops
from .core import PRIMITIVES as P
from .core import Trace, Tracer, aval_of, new_main


def zeros_like_aval(aval):
    return np.zeros(aval.shape, np.float64 if aval.dtype == bool else aval.dtype)


class JVPTracer(Tracer):
    def __init__(self, trace, primal, tangent):
        self._trace, self.primal, self.tangent = trace, primal, tangent

    @property
    def aval(self):
        return aval_of(self.primal)


class JVPTrace(Trace):
    def pure(self, x):
        return JVPTracer(self, x, zeros_like_aval(aval_of(x)))

    def process(self, prim, tracers, params):
        primals = [t.primal for t in tracers]
        tangents = [t.tangent for t in tracers]
        out, dout = prim.jvp(primals, tangents, **params)
        return JVPTracer(self, out, dout)


def jvp(f, primals, tangents):
    """(f(primals), J_f(primals) · tangents) for a function of arrays returning one array."""
    with new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        ins = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents, strict=True)]
        out = f(*ins)
        out = out if isinstance(out, Tracer) and out._trace.main is main else trace.pure(out)
        return out.primal, out.tangent


# -- rules -----------------------------------------------------------------------------
def rule(name):
    def deco(fn):
        P[name].jvp = fn
        return fn

    return deco


@rule("add")
def _(p, t):
    return P["add"].bind(*p), P["add"].bind(*t)


@rule("mul")
def _(p, t):
    (x, y), (dx, dy) = p, t
    return P["mul"].bind(x, y), P["add"].bind(P["mul"].bind(dx, y), P["mul"].bind(x, dy))


@rule("div")
def _(p, t):
    (x, y), (dx, dy) = p, t
    out = P["div"].bind(x, y)
    num = P["add"].bind(dx, P["neg"].bind(P["mul"].bind(out, dy)))
    return out, P["div"].bind(num, y)


@rule("neg")
def _(p, t):
    return P["neg"].bind(p[0]), P["neg"].bind(t[0])


@rule("exp")
def _(p, t):
    y = P["exp"].bind(p[0])
    return y, P["mul"].bind(t[0], y)


@rule("log")
def _(p, t):
    return P["log"].bind(p[0]), P["div"].bind(t[0], p[0])


@rule("sin")
def _(p, t):
    return P["sin"].bind(p[0]), P["mul"].bind(t[0], P["cos"].bind(p[0]))


@rule("cos")
def _(p, t):
    return P["cos"].bind(p[0]), P["neg"].bind(P["mul"].bind(t[0], P["sin"].bind(p[0])))


@rule("tanh")
def _(p, t):
    y = P["tanh"].bind(p[0])
    return y, P["mul"].bind(t[0], ops.sub(1.0, P["mul"].bind(y, y)))


@rule("max")
def _(p, t):
    (x, y), (dx, dy) = p, t
    take_x = P["greater"].bind(x, y)  # ties take y's tangent
    return P["max"].bind(x, y), P["select"].bind(take_x, dx, dy)


for _name in ("greater", "eq"):

    def _cmp(p, t, _name=_name):
        out = P[_name].bind(*p)
        return out, zeros_like_aval(aval_of(out))

    P[_name].jvp = _cmp


@rule("select")
def _(p, t):
    return P["select"].bind(*p), P["select"].bind(p[0], t[1], t[2])


@rule("reduce_sum")
def _(p, t, axes):
    return P["reduce_sum"].bind(p[0], axes=axes), P["reduce_sum"].bind(t[0], axes=axes)


@rule("reduce_max")
def _(p, t, axes):
    (x,), (dx,) = p, t
    out = P["reduce_max"].bind(x, axes=axes)
    back = P["broadcast"].bind(out, shape=aval_of(x).shape, axes=axes)
    mask = ops.where(P["eq"].bind(x, back), 1.0, 0.0)  # ties share the gradient equally
    count = P["reduce_sum"].bind(mask, axes=axes)
    return out, P["div"].bind(P["reduce_sum"].bind(P["mul"].bind(dx, mask), axes=axes), count)


def _structural(name):
    def f(p, t, **params):
        return P[name].bind(p[0], **params), P[name].bind(t[0], **params)

    P[name].jvp = f


for _name in ("broadcast", "transpose", "reshape"):
    _structural(_name)


@rule("einsum")
def _(p, t, spec):
    (a, b), (da, db) = p, t
    e = P["einsum"]
    return e.bind(a, b, spec=spec), P["add"].bind(
        e.bind(da, b, spec=spec), e.bind(a, db, spec=spec)
    )
