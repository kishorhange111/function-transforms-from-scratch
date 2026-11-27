"""Staging to a jaxpr, reverse mode by linearize + transpose, and jit.

**Staging.** A ``StageTrace`` records every primitive applied to its tracers as an
equation of a small typed program, a *jaxpr*. Any value from outside the trace (a numpy
constant, or a tracer of a lower level) becomes a constant of the jaxpr.

**Reverse mode** is not a separate autodiff system: it is forward mode, partially
evaluated, then transposed, as in JAX.
- ``linearize`` runs ``jvp`` with staged tangent inputs, one level below the jvp. Primal
  operations never touch a staged value, so they run immediately. Tangent operations
  always do, so they are recorded. What is recorded is a jaxpr *linear* in the
  tangents; the primal values it needs (residuals) are its constants.
- ``transpose`` runs that linear jaxpr backwards. Each linear primitive has a transpose
  rule (add ↔ fan-out, broadcast ↔ sum, einsum ↔ einsum with the cotangent), and
  cotangents are accumulated with ``add``.
- ``vjp`` is linearize then transpose. ``grad`` is vjp with cotangent 1.

The backward pass binds primitives too, so it can itself be differentiated (``grad`` of
``grad``), batched (``vmap(grad)``, ``jacrev``) or staged (``jit(grad)``).

**jit** stages a function once per input shape, removes duplicate (CSE) and unused (DCE)
equations, and compiles the jaxpr to straight-line Python. When called on concrete
arrays, it runs that code with no tracing at all. Under an outer transform, it replays
the jaxpr through ``bind`` instead.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np

from . import ops
from .core import PRIMITIVES as P
from .core import Trace, Tracer, aval_of, new_main, parse_einsum, shape_of
from .jvp import JVPTrace, JVPTracer, zeros_like_aval


# -- jaxprs ----------------------------------------------------------------------------
class Var:
    _ids = itertools.count()

    def __init__(self, aval):
        self.aval, self.id = aval, next(Var._ids)

    def __repr__(self):
        return f"v{self.id}"


class Eqn:
    __slots__ = ("prim", "ins", "params", "out")

    def __init__(self, prim, ins, params, out):
        self.prim, self.ins, self.params, self.out = prim, ins, params, out


class Jaxpr:
    def __init__(self, in_vars, consts, eqns, outs):
        self.in_vars, self.consts, self.eqns, self.outs = in_vars, consts, eqns, outs

    def __repr__(self):
        p = lambda d: "" if not d else "[" + ", ".join(f"{k}={v}" for k, v in d.items()) + "]"
        lines = [f"{{ lambda {', '.join(f'{v}:{v.aval}' for v in self.in_vars)} ."]
        for v, _ in self.consts:
            lines.append(f"    {v}:{v.aval} = const")
        for e in self.eqns:
            args = " ".join(map(repr, e.ins))
            lines.append(f"    {e.out}:{e.out.aval} = {e.prim}{p(e.params)} {args}")
        lines.append(f"  in ({', '.join(map(repr, self.outs))}) }}")
        return "\n".join(lines)


class StageTracer(Tracer):
    def __init__(self, trace, aval):
        self._trace, self._aval = trace, aval

    @property
    def aval(self):
        return self._aval


class Builder:
    def __init__(self):
        self.vars = {}  # id(tracer) → Var
        self.keep = []  # keep tracers and constants alive so ids stay unique
        self.consts, self.eqns = [], []

    def var(self, t):
        return self.vars[id(t)]

    def new(self, t):
        v = self.vars[id(t)] = Var(t.aval)
        self.keep.append(t)
        return v


class StageTrace(Trace):
    def pure(self, x):
        b = self.main.data
        t = StageTracer(self, aval_of(x))
        b.consts.append((b.new(t), x))
        b.keep.append(x)
        return t

    def process(self, prim, tracers, params):
        b = self.main.data
        out = StageTracer(self, prim.shape(*[t.aval for t in tracers], **params))
        b.eqns.append(Eqn(prim, [b.var(t) for t in tracers], params, b.new(out)))
        return out


def _stage(f, avals, wrap=None):
    """Trace f on staged inputs of the given avals. Returns (jaxpr, outputs of ``wrap``)."""
    b = Builder()
    with new_main(StageTrace, b) as main:
        trace = StageTrace(main)
        ins = [StageTracer(trace, a) for a in avals]
        in_vars = [b.new(t) for t in ins]
        res = f(*ins)
        outs, extra = (res, None) if wrap is None else wrap(res)
        outs = outs if isinstance(outs, (tuple, list)) else [outs]
        out_vars = []
        for o in outs:
            if not (isinstance(o, StageTracer) and o._trace.main is main):
                o = trace.pure(o)
            out_vars.append(b.var(o))
    return Jaxpr(in_vars, b.consts, b.eqns, out_vars), extra


def make_jaxpr(f, *args):
    jaxpr, _ = _stage(f, [aval_of(a) for a in args])
    return jaxpr


def eval_jaxpr(jaxpr, args):
    env = {v.id: val for v, val in jaxpr.consts}
    env.update({v.id: a for v, a in zip(jaxpr.in_vars, args, strict=True)})
    for e in jaxpr.eqns:
        env[e.out.id] = e.prim.bind(*[env[v.id] for v in e.ins], **e.params)
    return [env[v.id] for v in jaxpr.outs]


# -- linearize and transpose -------------------------------------------------------------
def linearize(f, *primals):
    """(f(primals), a jaxpr computing the JVP at primals, linear in the tangents)."""
    out_primal = []

    def tangent_fn(*tangents):
        with new_main(JVPTrace) as main:
            trace = JVPTrace(main)
            ins = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents, strict=True)]
            out = f(*ins)
            if not (isinstance(out, JVPTracer) and out._trace.main is main):
                out = trace.pure(out)
            out_primal.append(out.primal)
            return out.tangent

    avals = [aval_of(x) for x in primals]
    jaxpr, _ = _stage(tangent_fn, avals)
    return out_primal[0], jaxpr


class UndefPrimal:
    """A linear input of an equation in the transpose: only its aval is known."""

    def __init__(self, aval):
        self.aval = aval


def transpose(jaxpr, ct_out):
    """Cotangents of the linear jaxpr's inputs, given the output cotangent."""
    const = {v.id: x for v, x in jaxpr.consts}
    linear = {v.id for v in jaxpr.in_vars} | {e.out.id for e in jaxpr.eqns}
    ct = {}

    def acc(v, c):
        ct[v.id] = c if v.id not in ct else ops.add(ct[v.id], c)

    if jaxpr.outs[0].id in linear:
        acc(jaxpr.outs[0], ct_out)
    for e in reversed(jaxpr.eqns):
        c = ct.pop(e.out.id, None)
        if c is None:
            continue
        args = [UndefPrimal(v.aval) if v.id in linear else const[v.id] for v in e.ins]
        assert e.prim.transpose is not None, f"{e.prim} is not linear"
        for v, ci in zip(e.ins, e.prim.transpose(c, *args, **e.params), strict=True):
            if ci is not None and v.id in linear:
                acc(v, ci)
    return [ct.get(v.id, zeros_like_aval(v.aval)) for v in jaxpr.in_vars]


def vjp(f, *primals):
    out, lin = linearize(f, *primals)
    return out, lambda ct: transpose(lin, ct)


def grad(f, argnums=0):
    def g(*args):
        out, f_vjp = vjp(f, *args)
        assert shape_of(out) == (), "grad needs a scalar output"
        cts = f_vjp(np.ones((), aval_of(out).dtype))
        return cts[argnums] if isinstance(argnums, int) else tuple(cts[i] for i in argnums)

    return g


def value_and_grad(f, argnums=0):
    def g(*args):
        out, f_vjp = vjp(f, *args)
        cts = f_vjp(np.ones((), aval_of(out).dtype))
        return out, cts[argnums] if isinstance(argnums, int) else tuple(cts[i] for i in argnums)

    return g


# -- transpose rules -------------------------------------------------------------------
def tr(name):
    def deco(fn):
        P[name].transpose = fn
        return fn

    return deco


undef = lambda x: isinstance(x, UndefPrimal)


@tr("add")
def _(ct, x, y):
    return [ct if undef(x) else None, ct if undef(y) else None]


@tr("neg")
def _(ct, x):
    return [P["neg"].bind(ct)]


@tr("mul")
def _(ct, x, y):
    assert not (undef(x) and undef(y)), "mul of two linear values is not linear"
    return [P["mul"].bind(ct, y), None] if undef(x) else [None, P["mul"].bind(x, ct)]


@tr("div")
def _(ct, x, y):
    assert undef(x) and not undef(y), "div is linear only in its numerator"
    return [P["div"].bind(ct, y), None]


@tr("select")
def _(ct, p, a, b):
    z = zeros_like_aval(aval_of(ct))
    return [
        None,
        P["select"].bind(p, ct, z) if undef(a) else None,
        P["select"].bind(p, z, ct) if undef(b) else None,
    ]


@tr("reduce_sum")
def _(ct, x, axes):
    return [P["broadcast"].bind(ct, shape=x.aval.shape, axes=axes)]


@tr("broadcast")
def _(ct, x, shape, axes):
    return [P["reduce_sum"].bind(ct, axes=axes)]


@tr("transpose")
def _(ct, x, perm):
    return [P["transpose"].bind(ct, perm=tuple(np.argsort(perm)))]


@tr("reshape")
def _(ct, x, shape):
    return [P["reshape"].bind(ct, shape=x.aval.shape)]


def _einsum_ct(ct, other, s_ct, s_other, s_want, want_shape):
    """∂/∂(operand with indices s_want): contract ct with the other operand, then
    broadcast over indices that only the wanted operand has (summed in the forward)."""
    kept = "".join(c for c in s_want if c in s_ct or c in s_other)
    r = P["einsum"].bind(ct, other, spec=f"{s_ct},{s_other}->{kept}")
    if kept == s_want:
        return r
    axes = tuple(i for i, c in enumerate(s_want) if c not in kept)
    return P["broadcast"].bind(r, shape=tuple(want_shape), axes=axes)


@tr("einsum")
def _(ct, a, b, spec):
    sa, sb, out = parse_einsum(spec)
    assert not (undef(a) and undef(b)), "einsum of two linear values is not linear"
    if undef(a):
        return [_einsum_ct(ct, b, out, sb, sa, a.aval.shape), None]
    return [None, _einsum_ct(ct, a, out, sa, sb, b.aval.shape)]


# -- jacobians -------------------------------------------------------------------------
def jacfwd(f):
    from .jvp import jvp
    from .vmap import vmap

    def jac(x):
        xs = shape_of(x)
        n = int(np.prod(xs))
        basis = np.eye(n, dtype=aval_of(x).dtype).reshape((n, *xs))
        cols = vmap(lambda t: jvp(f, (x,), (t,))[1])(basis)  # (n, *out)
        out_shape = shape_of(cols)[1:]
        return ops.reshape(ops.transpose(ops.reshape(cols, (n, -1))), out_shape + xs)

    return jac


def jacrev(f):
    from .vmap import vmap

    def jac(x):
        out, f_vjp = vjp(f, x)
        os_ = shape_of(out)
        m = int(np.prod(os_))
        basis = np.eye(m, dtype=aval_of(out).dtype).reshape((m, *os_))
        rows = vmap(lambda ct: f_vjp(ct)[0])(basis)  # (m, *in)
        return ops.reshape(rows, os_ + shape_of(x))

    return jac


def hessian(f):
    return jacfwd(jacrev(f))


# -- jit -------------------------------------------------------------------------------
def _freeze(v):
    return tuple(v) if isinstance(v, list) else v


def optimize(jaxpr):
    """Common-subexpression elimination, then dead-code elimination."""
    alias, seen, eqns = {}, {}, []
    for e in jaxpr.eqns:
        ins = [alias.get(v.id, v) for v in e.ins]
        key = (
            e.prim.name,
            tuple(v.id for v in ins),
            tuple(sorted((k, _freeze(v)) for k, v in e.params.items())),
        )
        if key in seen:
            alias[e.out.id] = seen[key]
            continue
        seen[key] = e.out
        eqns.append(Eqn(e.prim, ins, e.params, e.out))
    outs = [alias.get(v.id, v) for v in jaxpr.outs]
    live = {v.id for v in outs}
    kept = []
    for e in reversed(eqns):
        if e.out.id in live:
            kept.append(e)
            live |= {v.id for v in e.ins}
    consts = [(v, x) for v, x in jaxpr.consts if v.id in live]
    return Jaxpr(jaxpr.in_vars, consts, kept[::-1], outs)


def compile_jaxpr(jaxpr):
    """Straight-line Python for a jaxpr with concrete constants: one numpy call per line."""
    ns, name = {}, {}
    args = []
    for i, v in enumerate(jaxpr.in_vars):
        name[v.id] = f"a{i}"
        args.append(f"a{i}")
    for i, (v, x) in enumerate(jaxpr.consts):
        name[v.id] = ns[f"c{i}"] = f"c{i}"
        ns[f"c{i}"] = np.asarray(x)
    lines = [f"def run({', '.join(args)}):"]
    for k, e in enumerate(jaxpr.eqns):
        ns[f"f{k}"] = functools.partial(e.prim.impl, **e.params)
        name[e.out.id] = f"v{k}"
        lines.append(f"    v{k} = f{k}({', '.join(name[v.id] for v in e.ins)})")
    lines.append(f"    return {name[jaxpr.outs[0].id]}")
    src = "\n".join(lines)
    exec(src, ns)
    return ns["run"], src


def jit(f):
    cache = {}

    def wrapped(*args):
        avals = tuple(aval_of(a) for a in args)
        entry = cache.get(avals)
        if entry is None:
            jaxpr = optimize(make_jaxpr(f, *avals))
            concrete = not any(isinstance(x, Tracer) for _, x in jaxpr.consts)
            entry = (jaxpr, compile_jaxpr(jaxpr)[0] if concrete else None)
            if concrete:  # a jaxpr closing over an outer tracer is only valid right now
                cache[avals] = entry
        jaxpr, run = entry
        if run is not None and not any(isinstance(a, Tracer) for a in args):
            return run(*args)
        return eval_jaxpr(jaxpr, args)[0]

    wrapped.cache = cache
    return wrapped
