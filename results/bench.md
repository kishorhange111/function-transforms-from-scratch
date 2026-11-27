## Per-example gradients (MLP 64 → 256 → 10, float64)

| batch | loop of grad (ms) | vmap(grad) (ms) | jit(vmap(grad)) (ms) | vmap speed-up | jit speed-up | max abs diff vs loop |
|---|---|---|---|---|---|---|
| 8 | 2.6 | 0.7 | 0.14 | 3.6× | 18× | 0.0e+00 |
| 32 | 12.0 | 2.6 | 1.64 | 4.7× | 7× | 0.0e+00 |
| 128 | 51.5 | 9.2 | 5.94 | 5.6× | 9× | 0.0e+00 |

## jit on one gradient (∂loss/∂W1, one example)

| eager grad | jit(grad) | hand-written numpy gradient | equations traced | after CSE + DCE |
|---|---|---|---|---|
| 0.43 ms | 0.078 ms | 0.130 ms | 19 | 14 |

jit(grad) differs from the hand-written gradient by at most 3.4e-15.

## Second order (f: R^24 → R, tanh-quadratic + sin)

| quantity | forward-over-reverse | reverse-over-reverse |
|---|---|---|
| Hessian-vector product | 0.55 ms, 39 equations | 1.88 ms, 24 equations |
| full Hessian (24×24) | 2.8 ms | 3.4 ms |

The two HVPs agree to 1.1e-16; the two Hessians and H·v agree to 1.1e-16.
