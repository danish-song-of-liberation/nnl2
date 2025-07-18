This file contains benchmarks comparing the speed of nnl2's zeros, empty (and ones, full, etc. in the future) functions with other frameworks (torch, numpy).

Attention! all code is very optimized

## Empty:

Compares the ability to create a number of empty 64x64 tensors (filled with garbage) per second

```text
nnl2 (C, float32): ~7M Tensors per second (0.142 sec to create 1m tensors)
nnl2 (C, float64): ~ 6.2M Tensors per second (0.159 sec to create 1m tensors)
nnl2 (Lisp, float32): ~ 5.65M Tensors per second (0.177 sec to create 1m tensors)
nnl2 (Lisp, float64): ~ 4.75M Tensors per second (0.211 sec to create 1m tensors)
numpy (float32): ~ 2.8M Tensors per second (0.358 sec to create 1m tensors)
numpy (float64): ~ 3.8M Tensors per second (0.263 sec to create 1m tensors) (lol why its faster that float32)
torch (float32): ~ 590K Tensors per second (1.712 sec to create 1m tensors)
torch (float64): ~ 610K Tensors per second (1.641 sec to create 1m tensors) (again faster that float32, why?)

Top 1: nnl2
Top 2: NumPy
Top 3: Torch
```

## Zeros:
```text
nnl2 (C, float32): ~ 4.2M (!) Tensors per second (0.236 sec to create 1m tensors)
nnl2 (C, float64): ~ 3.7M Tensors per second (0.267 sec to create 1m tensors)
nnl2 (Lisp, float32): ~ 4.1M (!) Tensors per second (0.240 sec to create 1m tensors)
nnl2 (Lisp, float64): ~ 2.1M Tensors per second (0.467 sec to create 1m tensors)
numpy (float32): ~ 110K Tensors per second (8.962 sec to create 1m tensors)
numpy (float64): ~ 50K Tensors per second (20.080 sec to create 1m tensors)
torch (float32): ~ 110K Tensors per second (8.982 sec to create 1m tensors)
torch (float64): ~ 55K Tensors per second (17.691 sec to create 1m tensors)

Top 1: nnl2 
Top 2: Torch
Top 3: NumPy
```
