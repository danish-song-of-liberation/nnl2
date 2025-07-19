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

Comparing the speed of creating 64x64 tensors from zeros

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

## Ones:

Comparing the speed of creating 64x64 tensors from ones (yet only torch and numpy)

P.S. The avx implementation is selected automatically if it is supported

```text
nnl2 (C, float32, AVX): ~ 4.6M (!) Tensors per second (0.219 sec to create 1m tensors)
nnl2 (C, float64, AVX): ~ 3.6M Tensors per second (0.279 sec to create 1m tensors)
nnl2 (C, float32, NAIVE): ~ 1.225M Tensors per second (0.816 sec to create 1m tensors)
nnl2 (C, float64, NAIVE): ~ 1.222M Tensors per second (0.818 sec to create 1m tensors)
nnl2 (Lisp, float32, AVX): ~ 3.66M (!) Tensors per second (0.273 sec to create 1m tensors)
nnl2 (Lisp, float64, AVX): ~ 2M Tensors per second (0.481 sec to create 1m tensors)
nnl2 (Lisp, float32, NAIVE): ~ 1.135M Tensors per second (0.881 sec to create 1m tensors)
nnl2 (Lisp, float64, NAIVE): ~ 1.031M Tensors per second (0.969 sec to create 1m tensors)
numpy (float32): ~ 90K Tensors per second (10.982 sec to create 1m tensors)
numpy (float64): ~ 40K Tensors per second (24.288 sec to create 1m tensors)
torch (float32): ~ 93K Tensors per second (10.753 sec to create 1m tensors)
torch (float64): ~ 38K Tensors per second (26.086 sec to create 1m tensors)

Top 1: nnl2
Top 2: Torch
Top 3: NumPy
```

## Full:

Comparing the speed of creating 64x64 tensors from ones (fills with 2.0)

P.S. The avx implementation is selected automatically if it is supported

```text
nnl2 (C, float32, AVX): ~ 4.9M (!) Tensors per second (0.204 sec to create 1m tensors)
nnl2 (C, float64, AVX): ~ 3.3M Tensors per second (0.300 sec to create 1m tensors)
nnl2 (C, float32, NAIVE): ~ 1.25M Tensors per second (0.800 sec to create 1m tensors)
nnl2 (C, float64, NAIVE): ~ 1.2M Tensors per second (0.835 sec to create 1m tensors)
nnl2 (Lisp, float32, AVX): ~ 3.86M (!) Tensors per second (0.259 sec to create 1m tensors)
nnl2 (Lisp, float64, AVX): ~ 2.06M Tensors per second (0.485 sec to create 1m tensors)
nnl2 (Lisp, float32, NAIVE): ~ 1.17M Tensors per second (0.854 sec to create 1m tensors)
nnl2 (Lisp, float64, NAIVE): ~ 1.04M Tensors per second (0.958 sec to create 1m tensors)
numpy (float32): ~ 1.27M Tensors per second (0.782 sec to create 1m tensors)
numpy (float64): ~ 1.36M Tensors per second (0.721 sec to create 1m tensors)
torch (float32): ~ 504K Tensors per second (1.980 sec to create 1m tensors)
torch (float64): ~ 494K Tensors per second (2.021 sec to create 1m tensors) 

Top 1: nnl2
Top 2: NumPy
Top 3: Torch
```


## Lisp code from benchmarks (Empty):

```lisp
(declaim (optimize (speed 3) (debug 0) (safety 0) (space 0))) ;; optimizing

(time (let ((shape (nnl2.hli.ts:make-shape-pntr #(64 64)))) ;; to avoid recreating the dimensions every time
        (dotimes (i 1000000) ;; 1_000_000
          (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:empty-with-pntr shape 2 :dtype :float64))))))) ;; an optimized form of empty that takes a pointer to dimensions instead of a vector. is 10-20% faster than the original zeros. 2 is rank

;; tlet is a tensor-based version of let that automatically clears memory at the end
```

## Lisp code from benchmarks (Zeros):

```lisp
(declaim (optimize (speed 3) (debug 0) (safety 0) (space 0))) ;; optimizing

(time (let ((shape (nnl2.hli.ts:make-shape-pntr #(64 64)))) ;; to avoid recreating the dimensions every time
        (dotimes (i 1000000) ;; 1_000_000
          (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:zeros-with-pntr shape 2 :dtype :float64))))))) ;; an optimized form of zeros that takes a pointer to dimensions instead of a vector. is 10-20% faster than the original zeros. 2 is rank

;; tlet is a tensor-based version of let that automatically clears memory at the end
```

## Lisp code from benchmarks (Ones):

```lisp
(declaim (optimize (speed 3) (debug 0) (safety 0) (space 0))) ;; optimizing

(time (let ((shape (nnl2.hli.ts:make-shape-pntr #(64 64)))) ;; to avoid recreating the dimensions every time
        (dotimes (i 1000000) ;; 1_000_000
          (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:ones-with-pntr shape 2 :dtype :float32))))))) ;; an optimized form of ones that takes a pointer to dimensions instead of a vector. is 10-20% faster than the original ones

;; AVX selected automatically

;; tlet is a tensor-based version of let that automatically clears memory at the end
```

## Lisp code from benchmarks (Full):

```lisp 
(declaim (optimize (speed 3) (debug 0) (safety 0) (space 0))) ;; optimizing

(time (let ((shape (nnl2.hli.ts:make-shape-pntr #(64 64)))
            (filler (nnl2.hli:make-foreign-pointer 2.0d0 :double)))

        (dotimes (i 1000000) ;; 1_000_000
          (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:full-with-pntr shape 2 :filler filler :dtype :float64))))))) ;; an optimized form of ones that takes a pointer to dimensions instead of a vector. is 10-20% faster than the original ones
 
;; AVX selected automatically

;; tlet is a tensor-based version of let that automatically clears memory at the end
```
