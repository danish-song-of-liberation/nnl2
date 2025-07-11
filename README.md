```text
 .--..--..--..--..--..--..--..--..--. 
/ .. \.. \.. \.. \.. \.. \.. \.. \.. \
\ \/\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ \/ /
 \/ /`--'`--'`--'`--'`--'`--'`--'\/ / 
 / /\                            / /\ 
/ /\ \               _ ____     / /\ \
\ \/ /   _ __  _ __ | |___ \    \ \/ /
 \/ /   | '_ \| '_ \| | __) |    \/ / 
 / /\   | | | | | | | |/ __/     / /\ 
/ /\ \  |_| |_|_| |_|_|_____|   / /\ \
\ \/ /                          \ \/ /
 \/ /                            \/ / 
 / /\.--..--..--..--..--..--..--./ /\ 
/ /\ \.. \.. \.. \.. \.. \.. \.. \/\ \
\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
 `--'`--'`--'`--'`--'`--'`--'`--'`--' 
```

# nnl2
Common Lisp (CL) neural network framework 

*About the Author:* This framework is being developed by a 14-year-old (already 15) as a personal solo project. All code, all bugs, all mine!

I write the framework mainly for myself because writing on torch or tensorflow develops into eternal procrastination and unwillingness to write code on it.

Framework has a first version (nnl), you can see it in my repositories.

# Why didn't I decide to finish the first version of NNL?

to be more specific, due to problems with BPTT, recurrent graphs, and poor library selection

# Framework Architecture

The framework is divided into three main components:

1. Tensor System (70-80% complete)
2. Autodiff System (with 5 different modes)
3. Neural Networks Implementations

# Tensor System

The tensor system is currently the most developed part of the framework but however it is already possible to use

**Example:**

```lisp
;; Create and print a 5x6x7 float64 tensor
(let ((tensor (nnl2.ffi::make-tensor-from-shape '(5 6 7) :dtype :float64)))
  (nnl2.ffi::print-tensor tensor))

;; Output:
;; #<NNL2-TENSOR FLOAT64 [5x6x7]>

;; Create and print a 5x6 matrix
(let ((tensor (nnl2.ffi::make-tensor-from-shape '(5 6) :dtype :float64)))
  (nnl2.ffi::print-tensor tensor))

;; Output:
;; #<NNL2-TENSOR FLOAT64 [5x6]:
;;       0.0d0      0.0d0      0.0d0      0.0d0      0.0d0      0.0d0
;;       0.0d0      0.0d0      0.0d0      0.0d0      0.0d0      0.0d0
;;       0.0d0      0.0d0      0.0d0      0.0d0      0.0d0      0.0d0
;;       0.0d0      0.0d0      0.0d0      0.0d0      0.0d0      0.0d0
;;       0.0d0      0.0d0      0.0d0      0.0d0      0.0d0      0.0d0>

;; Create and print a vector
(let ((tensor (nnl2.ffi::make-tensor-from-shape '(4) :dtype :float64)))
  (nnl2.ffi::print-tensor tensor))

;; Output:
;; #<NNL2-TENSOR FLOAT64 [4]:
;;       0.0d0
;;       0.0d0
;;       0.0d0
;;       0.0d0>

(let ((tensor (nnl2.ffi::make-tensor-from-shape '(-4 -6 -48234 -52384232 -1) :dtype :float64))) ;; WHY DOES THIS WORKING
  (nnl2.ffi::print-tensor tensor))

;; idk why this works but it works

;; Output:
;; #<NNL2-TENSOR FLOAT64 [-4x-6x-48234x-52384232x-1]> (wtf)
```

# NNL2 Documentation (Unstable)

This documentation is highly volatile and subject to daily changes. There is currently no stable, unified documentation for NNL2 as the framework is under active development.

W.I.P.

