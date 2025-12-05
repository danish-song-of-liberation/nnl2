# General nnl2 documentation


This documentation provides general information about all the components of the library.

In the same directory, there are the **click-me-if-you-lisp-advanced/click-me-if-you-lisp-beginner** folders. If you want more personalized and convenient documentation, you can navigate to them.

The library is divided into **3** main parts: **the tensor system (TS)**, **automatic differentiation (AD)**, **and neural network implementations (NN)**

The documentation is not fully complete at the moment because the library is still under active development, and the documentation will be gradually updated as the library is completed.

## Contents:
- [1. Getting Started] (#1-getting-started) 
- [2. Installing] (#2-installing)
- [3. Tensor system] (#3-tensor-system)
- [3.1. Making the first tensor] (#3-1-making-the-first-tensor)
- [3.2. Perform the first calculations] (#3-2-perform-the-first-calculations)
- [3.3. Initializing the tensor] (#3-3-initializing-the-tensor)
- [3.4. Like-constructors] (#3-4-like-constructors)
- [3.5. Activation functions] (#3-5-activation-functions)
- [3.6. Other] (#-3-6-ts-other)
- [4.0. Runtime-dispatching] (#-4-runtime-dispatching)
- [5.0. Automatic Differentation] (#-5-automatic-differentation)

## 1. Getting Started

nnl2 Is a **neural network library** in common lisp with a C core

The framework provides **high-level**, **user-friendly**, and **high-performance** tools for working with tensors, autodiff, and neural networks.

## 2. Installing

```bash
git clone https://github.com/danish-song-of-liberation/nnl2/tree/main
```

Then try it.

```lisp
(ql:quickload :nnl2)
```

If quicklisp does not find the system, try loading it yourself (see the code)

```lisp
(load #P"path-to-nnl2.lisp") ;; it should be in the nnl2 project folder
(ql:quickload :nnl2)
```

If an error occurs, try running Lisp from the project folder
If the error persists, please write to issues or nnl.dev@proton.me

# 3. Tensor System 

So far, there are only **3** types of tensors:

|  Element Type | Class Suffix | 
|---------------|--------------|
| DOUBLE-FLOAT  | FLOAT64      |
| SINGLE-FLOAT  | FLOAT32			 |
| INTEGER       | INT32        |

# Other Library Equivalents

*This table was adapted largely from the [Magicl Equivalents Table].*

[Magicl Equivalents Table]: https://github.com/quil-lang/magicl/blob/master/doc/high-level.md

*All nnl2 functions in the table will be from the :nnl2.hli.ts package (sometimes the nnl2.lli.ts package may appear)*

*I would also like to inform you that the table is very extensive.*

# Language used/optimizations

| nnl2 | MAGICL | MATLAB | NumPy | PyTorch | Description |
|:----:|:------:|:------:|:-----:|:-------:|:-----------:|
| Common Lisp + Pure C Core | Common Lisp | MATLAB | Python + C Core | Python + C++ Core | Language / Core |
| AVX, pthread.h, BLAS, -O3, prefetching, caching | BLAS, LAPACK, Expokit | JIT, MKL, LAPACK, multithreading | BLAS, LAPACK, OpenMP, SIMD | ATen/C++ backend, CUDA, OpenMP,  cuDNN, MKL, multithreading, SIMD | Low-level optimizations | 
| Tensor dispatch, runtime dynamic dispatching, dynamic backend system | | | | Kernel dispatch, dynamic dispatch | Tensor dispatch / kernel selection | 
| High-performance tensor ops + autodiff | High-level numerical ops | High-level numerical ops | High-level numerical ops | High-performance tensor ops + autodiff | High-level operations |
| Manual memory management via tlet/RAII-style | Garbage collected | Garbage collected | Garbage collected | Garbage collected + smart memory management (tensors) | Memory management |

# Main

|  nnl2               | MAGICL            | MATLAB          | NumPy                         | PyTorch | Description |
|:-------------------:|:-----------------:|:---------------:|:-----------------------------:|:-------:|:-----------:|
| ```(rank a)```  | ```(order a)```   | ```ndims(a)```  | ```ndim(a)``` or ```a.ndim``` | ```a.dim()``` | Get the number of dimensions of the array. |
| ```(size a)```      | ```(size a)```    | ```numel(a)```  | ```size(a)``` or ```a.size```  | ```a.numel()``` | Get the number of elements of the array. |
| ```(size-in-bytes a)``` | 	| It is usually done manually (numel * sizeof(dtype)) | It is usually done manually (numel * sizeof(dtype)) | ```a.element_size() * a.numel()``` | Gets the dimensions of the tensor in bytes |
| ```(shape a)``` | ```(shape a)```   | ```size(a)```   | ```shape(a)``` or ```a.shape``` | ```a.shape``` or ```a.size()``` | Get the shape of the array. |
| ```(dtype a)``` | 	 | ```class(a)``` | ```a.dtype``` | ```a.dtype``` |  Gets the data type of the array. |
| ```(tref a 1 4) or (view (a 1 4) or (nnl2.lli.ts:trefw a  1 4)```   | ```(tref a 1 4)``` | ```a(2,5)```   | ```a[1, 4]```                 | ```a[1, 4]``` | Get the element in the second row, fifth column of the array. |
| ```(view a 0)``` | | | ```a[0, :] or a[0]``` | ```a[0, :] or a[0]``` | Get the first subtensor from a two-dimensional tensor (view) |
| ```(tref a 0)``` or ```(copy (view a 0))``` | | ```a(1, :)```  | ```a[0, :].copy()``` | ```a[0, :].clone()``` | Get the first subtensor from a two-dimensional tensor (copy) |
| ```(setf (tref a '* 0) 3)``` | | ```a(:, 1) = 3``` | ```a[:, 0] = 3``` | ```a[:, 0] = 3``` | 	Set all elements in the first column to 3 |
| ```(nnl2.lli.ts:flat a 0)``` | | ```a(1)``` | ```a.flat[0]``` or ```a.ravel()[0]``` | ```a.flatten()[0]``` | Get an item by linear index |
| ```(nrows a)``` | | ```size(a, 1)``` | ```a.shape[0]``` | 	```a.size(0)``` or ```a.shape[0]``` | Get number of rows in the tensor |
| ```(ncols a)``` | | ```size(a, 2)``` | ```a.shape[1]``` | ```a.size(1)``` or ```a.shape[1]``` | Get number of columns in the tensor |

### Constructors 

| nnl2 | MAGICL | MATLAB | NumPy | PyTorch | Desctiption |
|------|--------|--------|-------|---------|-------------|
| W.I.P. | ```(from-list '(1d0 2d0 3d0 4d0 5d0 6d0) '(2 3))``` | ```[ 1 2 3; 4 5 6 ]``` | ```np.array([[1.,2.,3.], [4.,5.,6.]])``` | ```torch.tensor([[1,2,3],[4,5,6]])``` | Create a 2x3 matrix from given elements. |
| ```(empty #(3 2 1)``` | ```(empty '(3 2 1))``` | ```empty(3,2,1)``` | ```empty((3,2,1))``` | ```torch.empty(3,2,1)``` | Create an uninitialized 3x2x1 array |
| ```(zeros #(2 3 4))```| ```(zeros '(2 3 4)) or (const 0d0 '(2 3 4))``` | ```zeros(2,3,4)``` | ```zeros((2,3,4))``` | ```torch.zeros(2,3,4)``` |  Create a 2x3x4 dimensional array of zeroes of double-float element type. |
| ```(ones #(3 4))``` | ```(ones '(3 4))``` or ```(const 1d0 '(3 4))``` |  ```ones(3,4)``` | ```ones((3,4))``` | ```torch.ones(3,4)``` | Create a 3x4 dimensional array of ones of double-float element type. |
| ```(full #(6 9) :filler 2.0d0)``` | ```(const 2.0d0 '(6 9))``` | ```2 * ones(6,9)``` | ```full((6, 9), 2)``` | ```torch.full((6,9),2.0)``` | Create a 6x9 dimensional array of a double-float element-type filled with 2. |
| ```(rand #(5 5))``` | ```(rand '(5 5))``` | ```rand(5, 5)``` | ```np.random.rand(5, 5)``` | ```torch.rand(5, 5)``` | ```Generate a 5x5 tensor with random values from uniform distribution [0, 1)``` |
| ```(randn #(5 5))``` | | ```randn(5, 5)``` | ```np.random.randn(5, 5)``` | ```torch.randn(5, 5)``` | Generate a 5x5 tensor with random values from normal distribution (mean=0, std=1) |

###  Basic Operations 

| nnl2 | MAGICL | MATLAB | NumPy | PyTorch | Description |
|------|--------|--------|-------|---------|-------------|
| ```(gemm a b)``` | ```(@ a b)``` | ```a * b``` | ```a @ b``` | ```a @ b``` or ```torch.mm(a, b)``` | Matrix multiplication |
| ```(.+ a b)``` | ```(.+ a b)``` | ```a + b``` | ```a + b``` | ```a + b``` or ```torch.add(a, b)``` |	Element-wise add |
| ```(.- a b)``` | ```(.- a b)``` | ```a - b``` | ```a - b``` | ```a - b``` or ```torch.sub(a, b)``` | Element-wise subtract |
| ```(.* a b)``` | ```(.* a b)``` | ```a .* b``` | ```a * b``` | ```a * b``` or ```torch.mul(a, b)``` | Element-wise multiply |
| ```(./ a b)``` | ```(./ a b)``` | ```a ./ b ``` | ```a / b``` | ```a / b``` or ```torch.div(a, b)``` | Element-wise divide |
| ```(.^ a b)``` | ```(.^ a b)``` | ```a .^ b``` | ```np.power(a,b)``` | ```a ** b``` or ```torch.pow(a,b)``` | Element-wise exponentiation |
| ```(.exp a)``` | ```(.exp a)``` | ```exp(a)``` | ```np.exp(a)``` | ```torch.exp(a)``` | Element-wise exponential |
| ```(.exp! a)``` | | | ```np.exp(a, out=a)``` | ```a.exp_()``` | In-place exponential |
| ```(.log a)``` | ```(.log a)``` | ```log(a)``` | ```np.log(a)``` | ```torch.log(a)``` | Element-wise natural logarithm |
| ```(.log! a)``` | | | ```np.log(a, out=a)``` | ```a.log_()``` |	In-place logarithm |
| ```(.sqrt a)``` | | ```sqrt(a)``` | ```np.sqrt(a)``` | ```torch.sqrt(a)``` | Element-wise square root |
| ```(.sqrt! a)``` | | | ```np.sqrt(a, out=a)``` | ```a.sqrt_()``` | In-place element-wise square root |
| ```(scale a x)``` | ```(scale a x)``` | ```a * x``` | ```a * x``` | ```a * b``` or ```torch.mul(a, b)``` | Scale tensor by scalar |
| ```(scale! a x)``` | ```(scale! a x)``` | ```a = a * x``` | ```np.multiply(a, x, out=a)``` | ```a.mul_(x)``` | In-place scaling |
| ```(+= a b)``` | | ```a += b``` | ```a += b``` | ```a.add_(b)``` | In-place addition |
| ```(-= a b)``` | | ```a -= b``` | ```a -= b``` | ```a.sub_(b)``` | In-place subtraction |
| ```(*= a b)``` | | ```a *= b``` | ```a * b``` | ```a.mul_(b)``` | In-place multiplication |
| ```(/! a b)``` | | ```a /= b``` | ```a /= b``` | ```a.div_(b)``` | In-place division (using /! instead of /=) |
| ```(^= a b)``` | | ```a ** b``` | ```a ** b``` | ```a.pow_(b)``` | In-place exponentiation |
| ```(.max a b)``` | ```(.max a b)``` | ```max(a, b)``` | ```np.maximum(a, b)``` | ```torch.maximum(a, b)``` | Element-wise maximum |
| ```(.max! a b)``` | | | ```np.maximum(a, b, out=a)``` | ```a.set_(torch.maximum(a, b))``` | In-place element-wise maximum |
| ```(.min a b)``` | ```(.min a b)``` | ```min(a,b)``` | ```np.minimum(a, b)``` | ```torch.minimum(a, b)``` | Element-wise minimum |
| ```(.min! a b)``` |  |  | ```np.minimum(a, b, out=a)``` | ```a.set_(torch.minimum(a, b))``` | In-place element-wise minimum |
| ```(.abs a)``` | | ```abs(a)``` | ```np.abs(a)``` | ```torch.abs(a)``` | Element-wise absolute value |
| ```(.abs! a)``` | | | ```np.abs(a, out=a)	``` | ```a.abs_()``` | In-place absolute value |
| ```(.neg a)``` | | ```-a``` | ```-a``` | ```-a``` | Element-wise negation |
| ```(.neg! a)``` | | | ```np.negative(a, out=a)``` | ```a.neg_()``` | In-place element-wise negation |
| W.I.P. | ```(.realpart a)``` | | ```np.real(a)``` | ```a.real``` | Element-wise real part |
| W.I.P. | ```(.imagpart a)``` | | ```np.imag(a)``` | ```a.imag``` | Element-wise imaginary part |
| W.I.P. | ```(.complex a b)``` | | | ```torch.complex(a, b)``` | Complex matrix from rectangular parts |

### Block Matrix Constructors

| nnl2 | MAGICL | MATLAB | NumPy | Pytorch | Description |
|------|--------|--------|-------|---------|-------------|
| YAGNI | ```(block-matrix (list A B C D) '(2 2))``` | ```[A B; C D]``` | ```block([[A,B], [C, D]])``` | ```torch.cat((torch.cat((A,B), dim=1), torch.cat((C,D), dim=1)), dim=0)``` | Create a block matrix from matrices A,B,C,D. |
| YAGNI | ```(block-diag (list A B C))``` | ```blkdiag(A,B,C)``` | ```scipy.linalg.block_diag([A,B,C])``` | ```torch.block_diag(A, B, C)``` | Create a block diagonal matrix from matrices A,B,C. |
| ```(hstack A B C)``` | ```(hstack (list A B C))``` | ```[A B C]``` | ```hstack((A,B,C))``` | ```np.hstack((A,B,C))``` | Concatenate matrices A,B,C horizontally (column-wise). |
| ```(vstack A B C)``` | ```(vstack (list A B C))``` | ```[A; B; C]``` | ```vstack((A,B,C))``` | ```np.vstack((A,B,C))``` | Concatenate matrices A,B,C vertically (row-wise). |	
| ```(concat 2 A B C)``` | | ```cat(3, A, B, C)``` | ```np.concatenate((A, B, C), axis=2)``` | ```torch.cat((A, B, C), dim=2)``` | Concatenate tensors A, B, and C along the third dimension |

### Like-Constructors 

| nnl2 | MAGICL | MATLAB | NumPy | PyTorch | Description |
|------|--------|--------|-------|---------|-------------|
| ```(empty-like a)``` | | ```empty(size(a))``` | ```np.empty_like(a)``` | ```torch.empty_like(a)``` | Creates an uninitialized tensor with shape and type of a |
| ```(zeros-like a)``` | | ```zeros(size(a))``` | ```np.zeros_like(a)``` | ```torch.zeros_like(a)``` | Creates a tensor of zeros with shape and type of a |
| ```(ones-like a)``` | | ```ones(size(a))``` | ```np.ones_like(a)``` | ```torch.ones_like(a)``` | Creates a tensor of ones with shape and type of a | 
| ```(full-like a)``` | | ```x * ones(size(a))``` | ```np.full_like(a, x)``` | ```torch.full_like(a, x)``` | Creates a tensor filled with x as a |

### Mapping Operations

*Please note that nnl2, unlike magicl, can accept an unlimited number of arguments and an unlimited number of arguments in a lambda list.*

| nnl2 | MAGICL | MATLAB | NumPy | PyTorch | Description |
|------|--------|--------|-------|---------|-------------|
| ```(.map #'fn a b ...)``` | ```(map #'fn a)``` | ```arrayfun(fn, a, b, ...)``` | ```np.vectorize(fn)(a, b, ...)``` | ```torch.vmap(fn)(a, b, ...)``` or ```fn(a, b, ...)``` | Applies fn elementwise to tensors a, b, ... (returns a new tensor). |
| ```(.map! #'fn a b ...)``` | ```(map! #'fn a)``` | | | ```a.set_(fn(a, b, ...))``` | Applies fn elementwise, writing the result to the first tensor (a). |
| ```(/map #'fn a b)``` | | | ```np.apply_along_axis(fn, axis, a, b)``` | ```torch.stack([fn(x,y) for x,y in zip(a.unbind(), b.unbind())])``` | Apply fn to corresponding subtensors of a and b along the first dimension |
| ```(/map! #'fn a b)``` | | | | | Apply fn to corresponding subtensors of a and b, writing results in-place to a |

### Activation Functions

| nnl2 | MAGICL | MATLAB | NumPy | PyTorch | Description |
|------|--------|--------|-------|---------|-------------|
| ```(.relu a)``` | | ```max(0, a)``` | ```np.maximum(0, a)``` | ```torch.relu(a)``` | Wise-element Rectified Linear Unit (ReLU) |
| ```(.relu! a)``` | | | ```np.maximum(0, a, out=a)``` | ```torch.relu_(a)``` | In-place element-wise ReLU |
| ```(.leaky-relu a :alpha 0.01)``` | | ```max(0.01*a, a)``` | ```np.where(a > 0, a, 0.01*a)``` | ```torch.nn.functional.leaky_relu(a,0.01)``` | Wise-element Leaky ReLU with optional alpha parameter (usually 0.01) |
| ```(.leaky-relu! a :alpha 0.01)``` | | | ```np.maximum(0.01*a, a, out=a)``` | ```torch.nn.functional.leaky_relu_(a, negative_slope=0.01)``` or ```a.leaky_relu_(negative_slope=0.01)``` | Wise-element In-place Leaky ReLU with optional alpha parameter (usually 0.01) | 
| ```(.sigmoid a)``` | | ```1 ./ (1 + exp(-a))``` | ```1 / (1 + np.exp(-a))``` | ```torch.sigmoid(a)``` | Wise-element Sigmoid |
| ```(.sigmoid! a)``` | | | ```np.reciprocal(1 + np.exp(-a), out=a)``` | ```torch.sigmoid_(a)``` or ```a.sigmoid_()``` | Wise-element In-place Sigmoid |
| ```(.tanh a)``` | | ```tanh(a)``` | ```np.tanh(a)``` | ```torch.tanh(a)``` | Wise-element Hyperbolic Tangent |
| ```(.tanh! a)``` | | | ```np.tanh(a, out=a)``` | ```torch.tanh_(a)``` or ```a.tanh_()``` | Wise-element In-place Hyperbolic Tangent |

 ### Miscellaneous
 
 | nnl2 | MAGICL | MATLAB | NumPy | PyTorch | Description |
 |------|--------|--------|-------|---------|-------------|
 | ```(slice a :to #(5 3)``` | | | ```[:5, :3]``` | ```a[:5, :3]``` | Obtain a tensor with shape #(5 5) as a representation with shape #(5 3) |
 | ```(reinterpret a #(-1))``` | | | ```a.reshape(-1)``` | ```a.reshape(-1)``` | Flatten the tensor to 1D (inferred size) (view) | 
 | ```(reshape a #(-1))``` | | ```a(:)``` | ```a.reshape(-1).copy()``` | ```a.reshape(-1).clone()``` | Flatten the tensor to 1D (inferred size) (copy) |
 | ```(nnl2.hli.ts.utils:narrow a :dim 0 :start 0 :len 16)``` | | ```a(1:16)``` | ```a[0:16]``` | ```a.narrow(0, 0, 16)``` | Extract a subtensor along dimension 0, starting at index 0 with length 16 |
 
 
 ## 3.1 Making the first tensor
 
 Let's start taking the first steps.

 ```(nnl2.hli.ts:tlet ((form)) ...)``` - let-form for tensor. Automatically frees memory


 Let's make a tensor with a zero: ```(nnl2.hli.ts:tlet ((foo (nnl2.hli.ts:zeros #(1) :dtype :float64))) foo))```

 #(1) - Shape
 :dtype - Type of tensor (default: float64)
 
 You can also free up memory manually: ```(let ((a (nnl2.hli.ts:zeros #(1)))) ...
  (nnl2.hli.ts:free a))```
  
  
Make a tensor with manual data input: ```(nnl2.hli.ts:make-tensor #2A((1 2 3) (4 5 6) (7 8 9)))``` or ```(nnl2.hli.ts:from-flatten #(1 2 3 4 5 6 7 8 9) #(3 3))```

I tried to comment on all the functions, so you can see most of the explanations at ```(documentation #'fn 'function)``` (or ```(describe #'fn)```)

## 3.2 Perform the first calculations 

Consider an example of simple addition

```lisp
(use-package :nnl2.hli.ts)

(tlet ((a (ones #(5 5)))  ;; 5x5 float64 tensor filled with 1.0
       (b (full #(5 5) :filler 2))) ;; 5x5 float64 tensor filled with 2.0

  ;; Element-wise addition
  (tlet ((c (.+ a b))) ;; 5x5 tensor filled with 3.0
    (print-tensor c))) ;; Output the result
```

You can shorten the code using tlet* (like let* but for tensors)

```lisp
(use-package :nnl2.hli.ts)

(tlet* ((a (ones #(5 5)))
        (b (full #(5 5) :filler 2))
        ;; Element-wise addition
        (c (.+ a b)))

  (print-tensor c)) ;; Output the result
```

Look at an another example

```lisp
(use-package :nnl2.hli.ts)

(tlet ((a (full #(5 5) :filler 3 :dtype :float32)) ;; 5x5 float32 tensor filled with 3
       (b (full #(5 5) :filler 2 :dtype :int32)))  ;; 5x5 int32 tensor filled with 2

  ;; Element-wise 
  ;; addition in-place
  (+= a b) ;; nnl2 supports smart type conversion

  ;; Print the result
  (print-tensor a)) ;; 5x5 tensor filled with 5.0 (float32)

```

Or this

```lisp
(tlet ((a (full #(5 5) :filler 3 :dtype :float32))
       (b (full #(5 5) :filler 2 :dtype :int32)))

  ;; Element-wise multiplication
  ;; nnl2 has its own type hierarchy
  ;; float64 -> float32 -> int32
  (tlet ((c (.* a b)))

    ;; 5x5 float32 tensor filled with 6 (3 * 2)
    (print-tensor c)))
```
Look at this
```lisp
(tlet ((a (full #(5 5) :filler 2)))
  (.sqrt! a) ;; sqrt(2) ~= 1.414
  (print-tensor a))
```
nnl2 Has broadcasting support
```lisp
(tlet ((a (full #(5 5) :filler 2))
       (b (full #(5) :filler 4)))

  ;; 2 / 4
  ;; The 1D tensor 'b' is automatically broadcast along the rows of 'a'
  (tlet ((c (./ a b))) ;; Broadcasting support

    ;; Print the resulting tensor
    (print-tensor c)))
```
Look at this
```lisp
(tlet ((a (full #(5 5) :filler 2 :dtype :float64))
       (b (full #(5) :filler 4 :dtype :float32)))

  ;; 2 / 4
  ;; Broadcasting + Type conversion
  (tlet ((c (./ a b))) ;; float64

    ;; Print the resulting tensor
    (print-tensor c)))
```
Next example
```lisp
(tlet ((a (full #(5 5) :filler 2)))
  (if (zerop (random 2)) ;; generate random 0 or 1
    (scale! a 2) ;; multiply all elements of 'a' by 2
    (*= a 2)) ;; alternative in-place multiplication

  (print-tensor a)) ;; 4.0
```
Tensors can also work with scalars
```lisp
(tlet ((a (full #(5 5) :filler 2)))
  (^= a 4)
  (print-tensor a)) ;; 16.0

```
Or this
```lisp
(tlet* ((a (full #(5 5) :filler 5))
        (b (.* a 5)))

  (print-tensor b)) ;; 25.0

```
Matrix multiplication
```lisp
(tlet ((a (ones #(3 5)))
       (b (ones #(5 3))))

  (tlet ((c (gemm a b)))
    (print-tensor c))) ;; [3x3] result

```

## 3.3 Initializing the tensor

Filling empty tensor with 2
```lisp
(tlet ((a (empty #(5 5))))
  (fill! a 2))
```

Rand tensor
```lisp
(tlet ((a (rand #(5 5)))))
  
;; or

(tlet ((a (empty #(5 5))))
  (rand! a))
```

Normal distribution
```lisp
(tlet ((a (randn #(5 5))))
  (print-tensor a))

(tlet ((a (randn #(5 5) :std 1.0 :mean 0.0)))
  (print-tensor a))
```

Uniform distribution
```lisp
(tlet ((a (uniform #(5 5) :from 0.5 :to 0.7)))
  (print-tensor a))
```

Xavier/Glorot initialization
```lisp
(tlet ((a (xavier #(5 5) :in 2 :out 1))) ;; 2 Input neurons, 1 output neuron
  (print-tensor a))

;; or

(tlet ((a (xavier #(5 5) :in 2 :out 1 :gain (sqrt 2.0) :distribution :uniform)))
  (print-tensor a))
```

Kaiming/He initialization
```lisp
(tlet ((a (kaiming #(5 5) :in 4 :out 2))) ;; 4 Input neurons, 2 output neuron
  (print-tensor a))

;; or

(tlet ((a (kaiming #(5 5) :in 4 :out 2 :gain 0.7 :distribution :uniform)))
  (print-tensor a))
```

All functions can also be used in-place (see xavier!, etc.)

## 3.4 Like-constructors

Each tensor creation function also has a like version

```lisp
(tlet* ((a (ones #(5 5)))
        (b (zeros-like a))
        (c (ones-like a))
        (d (empty-like a))
        (qux (xavier-like a :in 2 :out 1))
        (quux (kaiming-like a :in 2 :out 1))
        (bar (rand-like a))
        (baz (randn-like a))
        (bazook (uniform-like a)) ;; default from 0.0 to 1.0
        (grample (full-like a :filler 3)))

  ...
       
  )
```

## 3.5 Activation functions

Sigmoid

```lisp
(tlet ((a (ones #(5 5))))
  (tlet ((b (.sigmoid a :approx nil))) ;; By default, :approx is t for performance
    (print-tensor b))) ;; 0.731
```

Tanh

```lisp
(tlet ((a (ones #(5 5))))
  (tlet ((b (.tanh a :approx nil)))
    (print-tensor b))) ;; 0.7615
```

ReLU

```lisp
(tlet ((a (uniform #(5 5) :dtype :int32 :from -1 :to 1)))
  (tlet ((b (.relu a))) 
    (print-tensor b)))
```

LReLU

```lisp
(tlet ((a (uniform #(5 5) :dtype :int32 :from -1 :to 1)))
  (tlet ((b (.leaky-relu a))) ;; Converts to float64
    (print-tensor b)))
```

All functions can also be called on the spot by adding an exclamation mark (!) at the end.

## 3.6 Other

Cast type

```lisp
(tlet* ((a (ones #(5 5) :dtype :float64))
        (b (ncast a :float32)))

  (print-tensor b))
```

Copying tensor

```lisp
(tlet* ((a (ones #(5 5)))
        (b (copy a)))

  (print-tensor b))
```

Reshape as copy

```lisp
(tlet ((a (ones #(5 5))))
  (tlet ((b (reshape a #(-1))))
    (print-tensor b))) ;; [25]

;; or 

(tlet ((a (ones #(5 5))))
  (tlet* ((b (reinterpret a #(-1)))
          (c (copy b)))
         
    (print-tensor c))) ;; [25]
```

Reshape as view
```lisp
(tlet ((a (ones #(5 5))))
  (tlet ((b (reinterpret a #(-1))))
    (print-tensor b))) ;; [25]
```

Slice

```lisp
(tlet ((a (ones #(5 5))))
  (tlet ((b (slice a :from #(2 2) :to #(3 3))))
    (print-tensor b))) ;; [1x1]
```

This is just a small part of the functions available in the tensor system.
You can go to src/lisp/highlevel/highlevel-package.lisp and view all :nnl2.hli.ts the available functions and their documentation using ```(describe ...)``` or ```(documentation ...)```.

# 4.0. Runtime dispatching

You can change backends for each function on the fly through runtime dispatch

```lisp
(tlet ((a (ones #(5000 5000) :dtype :float64)))
  (dolist (backend (get-backends/+=))
    (format t "~%~%~%Testing ~a backend. . ." backend)
    (with-backend/+= backend
      (time (+= a a)))))
```

Each function has its own:

```get-backend/foo``` — Gets the current backend (e.g., naive)

```get-backends/foo``` — Gets all available backends

```use-backend/foo``` (or (setf (get-backend/foo) :bar)) — Permanently sets a new backend for all future calls

```(with-backend/foo ... code ...)``` — Temporarily sets the selected backend for the duration of its body

```lisp
(tlet ((a (ones #(200 200))) (b (ones #(200 200))))
  (use-backend/gemm :naive)
  (time (tlet ((c (gemm a b)))))

  (use-backend/gemm :blas)
  (time (tlet ((c (gemm a b))))))
```

# 5.0. Automatic Differentation (AD)

package: :nnl2.hli.ad

So, automatic differentiation is no different from a tensor system except for a couple of features

First, instead of print-tensor, use print-data

```lisp
(use-package :nnl2.hli.ad)

(tlet ((a (ones #(5 5))))
  (print-data a))
```

Secondly, the :requires-grad key appears

```lisp
(tlet ((a (ones #(5 5) :requires-grad t)))
  (print-data a))
```

The :nnl2.hli.ad.r package is used for any out-in-place operations

```lisp
(use-package :nnl2.hli.ad)
(use-package :nnl2.hli.ad.r)

(tlet ((a (ones #(5 5) :requires-grad t))
       (b (ones #(5 5) :requires-grad t)))

  (tlet ((c (.+ a b)))
    (print-data c)))
```

Calculating the gradient:

```lisp
(use-package :nnl2.hli.ad)
(use-package :nnl2.hli.ad.r)

(tlet ((a (ones #(5 5) :requires-grad t))
       (b (ones #(5 5) :requires-grad t)))

  (tlet ((c (.+ a b)))
    (backpropagation c)

    (print-grad c)
    (print-grad a)
    (print-grad b)))
```

Each operation, including in-place operations, has a :track-graph flag (default: t)

```lisp
(tlet ((c (.+ a b :track-graph nil))) ;; <= DOES NOT TRACK THE GRAPH
```

Equivalent to with torch.no_grad():

```lisp
(with-notrack 
  (tlet ((c (.+ a b)))
    ...))
```

Detach in-place

```lisp
(tlet ((a (ones #(5 5) :requires-grad t))
       (b (ones #(5 5) :requires-grad t)))

  (detach! a)
  (detach! b))
```

Detach out-in-place

```lisp
(tlet ((a (ones #(5 5) :requires-grad t))
       (b (ones #(5 5) :requires-grad t)))

  (tlet ((c (detach a)) ;; <== view, not copy
         (b (detach b))) ;; <== view, not copy

    ...))
```

BPTT

```lisp
(let ((a (ones #(5 5) :requires-grad t)))
  (dotimes (i 10)
    (setf a (.* a a))) ;; Recurrent

  (bptt a)

  (print-grad a)) ;; 1024.0
```

As with nnl2.hli.ts, you can find all the exported functions in src/lisp/highlevel/highlevel-package.lisp

An example of complex code

```lisp
(use-package :nnl2.hli.ad)
(use-package :nnl2.hli.ad.r)

(tlet ((a (full #(5 5) :filler 3 :requires-grad t))
       (b (full #(5 5) :filler 2 :requires-grad t)))

  (tlet ((c (.* a b))
         (d (.+ a b)))

    (rotatef (nnl2.lli.ad:backward-fn c) (nnl2.lli.ad:backward-fn d))

    (backpropagation d))

  (print-grad a) ;; 2.0
  (print-grad b)) ;; 3.0
```

I'm too lazy to write more documentation, so it ends here. You can find code examples in nnl2/examples
