# General nnl2 documentation


This documentation provides general information about all the components of the framework.

In the same directory, there are the **click-me-if-you-lisp-advanced/click-me-if-you-lisp-beginner** folders. If you want more personalized and convenient documentation, you can navigate to them.

The framework is divided into **3** main parts: **the tensor system**, **automatic differentiation (AD)**, **and neural network implementations**

The documentation is not fully complete at the moment because the framework is still under active development, and the documentation will be gradually updated as the framework is completed.

## Contents:
- [1. Getting Started] (#1-getting-started) 
- [2. Installing] (#2-installing)
- [3. Tensor system] (#3-tensor-system)

## 1. Getting Started

nnl2 Is a **neural network framework** in common lisp with a C core

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

*All nnl2 functions in the table will be from the :nnl2.hli.ts package*

*I would also like to inform you that the table is very extensive.*


|  nnl2               | MAGICL            | MATLAB          | NumPy                         | Description |
|:-------------------:|:-----------------:|:---------------:|:-----------------------------:|:------------|
| ```(rank a)```  | ```(order a)```   | ```ndims(a)```  | ```ndim(a)``` or ```a.ndim``` | Get the number of dimensions of the array. |
| ```(size a)```      | ```(size a)```    | ```numel(a)```  | ```size(a)``` or ```a.size```  | Get the number of elements of the array. |
| ```(size-in-bytes a)``` | 	| It is usually done manually (numel * sizeof(dtype)) | It is usually done manually (numel * sizeof(dtype)) | Gets the dimensions of the tensor in bytes |
| ```(shape a)``` | ```(shape a)```   | ```size(a)```   | ```shape(a)``` or ```a.shape``` | Get the shape of the array. |
| ```(dtype a)``` | 	 | ```class(a)``` | ```a.dtype``` | Gets the data type of the array. |
| ```(tref a 1 4)```   | ```(tref a 1 4)``` | ```a(2,5)```   | ```a[1, 4]```                 | Get the element in the second row, fifth column of the array. |


### Constructors 

| nnl2 | MAGICL | MATLAB | NumPy | Desctiption |
|------|--------|--------|-------|-------------|
| yet nope | ```(from-list '(1d0 2d0 3d0 4d0 5d0 6d0) '(2 3))``` | ```[ 1 2 3; 4 5 6 ]``` | ```array([[1.,2.,3.], [4.,5.,6.]])``` | Create a 2x3 matrix from given elements. |
| ```(empty #(3 2 1)``` | ```(empty '(3 2 1))``` | ```empty(3,2,1)``` | ```empty((3,2,1))``` | Create an uninitialized 3x2x1 array |
| ```(zeros #(2 3 4))```| ```(zeros '(2 3 4)) or (const 0d0 '(2 3 4))``` | ```zeros(2,3,4)``` | ```zeros((2,3,4))``` |  Create a 2x3x4 dimensional array of zeroes of double-float element type. |
| ```(ones #(3 4))``` | ```(ones '(3 4))``` or ```(const 1d0 '(3 4))``` |  ```ones(3,4)``` | ```ones((3,4))``` | Create a 3x4 dimensional array of ones of double-float element type. |
| ```(full #(6 9) :filler 2.0d0)``` | ```(const 2.0d0 '(6 9))``` | ```2 * ones(6,9)``` | ```full((6, 9), 2)``` | Create a 6x9 dimensional array of a double-float element-type filled with 2. |

###  Basic Operations 

| nnl2 | MAGICL | MATLAB | NumPy | Description |
|------|--------|--------|-------|-------------|
| ```(gemm a b)``` | ```(@ a b)``` | ```a * b``` | ```a @ b``` | Matrix multiplication |
| ```(.+ a b)``` | ```(.+ a b)``` | ```a + b``` | ```a + b``` | 	Element-wise add |
| ```(.- a b)``` | ```(.- a b)``` | ```a - b``` | ```a - b``` | Element-wise subtract |
| ```(.* a b)``` | ```(.* a b)``` | ```a .* b``` | ```a * b``` | Element-wise multiply |
| ```(./ a b)``` | ```(./ a b)``` | ```a ./ b ``` | ```a / b``` | Element-wise divide |
| ```(.^ a b)``` | ```(.^ a b)``` | ```a .^ b``` | ```np.power(a,b)``` | Element-wise exponentiation |
| ```(.exp a)``` | ```(.exp a)``` | ```exp(a)``` | ```np.exp(a)``` | Element-wise exponential |
| ```(.exp! a)``` | | | ```np.exp(a, out=a)``` | In-place exponential |
| ```(.log a)``` | ```(.log a)``` | ```log(a)``` | ```np.log(a)``` | Element-wise natural logarithm |
| ```(.log! a)``` | | | ```np.log(a, out=a)``` | 	In-place logarithm |
| ```(scale a x)``` | ```(scale a x)``` | ```a * x``` | ```a * x``` | 	Scale tensor by scalar |
| ```(scale! a x)``` | ```(scale! a x)``` | ```a = a * x``` | ```np.multiply(a, x, out=a)``` | In-place scaling |
| ```(+= a b)``` | | ```a += b``` | ```a += b``` | In-place addition |
| ```(-= a b)``` | | ```a -= b``` | ```a -= b``` | In-place subtraction |
| ```(*= a b)``` | | ```a *= b``` | ```a * b``` | In-place multiplication |
| ```(/! a b)``` | | ```a /= b``` | ```a /= b``` | In-place division (using /! instead of /=) |
| ```(.max a b)``` | ```(.max a b)``` | ```max(a, b)``` | ```np.maximum(a, b)``` | Element-wise maximum |
| ```(.max! a b)``` | | | ```np.maximum(a, b, out=a)``` | In-place element-wise maximum |
| ```(.min a b)``` | ```(.min a b)``` | ```min(a,b)``` | ```np.minimum(a, b)``` | Element-wise minimum |
| ```(.min! a b)``` |  |  | ```np.minimum(a, b, out=a)``` | In-place element-wise minimum |
| ```(.abs a)``` | | ```abs(a)``` | ```np.abs(a)``` | 	Element-wise absolute value |
| ```(.abs! a)``` | | | ```np.abs(a, out=a)	``` | In-place absolute value |
| | ```(.realpart a)``` | | ```np.real(a)``` | Element-wise real part |
| | ```(.imagpart a)``` | | ```np.imag(a)``` | Element-wise imaginary part |
| | ```(.complex a b)``` | | | Complex matrix from rectangular parts |

### Block Matrix Constructors

| nnl2 | MAGICL | MATLAB | NumPy | Description |
|------|--------|--------|-------|-------------|
| yet nope | ```(block-matrix (list A B C D) '(2 2))``` | ```[A B; C D]``` | ```block([[A,B], [C, D]])``` | Create a block matrix from matrices A,B,C,D. |
| yet nope | ```(block-diag (list A B C))``` | ```blkdiag(A,B,C)``` | ```scipy.linalg.block_diag([A,B,C])``` | 	Create a block diagonal matrix from matrices A,B,C. |
| ```(hstack A B C)``` | ```(hstack (list A B C))``` | ```[A B C]``` | ```hstack((A,B,C))``` | Concatenate matrices A,B,C horizontally (column-wise). |
| ```(vstack A B C)``` | ```(vstack (list A B C))``` | ```[A; B; C]``` | ```vstack((A,B,C))``` | Concatenate matrices A,B,C vertically (row-wise). |	

### Like-Constructors 

| nnl2 | MAGICL | MATLAB | NumPy | Description |
|------|--------|--------|-------|-------------|
| ```(empty-like a)``` | | ```empty(size(a))``` | ```np.empty_like(a)``` | Creates an uninitialized tensor with shape and type of a |
| ```(zeros-like a)``` | | ```zeros(size(a))``` | ```np.zeros_like(a)``` | Creates a tensor of zeros with shape and type of a |
| ```(ones-like a)``` | | ```ones(size(a))``` | ```np.ones_like(a)``` | Creates a tensor of ones with shape and type of a | 
| ```(full-like a)``` | | ```x * ones(size(a))``` | ```np.full_like(a, x)``` | Creates a tensor filled with x as a |

### Mapping Operations

*Please note that nnl2, unlike magicl, can accept an unlimited number of arguments and an unlimited number of arguments in a lambda list.*

| nnl2 | MAGICL | MATLAB | NumPy | Description |
|------|--------|--------|-------|-------------|
| ```(.map #'fn a b ...)``` | ```(map #'fn a)``` | ```arrayfun(fn, a, b, ...)``` | ```np.vectorize(fn)(a, b, ...)``` | Applies fn elementwise to tensors a, b, ... (returns a new tensor). |
| ```(.map! #'fn a b ...)``` | ```(map! #'fn a)``` | | | Applies fn elementwise, writing the result to the first tensor (a). |

### Activation Functions

| nnl2 | MAGICL | MATLAB | NumPy | Description |
|------|--------|--------|-------|-------------|
| ```(.relu a)``` | | ```max(0, a)``` | ```np.maximum(0, a)``` | Wise-element Rectified Linear Unit (ReLU) |
| ```(.relu! a)``` | | | ```np.maximum(0, a, out=a)``` | In-place element-wise ReLU |
| ```(.leaky-relu a :alpha 0.01)``` | | ```max(0.01*a, a)``` | ```np.where(a > 0, a, 0.01*a)``` | Wise-element Leaky ReLU with optional alpha parameter (usually 0.01) |
| ```(.leaky-relu! a :alpha 0.01)``` | | | ```np.maximum(0.01*a, a, out=a)``` | Wise-element In-place Leaky ReLU with optional alpha parameter (usually 0.01) | 
