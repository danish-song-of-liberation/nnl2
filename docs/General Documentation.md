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

|  nnl2               | Magicl            | MATLAB          | NumPy                         | Description |
|:-------------------:|:-----------------:|:---------------:|:-----------------------------:|:------------|
| ```(rank a)```  | ```(order a)```   | ```ndims(a)```  | ```ndim(a)``` or ```a.ndim``` | Get the number of dimensions of the array. |
| ```(size a)```      | ```(size a)```    | ```numel(a)```  | ```size(a)``` or ```a.size```  | Get the number of elements of the array. |
| ```(shape a)``` | ```(shape a)```   | ```size(a)```   | ```shape(a)``` or ```a.shape``` | Get the shape of the array. |
| yet nope ffi        | ```(tref a 1 4)``` | ```a(2,5)```   | ```a[1, 4]```                 | Get the element in the second row, fifth column of the array. |

### Constructors 

| nnl2 | MAGICL | MATLAB | NumPy | Desctiption |
|------|--------|--------|-------|-------------|
| yet nope | ```(from-list '(1d0 2d0 3d0 4d0 5d0 6d0) '(2 3))``` | ```[ 1 2 3; 4 5 6 ]``` | ```array([[1.,2.,3.], [4.,5.,6.]])``` | Create a 2x3 matrix from given elements. |
| ```(zeros #(2 3 4))```| ```(zeros '(2 3 4)) or (const 0d0 '(2 3 4))``` | ```zeros(2,3,4)``` | ```zeros((2,3,4))``` |  Create a 2x3x4 dimensional array of zeroes of double-float element type. |
| ```(ones #(3 4))``` | ```(ones '(3 4))``` or ```(const 1d0 '(3 4))``` |  ```ones(3,4)``` | ```ones((3,4))``` | Create a 3x4 dimensional array of ones of double-float element type.}

###  Basic Operations 

| nnl2 | MAGICL | MATLAB | NumPy | Description |
|------|--------|--------|-------|-------------|
| ```(gemm a b)``` | ```(@ a b)``` | ```a * b``` | ```a @ b``` | Matrix multiplication |
| ```(+ a b)``` | ```(.+ a b)``` | ```a + b``` | ```a + b``` | 	Element-wise add |
| ```(- a b)``` | ```(.- a b)``` | ```a - b``` | ```a - b``` | Element-wise subtract |

