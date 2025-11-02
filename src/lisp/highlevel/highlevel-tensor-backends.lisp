(in-package :nnl2.hli.ts)

;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/highlevel-tensor-backends.lisp
;; File: highlevel-tensor-backends.lisp

;; The file contains the implementation of dispatching by tensor implementations

;; I had a choice: expand the automatic dispatching and 
;; backend architecture in C, increasing the code size 
;; and complexity tenfold, or delegate all the logic to 
;; Lisp, leaving only the C-specific capabilities for 
;; backend switching. Unprofessional? Perhaps, but it's 
;; at least better than spending hundreds of hours 
;; completely redesigning the backends for the sake of 
;; a "good architecture" that complicates the structure 
;; a thousandfold. Both options produce the same effect

;; I chose the latter.

;; Note:
;;	 I only left the docstring in functions that will be 
;;	 called regularly. I won't/will leave little documentation 
;;	 in copy-paste functions that do things that are already 
;;	 clear, leaving normal documentation only in functions that 
;;	 may be called regularly in a high-level interface

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun symbol-to-uppercase-string (symbol)
  (string-upcase (symbol-name symbol)))

(defun uppercase-string-to-symbol (upstring)
  (intern (string-downcase upstring) :keyword))
 
(defmacro define-backend-setter (function-name &rest backend-functions)
  "Creates a function for configuring the backend with the specified backend functions"
  (let ((sigsym (gensym "SIG"))
        (namesym (gensym "NAME")))
    `(defun ,function-name (,namesym)
       ,(format nil "Sets new backend for ~A. NAME: New backend symbol (e.g., 'naive, 'avx256, 'blas)	 (Documentation was generated automatically)" function-name)
       (let ((,sigsym (symbol-to-uppercase-string ,namesym)))
         ,@(loop for backend-fn in backend-functions
                 collect `(,backend-fn ,sigsym))))))
  
(define-backend-setter use-backend/tref
  nnl2.ffi:%set-tref-getter-backend
  nnl2.ffi:%set-tref-setter-backend)

(define-backend-setter use-backend/view
  nnl2.ffi:%set-view-backend)

(define-backend-setter use-backend/.abs
  nnl2.ffi:%set-abs-backend)

(define-backend-setter use-backend/.abs!
  nnl2.ffi:%set-abs-inplace-backend)

(define-backend-setter use-backend/full
  nnl2.ffi:%set-inplace-fill-backend)

(define-backend-setter use-backend/empty
  nnl2.ffi:%set-empty-backend)

(define-backend-setter use-backend/zeros
  nnl2.ffi:%set-inplace-fill-backend)

(define-backend-setter use-backend/gemm
  nnl2.ffi:%set-sgemminplace-backend
  nnl2.ffi:%set-dgemminplace-backend
  nnl2.ffi:%set-i32gemminplace-backend)

(define-backend-setter use-backend/+=
  nnl2.ffi:%set-addinplace-backend
  nnl2.ffi:%set-add-incf-inplace-backend
  nnl2.ffi:%set-add-broadcasting-inplace-backend)

(define-backend-setter use-backend/-=
  nnl2.ffi:%set-subinplace-backend
  nnl2.ffi:%set-sub-decf-inplace-backend
  nnl2.ffi:%set-sub-broadcasting-inplace-backend)

(define-backend-setter use-backend/*=
  nnl2.ffi:%set-mulinplace-backend
  nnl2.ffi:%set-mul-mulf-inplace-backend
  nnl2.ffi:%set-mul-broadcasting-inplace-backend)

(define-backend-setter use-backend//!
  nnl2.ffi:%set-divinplace-backend
  nnl2.ffi:%set-div-divf-inplace-backend
  nnl2.ffi:%set-div-broadcasting-inplace-backend)

(define-backend-setter use-backend/^=
  nnl2.ffi:%set-powinplace-backend
  nnl2.ffi:%set-pow-powf-inplace-backend
  nnl2.ffi:%set-pow-broadcasting-inplace-backend)

(define-backend-setter use-backend/.log!
  nnl2.ffi:%set-loginplace-backend)

(define-backend-setter use-backend/scale!
  nnl2.ffi:%set-scaleinplace-backend)

(define-backend-setter use-backend/.min!
  nnl2.ffi:%set-mininplace-backend
  nnl2.ffi:%set-min-minf-inplace-backend
  nnl2.ffi:%set-min-broadcasting-inplace-backend)

(define-backend-setter use-backend/.max!
  nnl2.ffi:%set-maxinplace-backend
  nnl2.ffi:%set-max-maxf-inplace-backend
  nnl2.ffi:%set-max-broadcasting-inplace-backend)

(define-backend-setter use-backend/.+
  nnl2.ffi:%set-add-backend
  nnl2.ffi:%set-add-incf-backend
  nnl2.ffi:%set-add-broadcasting-backend)

(define-backend-setter use-backend/.-
  nnl2.ffi:%set-sub-backend
  nnl2.ffi:%set-sub-decf-backend
  nnl2.ffi:%set-sub-broadcasting-backend)

(define-backend-setter use-backend/.*
  nnl2.ffi:%set-div-backend
  nnl2.ffi:%set-mul-mulf-backend
  nnl2.ffi:%set-mul-broadcasting-backend)

(define-backend-setter use-backend/./
  nnl2.ffi:%set-div-backend
  nnl2.ffi:%set-div-divf-backend
  nnl2.ffi:%set-div-broadcasting-backend)

(define-backend-setter use-backend/.^
  nnl2.ffi:%set-pow-backend
  nnl2.ffi:%set-pow-powf-backend
  nnl2.ffi:%set-pow-broadcasting-backend)

(define-backend-setter use-backend/.log
  nnl2.ffi:%set-log-backend)

(define-backend-setter use-backend/scale
  nnl2.ffi:%set-scale-backend)

(define-backend-setter use-backend/.min
  nnl2.ffi:%set-min-backend
  nnl2.ffi:%set-min-minf-backend
  nnl2.ffi:%set-min-broadcasting-backend)

(define-backend-setter use-backend/.max
  nnl2.ffi:%set-max-backend
  nnl2.ffi:%set-max-maxf-backend
  nnl2.ffi:%set-max-broadcasting-backend)

(define-backend-setter use-backend/hstack
  nnl2.ffi:%set-hstack-backend)

(define-backend-setter use-backend/vstack
  nnl2.ffi:%set-vstack-backend)

(define-backend-setter use-backend/.relu!
  nnl2.ffi:%set-reluinplace-backend)

(define-backend-setter use-backend/.relu
  nnl2.ffi:%set-relu-backend)

(define-backend-setter use-backend/.leaky-relu!
  nnl2.ffi:%set-leakyreluinplace-backend)

(define-backend-setter use-backend/.leaky-relu
  nnl2.ffi:%set-leakyrelu-backend)

(define-backend-setter use-backend/.sigmoid!
  nnl2.ffi:%set-sigmoidinplace-backend)

(define-backend-setter use-backend/.sigmoid
  nnl2.ffi:%set-sigmoid-backend)

(define-backend-setter use-backend/.tanh!
  nnl2.ffi:%set-tanhinplace-backend)

(define-backend-setter use-backend/.tanh
  nnl2.ffi:%set-tanh-backend)

(define-backend-setter use-backend/concat
  nnl2.ffi:%set-concat-backend)

(define-backend-setter use-backend/randn
  nnl2.ffi:%set-randn-backend)

(define-backend-setter use-backend/xavier
  nnl2.ffi:%set-xavier-backend)

(define-backend-setter use-backend/transpose!
  nnl2.ffi:%set-transposeinplace-backend)

(define-backend-setter use-backend/transpose
  nnl2.ffi:%set-transpose-backend)

(define-backend-setter use-backend/sum
  nnl2.ffi:%set-sum-without-axis-backend
  nnl2.ffi:%set-sum-with-axis-backend)

(define-backend-setter use-backend/l2norm
  nnl2.ffi:%set-l2norm-backend)

(define-backend-setter use-backend/copy
  nnl2.ffi:%set-copy-backend)

(define-backend-setter use-backend/reshape
  nnl2.ffi:%set-reshape-backend)

(define-backend-setter use-backend/reinterpret
  nnl2.ffi:%set-reinterpret-backend)
  
(define-backend-setter use-backend/slice
  nnl2.ffi:%set-slice-backend)  
  
(define-backend-setter use-backend/cut
  nnl2.ffi:%set-cut-backend)    
  
(define-backend-setter use-backend/transposition
  nnl2.ffi:%set-transposition-backend)  

(define-backend-setter use-backend/transposition!
  nnl2.ffi:%set-transposition-inplace-backend)      
  
(define-backend-setter use-backend/axpy!
  nnl2.ffi:%set-axpy-inplace-backend) 
  
(define-backend-setter use-backend/axpy
  nnl2.ffi:%set-axpy-backend)   
  
(define-backend-setter use-backend/.neg!
  nnl2.ffi:%set-neginplace-backend)  

(define-backend-setter use-backend/.neg
  nnl2.ffi:%set-neg-backend)    

(defun use-backend/ones (name) (use-backend/full name))
(defun use-backend/full-like (name) (use-backend/full name))  
(defun use-backend/empty-like (name) (use-backend/empty name))  
(defun use-backend/zeros-like (name) (use-backend/zeros name))  
(defun use-backend/ones-like (name) (use-backend/ones name))
(defun use-backend/randn-like (name) (use-backend/randn name))

(defun use-backend (name)
  "Sets a new backend for all supported operations
   name: Backend symbol (e.g., 'naive, 'avx256, 'blas, ...)"
   
  (dolist (backend-function '(use-backend/tref use-backend/view use-backend/+= 
                              use-backend/-= use-backend/*= use-backend//! 
                              use-backend/^= use-backend/.log! use-backend/.max!
                              use-backend/.min! use-backend/scale! use-backend/.+
                              use-backend/.- use-backend/.* use-backend/./
                              use-backend/.^ use-backend/.log use-backend/.min
                              use-backend/.max use-backend/scale use-backend/empty
                              use-backend/.abs use-backend/xavier use-backend/randn
                              use-backend/full use-backend/zeros use-backend/sum
                              use-backend/l2norm use-backend/copy use-backend/gemm
                              use-backend/.relu use-backend/.relu! use-backend/.leaky-relu
                              use-backend/.leaky-relu! use-backend/.sigmoid use-backend/.sigmoid!
                              use-backend/.tanh use-backend/.tanh! use-backend/transpose
                              use-backend/transpose! use-backend/reshape use-backend/reinterpret
							  use-backend/slice use-backend/cut use-backend/transposition
							  use-backend/transposition! use-backend/.neg! use-backend/.neg))
							  
      (funcall backend-function name)))
	  
(defmacro define-backend-getter-setter (getter-name setter-name backend-get-function &key (like nil))
  "Creates getter and setter functions for backend configuration
   getter-name: Name of the getter function
   setter-name: Name of the setter function  
   backend-get-function: CFFI function to get current backend
   like (&key): If specified, creates alias to another getter"
   
  (if like
    `(progn
       (defun ,getter-name ()
         ,(format nil "Gets current backend for ~A. Returns keyword symbol.    (Documentation was generated automatically)" getter-name)
          (,like))
       
       (defun (setf ,getter-name) (name)
         ,(format nil "Sets backend for ~A to NAME.    (Documentation was generated automatically)" getter-name)
          (funcall (function ,setter-name) name)))
    
    `(progn
       (defun ,getter-name ()
         ,(format nil "Gets current backend for ~A. Returns keyword symbol.    (Documentation was generated automatically)" getter-name)
          (uppercase-string-to-symbol (,backend-get-function)))

       (defun (setf ,getter-name) (name)
         ,(format nil "Sets backend for ~A to NAME.    (Documentation was generated automatically)" getter-name)
          (,setter-name name)))))

(define-backend-getter-setter get-backend/tref use-backend/tref nnl2.ffi:%get-tref-getter-backend)
(define-backend-getter-setter get-backend/view use-backend/view nnl2.ffi:%get-view-backend)
(define-backend-getter-setter get-backend/copy use-backend/copy nnl2.ffi:%get-copy-backend)
(define-backend-getter-setter get-backend/empty use-backend/empty nnl2.ffi:%get-empty-backend)
(define-backend-getter-setter get-backend/zeros use-backend/zeros nnl2.ffi:%get-inplace-fill-backend)
(define-backend-getter-setter get-backend/full use-backend/full nnl2.ffi:%get-inplace-fill-backend)
(define-backend-getter-setter get-backend/ones use-backend/ones nnl2.ffi:%get-inplace-fill-backend)
(define-backend-getter-setter get-backend/full-like use-backend/full-like nil :like get-backend/full)
(define-backend-getter-setter get-backend/empty-like use-backend/empty-like nil :like get-backend/empty)
(define-backend-getter-setter get-backend/zeros-like use-backend/zeros-like nil :like get-backend/zeros)
(define-backend-getter-setter get-backend/ones-like use-backend/ones-like nil :like get-backend/ones)
(define-backend-getter-setter get-backend/gemm use-backend/gemm nnl2.ffi:%get-gemm-backend)
(define-backend-getter-setter get-backend/gemm! use-backend/gemm! nnl2.ffi:%get-gemm-backend)
(define-backend-getter-setter get-backend/axpy use-backend/axpy nnl2.ffi:%get-axpy-backend)
(define-backend-getter-setter get-backend/axpy! use-backend/axpy! nnl2.ffi:%get-axpy-inplace-backend)
(define-backend-getter-setter get-backend/+= use-backend/+= nnl2.ffi:%get-addinplace-backend)
(define-backend-getter-setter get-backend/-= use-backend/-= nnl2.ffi:%get-subinplace-backend)
(define-backend-getter-setter get-backend/*= use-backend/*= nnl2.ffi:%get-mulinplace-backend)
(define-backend-getter-setter get-backend//! use-backend//! nnl2.ffi:%get-divinplace-backend)
(define-backend-getter-setter get-backend/^= use-backend/^= nnl2.ffi:%get-powinplace-backend)
(define-backend-getter-setter get-backend/.+ use-backend/.+ nnl2.ffi:%get-add-backend)
(define-backend-getter-setter get-backend/.- use-backend/.- nnl2.ffi:%get-sub-backend)
(define-backend-getter-setter get-backend/.* use-backend/.* nnl2.ffi:%get-mul-backend)
(define-backend-getter-setter get-backend/./ use-backend/./ nnl2.ffi:%get-div-backend)
(define-backend-getter-setter get-backend/.^ use-backend/.^ nnl2.ffi:%get-pow-backend)
(define-backend-getter-setter get-backend/.exp use-backend/.exp nnl2.ffi:%get-exp-backend)
(define-backend-getter-setter get-backend/.exp! use-backend/.exp! nnl2.ffi:%get-expinplace-backend)
(define-backend-getter-setter get-backend/.log use-backend/.log nnl2.ffi:%get-log-backend)
(define-backend-getter-setter get-backend/.log! use-backend/.log! nnl2.ffi:%get-loginplace-backend)
(define-backend-getter-setter get-backend/scale use-backend/scale nnl2.ffi:%get-scale-backend)
(define-backend-getter-setter get-backend/scale! use-backend/scale! nnl2.ffi:%get-scaleinplace-backend)
(define-backend-getter-setter get-backend/.min use-backend/.min nnl2.ffi:%get-min-backend)
(define-backend-getter-setter get-backend/.min! use-backend/.min! nnl2.ffi:%get-mininplace-backend)
(define-backend-getter-setter get-backend/.max use-backend/.max nnl2.ffi:%get-max-backend)
(define-backend-getter-setter get-backend/.max! use-backend/.max! nnl2.ffi:%get-maxinplace-backend)
(define-backend-getter-setter get-backend/.abs use-backend/.abs nnl2.ffi:%get-abs-backend)
(define-backend-getter-setter get-backend/.abs! use-backend/.abs! nnl2.ffi:%get-absinplace-backend)
(define-backend-getter-setter get-backend/hstack use-backend/hstack nnl2.ffi:%get-hstack-backend)
(define-backend-getter-setter get-backend/vstack use-backend/vstack nnl2.ffi:%get-vstack-backend)
(define-backend-getter-setter get-backend/concat use-backend/concat nnl2.ffi:%get-concat-backend)
(define-backend-getter-setter get-backend/.relu use-backend/.relu nnl2.ffi:%get-relu-backend)
(define-backend-getter-setter get-backend/.relu! use-backend/.relu! nnl2.ffi:%get-reluinplace-backend)
(define-backend-getter-setter get-backend/.leaky-relu use-backend/.leaky-relu nnl2.ffi:%get-leakyrelu-backend)
(define-backend-getter-setter get-backend/.leaky-relu! use-backend/.leaky-relu! nnl2.ffi:%get-leakyreluinplace-backend)
(define-backend-getter-setter get-backend/.sigmoid use-backend/.sigmoid nnl2.ffi:%get-sigmoid-backend)
(define-backend-getter-setter get-backend/.sigmoid! use-backend/.sigmoid! nnl2.ffi:%get-sigmoidinplace-backend)
(define-backend-getter-setter get-backend/.tanh use-backend/.tanh nnl2.ffi:%get-tanh-backend)
(define-backend-getter-setter get-backend/.tanh! use-backend/.tanh! nnl2.ffi:%get-tanhinplace-backend)
(define-backend-getter-setter get-backend/randn use-backend/randn nnl2.ffi:%get-randn-backend)
(define-backend-getter-setter get-backend/randn-like use-backend/randn-like nil :like get-backend/randn)
(define-backend-getter-setter get-backend/xavier use-backend/xavier nnl2.ffi:%get-xavier-backend)
(define-backend-getter-setter get-backend/transpose use-backend/transpose nnl2.ffi:%get-transpose-backend)
(define-backend-getter-setter get-backend/transpose! use-backend/transpose! nnl2.ffi:%get-transposeinplace-backend)
(define-backend-getter-setter get-backend/reshape use-backend/reshape nnl2.ffi:%get-reshape-backend)
(define-backend-getter-setter get-backend/reinterpret use-backend/reinterpret nnl2.ffi:%get-reinterpret-backend)
(define-backend-getter-setter get-backend/sum use-backend/sum nnl2.ffi:%get-sum-without-axis-backend)
(define-backend-getter-setter get-backend/slice use-backend/slice nnl2.ffi:%get-slice-backend)
(define-backend-getter-setter get-backend/cut use-backend/cut nnl2.ffi:%get-cut-backend)
(define-backend-getter-setter get-backend/transposition use-backend/transposition nnl2.ffi:%get-transposition-backend)
(define-backend-getter-setter get-backend/transposition! use-backend/transposition! nnl2.ffi:%get-transposition-inplace-backend)
(define-backend-getter-setter get-backend/.neg! use-backend/.neg! nnl2.ffi:%get-neginplace-backend)
(define-backend-getter-setter get-backend/.neg use-backend/.neg nnl2.ffi:%get-neg-backend)

(defun get-backend/norm (&key (p :l2))
  "Gets current backend for norm operation. P: Norm type (:l2 supported)"
  (case p
    (:l2 (uppercase-string-to-symbol (nnl2.ffi:%get-l2norm-backend)))
    (otherwise (error "Unsupported norm type: ~A" p))))

(defun (setf get-backend/norm) (name &key (p :l2))
  "Sets backend for norm operation to NAME. P: Norm type (:l2 supported)"
  (case p
    (:l2 (use-backend/l2norm name))
    (otherwise (error "Unsupported norm type: ~A" p))))
  
(defmacro define-backends-getter (getter-name num-backends-fn backends-fn &key (like nil))
  "Creates a function to get available backends for an operation.
   getter-name: Name of the getter function
   num-backends-fn: CFFI function that returns number of available backends
   backends-fn: CFFI function that returns array of backend names
   like: If specified, creates alias to another getter function"
   
  (if like
    `(defun ,getter-name ()
       ,(format nil "Returns list of available backends for ~A as keyword symbols.    (Documentation was generated automatically)" getter-name)
       (,like))
	   
    `(defun ,getter-name ()
       ,(format nil "Returns list of available backends for ~A as keyword symbols.    (Documentation was generated automatically)" getter-name)
        (let ((num-backends (,num-backends-fn))
              (backends (,backends-fn)))
          (loop for i from 0 below num-backends
                collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))))

(define-backends-getter get-backends/tref 
  nnl2.ffi:%get-tref-getter-num-backends 
  nnl2.ffi:%get-tref-getter-backends)

(define-backends-getter get-backends/view 
  nnl2.ffi:%get-view-num-backends 
  nnl2.ffi:%get-view-backends)

(define-backends-getter get-backends/copy 
  nnl2.ffi:%get-copy-num-backends 
  nnl2.ffi:%get-copy-backends)

(define-backends-getter get-backends/empty 
  nnl2.ffi:%get-empty-num-backends 
  nnl2.ffi:%get-empty-backends)

(define-backends-getter get-backends/zeros 
  nnl2.ffi:%get-inplace-fill-num-backends 
  nnl2.ffi:%get-inplace-fill-backends)

(define-backends-getter get-backends/full 
  nnl2.ffi:%get-inplace-fill-num-backends 
  nnl2.ffi:%get-inplace-fill-backends)

(define-backends-getter get-backends/ones 
  nnl2.ffi:%get-inplace-fill-num-backends 
  nnl2.ffi:%get-inplace-fill-backends)

(define-backends-getter get-backends/full-like :like get-backends/full)
(define-backends-getter get-backends/empty-like :like get-backends/empty)
(define-backends-getter get-backends/zeros-like :like get-backends/zeros)
(define-backends-getter get-backends/ones-like :like get-backends/ones)

(define-backends-getter get-backends/gemm 
  nnl2.ffi:%get-gemm-num-backends 
  nnl2.ffi:%get-gemm-backends)

(define-backends-getter get-backends/gemm! 
  nnl2.ffi:%get-gemm-num-backends 
  nnl2.ffi:%get-gemm-backends)

(define-backends-getter get-backends/axpy 
  nnl2.ffi:%get-axpy-num-backends 
  nnl2.ffi:%get-axpy-backends)

(define-backends-getter get-backends/axpy! 
  nnl2.ffi:%get-axpy-inplace-num-backends 
  nnl2.ffi:%get-axpy-inplace-backends)

(define-backends-getter get-backends/+= 
  nnl2.ffi:%get-addinplace-num-backends 
  nnl2.ffi:%get-addinplace-backends)

(define-backends-getter get-backends/-= 
  nnl2.ffi:%get-subinplace-num-backends 
  nnl2.ffi:%get-subinplace-backends)

(define-backends-getter get-backends/*= 
  nnl2.ffi:%get-mulinplace-num-backends 
  nnl2.ffi:%get-mulinplace-backends)

(define-backends-getter get-backends//! 
  nnl2.ffi:%get-divinplace-num-backends 
  nnl2.ffi:%get-divinplace-backends)

(define-backends-getter get-backends/^= 
  nnl2.ffi:%get-powinplace-num-backends 
  nnl2.ffi:%get-powinplace-backends)

(define-backends-getter get-backends/.+ 
  nnl2.ffi:%get-add-num-backends 
  nnl2.ffi:%get-add-backends)

(define-backends-getter get-backends/.- 
  nnl2.ffi:%get-sub-num-backends 
  nnl2.ffi:%get-sub-backends)

(define-backends-getter get-backends/.* 
  nnl2.ffi:%get-mul-num-backends 
  nnl2.ffi:%get-mul-backends)

(define-backends-getter get-backends/./ 
  nnl2.ffi:%get-div-num-backends 
  nnl2.ffi:%get-div-backends)

(define-backends-getter get-backends/.^ 
  nnl2.ffi:%get-pow-num-backends 
  nnl2.ffi:%get-pow-backends)

(define-backends-getter get-backends/.exp 
  nnl2.ffi:%get-exp-num-backends 
  nnl2.ffi:%get-exp-backends)

(define-backends-getter get-backends/.exp! 
  nnl2.ffi:%get-expinplace-num-backends 
  nnl2.ffi:%get-expinplace-backends)

(define-backends-getter get-backends/.log 
  nnl2.ffi:%get-log-num-backends 
  nnl2.ffi:%get-log-backends)

(define-backends-getter get-backends/.log! 
  nnl2.ffi:%get-loginplace-num-backends 
  nnl2.ffi:%get-loginplace-backends)

(define-backends-getter get-backends/scale 
  nnl2.ffi:%get-scale-num-backends 
  nnl2.ffi:%get-scale-backends)

(define-backends-getter get-backends/scale! 
  nnl2.ffi:%get-scaleinplace-num-backends 
  nnl2.ffi:%get-scaleinplace-backends)

(define-backends-getter get-backends/.min 
  nnl2.ffi:%get-min-num-backends 
  nnl2.ffi:%get-min-backends)

(define-backends-getter get-backends/.min! 
  nnl2.ffi:%get-mininplace-num-backends 
  nnl2.ffi:%get-mininplace-backends)

(define-backends-getter get-backends/.max 
  nnl2.ffi:%get-max-num-backends 
  nnl2.ffi:%get-max-backends)

(define-backends-getter get-backends/.max! 
  nnl2.ffi:%get-maxinplace-num-backends 
  nnl2.ffi:%get-maxinplace-backends)

(define-backends-getter get-backends/.abs 
  nnl2.ffi:%get-abs-num-backends 
  nnl2.ffi:%get-abs-backends)

(define-backends-getter get-backends/.abs! 
  nnl2.ffi:%get-absinplace-num-backends 
  nnl2.ffi:%get-absinplace-backends)

(define-backends-getter get-backends/hstack 
  nnl2.ffi:%get-hstack-num-backends 
  nnl2.ffi:%get-hstack-backends)

(define-backends-getter get-backends/vstack 
  nnl2.ffi:%get-vstack-num-backends 
  nnl2.ffi:%get-vstack-backends)

(define-backends-getter get-backends/concat 
  nnl2.ffi:%get-concat-num-backends 
  nnl2.ffi:%get-concat-backends)

(define-backends-getter get-backends/.relu 
  nnl2.ffi:%get-relu-num-backends 
  nnl2.ffi:%get-relu-backends)

(define-backends-getter get-backends/.relu! 
  nnl2.ffi:%get-reluinplace-num-backends 
  nnl2.ffi:%get-reluinplace-backends)

(define-backends-getter get-backends/.leaky-relu 
  nnl2.ffi:%get-leakyrelu-num-backends 
  nnl2.ffi:%get-leakyrelu-backends)

(define-backends-getter get-backends/.leaky-relu! 
  nnl2.ffi:%get-leakyreluinplace-num-backends 
  nnl2.ffi:%get-leakyreluinplace-backends)

(define-backends-getter get-backends/.sigmoid 
  nnl2.ffi:%get-sigmoid-num-backends 
  nnl2.ffi:%get-sigmoid-backends)

(define-backends-getter get-backends/.sigmoid! 
  nnl2.ffi:%get-sigmoidinplace-num-backends 
  nnl2.ffi:%get-sigmoidinplace-backends)

(define-backends-getter get-backends/.tanh 
  nnl2.ffi:%get-tanh-num-backends 
  nnl2.ffi:%get-tanh-backends)

(define-backends-getter get-backends/.tanh! 
  nnl2.ffi:%get-tanhinplace-num-backends 
  nnl2.ffi:%get-tanhinplace-backends)

(define-backends-getter get-backends/randn 
  nnl2.ffi:%get-randn-num-backends 
  nnl2.ffi:%get-randn-backends)

(define-backends-getter get-backends/randn-like :like get-backends/randn)

(define-backends-getter get-backends/xavier 
  nnl2.ffi:%get-xavier-num-backends 
  nnl2.ffi:%get-xavier-backends)

(define-backends-getter get-backends/transpose 
  nnl2.ffi:%get-transpose-num-backends 
  nnl2.ffi:%get-transpose-backends)

(define-backends-getter get-backends/transpose! 
  nnl2.ffi:%get-transposeinplace-num-backends 
  nnl2.ffi:%get-transposeinplace-backends)

(define-backends-getter get-backends/reshape 
  nnl2.ffi:%get-reshape-num-backends 
  nnl2.ffi:%get-reshape-backends)

(define-backends-getter get-backends/reinterpret 
  nnl2.ffi:%get-reinterpret-num-backends 
  nnl2.ffi:%get-reinterpret-backends)

(define-backends-getter get-backends/sum 
  nnl2.ffi:%get-sum-without-axis-num-backends 
  nnl2.ffi:%get-sum-without-axis-backends)

(define-backends-getter get-backends/l2norm 
  nnl2.ffi:%get-l2norm-num-backends 
  nnl2.ffi:%get-l2norm-backends)
  
(define-backends-getter get-backends/slice
  nnl2.ffi:%get-slice-num-backends
  nnl2.ffi:%get-slice-backends)
  
(define-backends-getter get-backends/cut
  nnl2.ffi:%get-cut-num-backends
  nnl2.ffi:%get-cut-backends)  
  
(define-backends-getter get-backends/transposition
  nnl2.ffi:%get-transposition-num-backends
  nnl2.ffi:%get-transposition-backends)  
  
(define-backends-getter get-backends/transposition!
  nnl2.ffi:%get-transposition-inplace-num-backends
  nnl2.ffi:%get-transposition-inplace-backends)    

(define-backends-getter get-backends/.neg!
  nnl2.ffi:%get-neginplace-num-backends
  nnl2.ffi:%get-neginplace-backends) 
  
(define-backends-getter get-backends/.neg
  nnl2.ffi:%get-neg-num-backends
  nnl2.ffi:%get-neg-backends) 
  
(defun get-backends/norm (&key (p :l2))
  "Returns list of available backends for norm operation. 
   P: Norm type (:l2 supported)"
  (case p
    (:l2 (get-backends/l2norm))
    (otherwise (error "Unsupported norm type: ~A" p))))  
	
(defmacro define-with-backend (macro-name getter-name)
  "Creates a with-backend macro that temporarily sets the backend for an operation
   macro-name: Name of the macro to create
   getter-name: Name of the getter function for the backend"
   
  (let ((old-backend-sym (gensym "OLD-BACKEND-"))
        (name-sym (gensym "NAME-")))
    `(defmacro ,macro-name (,name-sym &body body)
       ,(format nil "Temporarily sets backend for ~A during &body execution    (Documentation was generated automatically)" getter-name)
       (let ((,old-backend-sym (gensym "OLD-BACKEND-")))
         `(let ((,,old-backend-sym (,',getter-name)))
            (unwind-protect
                (progn
                  (setf (,',getter-name) ,,name-sym)
                  ,@body)
              (setf (,',getter-name) ,,old-backend-sym)))))))

(define-with-backend with-backend/tref get-backend/tref)
(define-with-backend with-backend/view get-backend/view)
(define-with-backend with-backend/copy get-backend/copy)
(define-with-backend with-backend/empty get-backend/empty)
(define-with-backend with-backend/zeros get-backend/zeros)
(define-with-backend with-backend/full get-backend/full)
(define-with-backend with-backend/ones get-backend/ones)
(define-with-backend with-backend/full-like get-backend/full-like)
(define-with-backend with-backend/empty-like get-backend/empty-like)
(define-with-backend with-backend/zeros-like get-backend/zeros-like)
(define-with-backend with-backend/ones-like get-backend/ones-like)
(define-with-backend with-backend/gemm get-backend/gemm)
(define-with-backend with-backend/gemm! get-backend/gemm!)
(define-with-backend with-backend/axpy get-backend/axpy)
(define-with-backend with-backend/axpy! get-backend/axpy!)
(define-with-backend with-backend/+= get-backend/+=)
(define-with-backend with-backend/-= get-backend/-=)
(define-with-backend with-backend/*= get-backend/*=)
(define-with-backend with-backend//! get-backend//!)
(define-with-backend with-backend/^= get-backend/^=)
(define-with-backend with-backend/.+ get-backend/.+)
(define-with-backend with-backend/.- get-backend/.-)
(define-with-backend with-backend/.* get-backend/.*)
(define-with-backend with-backend/./ get-backend/./)
(define-with-backend with-backend/.^ get-backend/.^)
(define-with-backend with-backend/.exp get-backend/.exp)
(define-with-backend with-backend/.exp! get-backend/.exp!)
(define-with-backend with-backend/.log get-backend/.log)
(define-with-backend with-backend/.log! get-backend/.log!)
(define-with-backend with-backend/scale get-backend/scale)
(define-with-backend with-backend/scale! get-backend/scale!)
(define-with-backend with-backend/.min get-backend/.min)
(define-with-backend with-backend/.min! get-backend/.min!)
(define-with-backend with-backend/.max get-backend/.max)
(define-with-backend with-backend/.max! get-backend/.max!)
(define-with-backend with-backend/.abs get-backend/.abs)
(define-with-backend with-backend/.abs! get-backend/.abs!)
(define-with-backend with-backend/hstack get-backend/hstack)
(define-with-backend with-backend/vstack get-backend/vstack)
(define-with-backend with-backend/concat get-backend/concat)
(define-with-backend with-backend/.relu get-backend/.relu)
(define-with-backend with-backend/.relu! get-backend/.relu!)
(define-with-backend with-backend/.leaky-relu get-backend/.leaky-relu)
(define-with-backend with-backend/.leaky-relu! get-backend/.leaky-relu!)
(define-with-backend with-backend/.sigmoid get-backend/.sigmoid)
(define-with-backend with-backend/.sigmoid! get-backend/.sigmoid!)
(define-with-backend with-backend/.tanh get-backend/.tanh)
(define-with-backend with-backend/.tanh! get-backend/.tanh!)
(define-with-backend with-backend/randn get-backend/randn)
(define-with-backend with-backend/xavier get-backend/xavier)
(define-with-backend with-backend/transpose get-backend/transpose)
(define-with-backend with-backend/transpose! get-backend/transpose!)
(define-with-backend with-backend/reshape get-backend/reshape)
(define-with-backend with-backend/reinterpret get-backend/reinterpret)
(define-with-backend with-backend/sum get-backend/sum)
(define-with-backend with-backend/norm get-backend/norm)
(define-with-backend with-backend/slice get-backend/slice)
(define-with-backend with-backend/cut get-backend/cut)
(define-with-backend with-backend/transposition get-backend/transposition)
(define-with-backend with-backend/transposition! get-backend/transposition!)
(define-with-backend with-backend/.neg! get-backend/.neg!)
(define-with-backend with-backend/.neg get-backend/.neg)
		 