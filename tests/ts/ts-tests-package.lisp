(defpackage :nnl2.hli.ts.tests
  (:use :cl)
  (:export #:nnl2.hli.ts-suite 
		   #:+default-backend-tensor-type+ 
		   #:+enum-int32+ 
		   #:+enum-float32+ 
		   #:+enum-float64+ 
		   #:assert-tensor-properties
		   #:check-tensor-data))
  
(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-package.lisp
;; File: ts-tests-package.lisp

;; The file contains the definition of the :nnl2.hli.ts.tests package, the 
;; export of functions to it, and the definition of helper functions for the 
;; convenience of the tests

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:def-suite nnl2.hli.ts-suite :description "Tests for nnl2.hli.ts (tensor-system)")  

(defconstant +default-backend-tensor-type+ 'sb-sys:system-area-pointer
  "Default tensor type in the lisp environment")

;; The full definition of an enum with tensor types is something like this:
;; INT32 (0)
;; FLOAT32 (1)
;; FLOAT64 (2)

(defconstant +enum-int32+ 0)
(defconstant +enum-float32+ 1)
(defconstant +enum-float64+ 2)
  
(defun assert-tensor-properties (&key tensor expected-shape expected-dtype expected-size)
  "Checks whether the tensor passes basic checks"

  (fiveam:is (typep tensor +default-backend-tensor-type+) "Tensor must be of type SB-SYS:SYSTEM-AREA-POINTE")
  (fiveam:is (equal (nnl2.hli.ts:shape tensor :as :list) expected-shape) (format nil "The tensor shape should be ~a" expected-shape))  
  (fiveam:is (eql (nnl2.hli.ts:dtype tensor) expected-dtype) (format nil "The tensor data type must be ~a" expected-dtype))
  (fiveam:is (= (nnl2.hli.ts:size tensor) expected-size) (format nil "The size of the tensor should be ~a" expected-size)))
  
(defun check-tensor-data (&key tensor shape (expected-value 0.0d0) (tolerance 0.0))
  "The function recursively traverses the tensor data and checks, using fiveam:is, for similarity to the expected-value"

  (assert (not (null tensor)) nil (format nil "~%An empty shape was passed to the function ~a~%" #'check-tensor-data))

  (let ((length-tensor (length shape)))
    (case length-tensor
	  (0 (fiveam:is (nnl2.tests.utils:approximately-equal tensor expected-value :tolerance tolerance) 
		   "Value mismatch in function ~a: ~a != ~a" #'check-tensor-data tensor expected-value))
		   
	  (1 (dotimes (i (first shape)) 
		   (let ((current-index (nnl2.hli.ts:tref tensor i)))
		     (fiveam:is (nnl2.tests.utils:approximately-equal current-index expected-value :tolerance tolerance) "Value mismatch at index ~a in function ~a: ~a != ~a" 
		       i #'check-tensor-data current-index expected-value))))
			   
	  (otherwise 
		(let ((current-dim (first shape)))
		  (dotimes (i current-dim) ;; Recursive traversal
			(check-tensor-data :tensor (nnl2.hli.ts:tref tensor i) :shape (rest shape) :expected-value expected-value :tolerance tolerance)))))))
	