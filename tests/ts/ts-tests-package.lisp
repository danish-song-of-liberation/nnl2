(defpackage :nnl2.hli.ts.tests
  (:use :cl)
  (:export #:nnl2.hli.ts-suite 
		   #:+default-backend-tensor-type+ 
		   #:+enum-int32+ 
		   #:+enum-float32+ 
		   #:+enum-float64+ 
		   #:assert-tensor-properties))
  
(in-package :nnl2.hli.ts.tests)

(fiveam:def-suite nnl2.hli.ts-suite :description "Tests for nnl2.hli.ts (tensor-system)")  

(defconstant +default-backend-tensor-type+ 'sb-sys:system-area-pointer
  "Default tensor type in the lisp environment")

(defconstant +enum-int32+ 0)
(defconstant +enum-float32+ 1)
(defconstant +enum-float64+ 2)
  
(defun assert-tensor-properties (&key tensor expected-shape expected-dtype expected-size)
  "Checks whether the tensor passes basic checks"

  (fiveam:is (typep tensor +default-backend-tensor-type+) "Tensor must be of type SB-SYS:SYSTEM-AREA-POINTE")
  (fiveam:is (equal (nnl2.hli.ts:shape tensor :as :list) expected-shape) (format nil "The tensor shape should be ~a" expected-shape))  
  (fiveam:is (eql (nnl2.hli.ts:dtype tensor) expected-dtype) (format nil "The tensor data type must be ~a" expected-dtype))
  (fiveam:is (= (nnl2.hli.ts:size tensor) expected-size) (format nil "The size of the tensor should be ~a" expected-size)))
  