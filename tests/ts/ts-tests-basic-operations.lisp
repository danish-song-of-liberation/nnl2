(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-basic-operations.lisp
;; File: ts-tests-basic-operations.lisp

;; Tests for basic tensor operations are not in place, namely .+ .- .* ./ .^ .max .min 

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/operation (&key dtype shape first-value second-value expected-value function inplace (tolerance 0.0))
  "Checks the selected operation (like .+) for correctness.
   it also supports inplace functions by specifying a key.	
	
   First, the function creates three tensors, the first one 
   is filled with the first value, the second one is filled 
   with the second value, and the third one is filled with 
   the result of applying the function to them (if the operation 
   is inplace, then nil will be placed there)
   
   Then, the resulting tensor is passed to 
   nnl2.hli.ts.tests:check-tensor-data, which checks 
   the contents of the tensor. If the operation is 
   inplace, then the resulting tensor will be the 
   first tensor, otherwise it will be the result variable
   
   dtype - Type of tensor
   shape - Tensor shape as a list (e.g. '(3 2 1))
   first-value - Value that the first tensor will be filled with
   second-value - Value that the second tensor will be filled with
   expected-value - Value that is expected in the new tensor
   function - A function used to create a result tensor
   inplace - A predicate that determines whether the passed function is in-place or not
   tolerance - Ignore the spread
   
   How ironic, isn't it? 
   Docstring is bigger than the function itself"
   
  (nnl2.tests.utils:start-log-for-test :function function :dtype dtype)
  
  (handler-case
      (let ((tensor-shape (coerce shape 'vector)))
		(nnl2.hli.ts:tlet* ((first-tensor (nnl2.hli.ts:full shape :dtype dtype :filler first-value))
						    (second-tensor (nnl2.hli.ts:full shape :dtype dtype :filler second-value))
							(result-tensor (funcall function first-tensor second-tensor)))		
										
		  (nnl2.hli.ts.tests:check-tensor-data 
		    :tensor (if inplace first-tensor result-tensor)
		    :shape shape
		    :expected-value expected-value
			:tolerance tolerance)
			
		  (nnl2.tests.utils:end-log-for-test :function function :dtype dtype)))

	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function function :dtype dtype)
	
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when you call the function"
		:error-type :error
		:message e 
		:function function
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%shape - ~d~%first-value - ~a~%second-value - ~a~%expected-value - ~a"
					dtype shape first-value second-value expected-value)))))
					


;; '(5 5) is just a test form; it contains nothing unusual					
					
(defparameter *default-.+-operation-shape* '(5 5))
(defparameter *default-.--operation-shape* '(5 5))
(defparameter *default-.*-operation-shape* '(5 5))
(defparameter *default-./-operation-shape* '(5 5))
(defparameter *default-.^-operation-shape* '(5 5))
(defparameter *default-.max-operation-shape* '(5 5))
(defparameter *default-.min-operation-shape* '(5 5))
(defparameter *default-axpy-operation-shape* '(5 5))

;; -- `.+` tests section --		
	
(fiveam:test nnl2.hli.ts/.+/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-.+-operation-shape* :first-value 2.0d0 :second-value 1.0d0 :expected-value 3.0d0 :function #'nnl2.hli.ts:.+))					
	
(fiveam:test nnl2.hli.ts/.+/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-.+-operation-shape* :first-value 2.0s0 :second-value 1.0s0 :expected-value 3.0s0 :function #'nnl2.hli.ts:.+))					
						
(fiveam:test nnl2.hli.ts/.+/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-.+-operation-shape* :first-value 2 :second-value 1 :expected-value 3 :function #'nnl2.hli.ts:.+))					
	
;; -- `.-` tests section --		
	
(fiveam:test nnl2.hli.ts/.-/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-.--operation-shape* :first-value 3.0d0 :second-value 2.0d0 :expected-value 1.0d0 :function #'nnl2.hli.ts:.-))					
	
(fiveam:test nnl2.hli.ts/.-/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-.--operation-shape* :first-value 3.0s0 :second-value 2.0s0 :expected-value 1.0s0 :function #'nnl2.hli.ts:.-))					
						
(fiveam:test nnl2.hli.ts/.-/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-.--operation-shape* :first-value 3 :second-value 2 :expected-value 1 :function #'nnl2.hli.ts:.-))					
	
;; -- `.*` tests section --		
	
(fiveam:test nnl2.hli.ts/.*/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-.*-operation-shape* :first-value 3.0d0 :second-value 2.0d0 :expected-value 6.0d0 :function #'nnl2.hli.ts:.*))					
	
(fiveam:test nnl2.hli.ts/.*/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-.*-operation-shape* :first-value 3.0s0 :second-value 2.0s0 :expected-value 6.0s0 :function #'nnl2.hli.ts:.*))					
						
(fiveam:test nnl2.hli.ts/.*/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-.*-operation-shape* :first-value 3 :second-value 2 :expected-value 6 :function #'nnl2.hli.ts:.*))					

;; -- `./` tests section --		
																																		
(fiveam:test nnl2.hli.ts/.//float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-./-operation-shape* :first-value 10.0d0 :second-value 2.0d0 :expected-value 5.0d0 :function #'nnl2.hli.ts:./))					
	
(fiveam:test nnl2.hli.ts/.//float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-./-operation-shape* :first-value 10.0s0 :second-value 2.0s0 :expected-value 5.0s0 :function #'nnl2.hli.ts:./))					
						
(fiveam:test nnl2.hli.ts/.//int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-./-operation-shape* :first-value 10 :second-value 2 :expected-value 5 :function #'nnl2.hli.ts:./))			
  
;; -- `.^` tests section --

(fiveam:test nnl2.hli.ts/.^/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-.^-operation-shape* :first-value 2.0d0 :second-value 3.0d0 :expected-value 8.0d0 :function #'nnl2.hli.ts:.^))
   
(fiveam:test nnl2.hli.ts/.^/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-.^-operation-shape* :first-value 2.0s0 :second-value 3.0s0 :expected-value 8.0s0 :function #'nnl2.hli.ts:.^))   
   
(fiveam:test nnl2.hli.ts/.^/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-.^-operation-shape* :first-value 2 :second-value 3 :expected-value 8 :function #'nnl2.hli.ts:.^))        
 
;; -- `.max` tests section --
 
(fiveam:test nnl2.hli.ts/.max/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-.max-operation-shape* :first-value 2.0d0 :second-value 3.0d0 :expected-value 3.0d0 :function #'nnl2.hli.ts:.max))
   
(fiveam:test nnl2.hli.ts/.max/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-.max-operation-shape* :first-value 2.0s0 :second-value 3.0s0 :expected-value 3.0s0 :function #'nnl2.hli.ts:.max))   
   
(fiveam:test nnl2.hli.ts/.max/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-.max-operation-shape* :first-value 2 :second-value 3 :expected-value 3 :function #'nnl2.hli.ts:.max))      
 
;; -- `.min` tests section -- 
 
(fiveam:test nnl2.hli.ts/.min/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-.min-operation-shape* :first-value 2.0d0 :second-value 3.0d0 :expected-value 2.0d0 :function #'nnl2.hli.ts:.min))
   
(fiveam:test nnl2.hli.ts/.min/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-.min-operation-shape* :first-value 2.0s0 :second-value 3.0s0 :expected-value 2.0s0 :function #'nnl2.hli.ts:.min))   
   
(fiveam:test nnl2.hli.ts/.min/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-.min-operation-shape* :first-value 2 :second-value 3 :expected-value 2 :function #'nnl2.hli.ts:.min))      

;; -- `axpy` tests section -- 

;; By default, the alpha key for axpy is 1.0, and it would 
;; be impractical to create a separate test function for this 
;; purpose, so I decided to include axpy in the existing test 
;; functions without specifying the key.

(fiveam:test nnl2.hli.ts/axpy/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-axpy-operation-shape* :first-value 2.0d0 :second-value 3.0d0 :expected-value 5.0d0 :function #'nnl2.hli.ts:axpy))
   
(fiveam:test nnl2.hli.ts/axpy/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-axpy-operation-shape* :first-value 2.0s0 :second-value 3.0s0 :expected-value 5.0s0 :function #'nnl2.hli.ts:axpy))   

(fiveam:test nnl2.hli.ts/axpy/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-axpy-operation-shape* :first-value 2 :second-value 3 :expected-value 5 :function #'nnl2.hli.ts:axpy))
   
;; This is not exactly a DRY violation, I could create a 
;; macro for faster creation, but this is already a KISS violation

;; A continuation of this file is ts-tests-basic-operations-inplace.lisp, 
;; which does something similar but with in-place operations
