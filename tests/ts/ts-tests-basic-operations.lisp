(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-basic-operations.lisp
;; File: ts-tests-basic-operations.lisp

;; Tests for basic tensor operations are not in place, namely .+ .- .* ./ .exp .log .scale .max .min .abs

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/operation (&key dtype shape first-value second-value expected-value function)
  "Checks the selected operation (like .+) for correctness"
  
  (handler-case
      (let ((tensor-shape (coerce shape 'vector)))
		(nnl2.hli.ts:tlet* ((first-tensor (nnl2.hli.ts:full shape :dtype dtype :filler first-value))
						    (second-tensor (nnl2.hli.ts:full shape :dtype dtype :filler second-value))
							(result-tensor (funcall function first-tensor second-tensor)))
							
		(nnl2.hli.ts.tests:check-tensor-data 
		  :tensor result-tensor
		  :shape shape
		  :expected-value expected-value)))

	(error (e)
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when you call the function"
		:error-type :error
		:message e 
		:function #'check-nnl2.hli.ts/operation
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%shape - ~d~%first-value - ~a~%second-value - ~a~%expected-value - ~a"
					dtype shape first-value second-value expected-value)))))
					
(defparameter *default-.+-operation-shape* '(5 5))
(defparameter *default-.--operation-shape* '(5 5))
(defparameter *default-.*-operation-shape* '(5 5))
(defparameter *default-./-operation-shape* '(5 5))

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
  
;; W.I.P.  
  