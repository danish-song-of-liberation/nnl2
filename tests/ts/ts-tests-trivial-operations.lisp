(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-trivial-operations.lisp
;; File: ts-tests-trivial-operations.lisp

;; By trivial operations, I mean operations that do not require 
;; any arguments other than the tensor itself (such as exp, log and abs)

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/trivial-operation (&key dtype shape val expected op inplace (tolerance 0.0))
  "Checks the correctness of a trivial operation, that 
   is, one that either does not require any arguments 
   other than the tensor
   
   logic is almost the same as the function #'check-nnl2.hli.ts/operation but for one tensor"
   
  (handler-case
      (let ((tensor-shape (coerce shape 'vector)))
	    (nnl2.hli.ts:tlet* ((tensor (nnl2.hli.ts:full shape :dtype dtype :filler val))
							(result-tensor (funcall op tensor)))
		
		  (nnl2.hli.ts.tests:check-tensor-data 
		    :tensor (if inplace tensor result-tensor)
		    :shape shape
		    :expected-value expected
			:tolerance tolerance)))
	  
	(error (e)
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when you call the function"
		:error-type :error
		:message e 
		:function #'check-nnl2.hli.ts/trivial-operation
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%shape - ~d~%value - ~a~%expected - ~a"
					dtype shape val expected)))))

(defparameter *default-.exp-operation-shape* '(5 5))
(defparameter *default-.log-operation-shape* '(5 5))
(defparameter *default-.abs-operation-shape* '(5 5))

;; -- `.exp` tests section --		
	
(fiveam:test nnl2.hli.ts/.exp/float64
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.exp-operation-shape* :val 2.0d0 :expected 7.38d9 :op #'nnl2.hli.ts:.exp))					
	
(fiveam:test nnl2.hli.ts/.exp/float32
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.exp-operation-shape* :val 2.0s0 :expected 7.389 :op #'nnl2.hli.ts:.exp))			

(fiveam:test nnl2.hli.ts/.exp/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.exp-operation-shape* :val 0 :expected 1 :op #'nnl2.hli.ts:.exp))			

;; -- `.log` tests section --

(fiveam:test nnl2.hli.ts/.log/float64
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.exp-operation-shape* :val 3.0d0 :expected 1.0986122886681098d0 :op #'nnl2.hli.ts:.log))					
	
(fiveam:test nnl2.hli.ts/.log/float32
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.exp-operation-shape* :val 3.0s0 :expected 1.0986123 :op #'nnl2.hli.ts:.log))			

(fiveam:test nnl2.hli.ts/.log/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.exp-operation-shape* :val 3 :expected 1.0d98 :op #'nnl2.hli.ts:.log))		

;; -- `.abs` tests section --

(fiveam:test nnl2.hli.ts/.abs/float64
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.abs-operation-shape* :val -1.0d0 :expected 1.0d0 :op #'nnl2.hli.ts:.abs))					
	
(fiveam:test nnl2.hli.ts/.abs/float32
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.abs-operation-shape* :val -1.0s0 :expected 1.0s0 :op #'nnl2.hli.ts:.abs))			

(fiveam:test nnl2.hli.ts/.abs/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.abs-operation-shape* :val -1 :expected 1 :op #'nnl2.hli.ts:.abs))		
