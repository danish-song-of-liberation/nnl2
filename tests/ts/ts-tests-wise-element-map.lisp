(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-wise-element-map.lisp
;; File: ts-tests-wise-element-map.lisp

;; Contains tests for element-wise application of a function to tensors

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun map-test (&key shape dtype inplace op)
  "Makes a test for element-wise application of a function to a tensor"
 
  (nnl2.tests.utils:start-log-for-test :function #'nnl2.hli.ts:.map :dtype dtype)

  (handler-case
	  (let ((tensor-shape (coerce shape 'vector))
			(lisp-type (nnl2.hli.ts:ts-type-to-lisp dtype)))
			
	    (nnl2.hli.ts:tlet* ((tensor (nnl2.hli.ts:full tensor-shape :dtype dtype :filler (coerce -1 lisp-type)))
							(result-tensor (funcall op #'(lambda (x) (abs x)) tensor)))

		  (nnl2.hli.ts.tests:check-tensor-data 
		    :tensor (if inplace tensor result-tensor)
		    :shape shape
		    :expected-value (coerce 1 lisp-type))
			
		  (nnl2.tests.utils:end-log-for-test :function op :dtype dtype)))	
							
	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function #'nnl2.hli.ts:.map :dtype dtype)
	  
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when you call the function"
		:error-type :error
		:message e 
		:function op
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%shape - ~a~%dtype - ~d"
					shape dtype)))))

(defparameter *default-.map-operation-shape* '(5 5))

;; -- `.map` tests section --	

(fiveam:test nnl2.hli.ts/.map/float64
  (map-test :dtype :float64 :shape *default-.map-operation-shape* :op #'nnl2.hli.ts:.map))		

(fiveam:test nnl2.hli.ts/.map/float32
  (map-test :dtype :float32 :shape *default-.map-operation-shape* :op #'nnl2.hli.ts:.map))	
  
(fiveam:test nnl2.hli.ts/.map/int32
  (map-test :dtype :int32 :shape *default-.map-operation-shape* :op #'nnl2.hli.ts:.map))	
  
;; the continuation of the file is `ts-tests-wise-element-map-inplace.lisp`, which 
;; implements the same but in-place  