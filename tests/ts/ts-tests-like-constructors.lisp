(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-like-constructors.lisp
;; File: ts-tests-like-constructors.lisp

;; File contains tests with like constructors (such as zeros-like and so on)

;; Note: 
;;   Function `check-nnl2.hli.ts/trivial-operation` from the file
;;  `ts-tests-trivial-operations.lisp` will be used for tests with 
;;   zeros and ones

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/empty-like (&key dtype int-dtype shape fill)
  "Checks that the `nnl2.hli.ts:empty-like` function creates a tensor with the specified properties.
   It only checks the metadata because there is no tensor to check the contents of"
  
  (nnl2.tests.utils:start-log-for-test :function #'nnl2.hli.ts:empty-like :dtype dtype) 
  
  (handler-case
      (let ((expected-size (apply #'* shape))
		    (expected-shape (coerce shape 'vector)))
		
		(nnl2.hli.ts:tlet* ((tensor (nnl2.hli.ts:zeros expected-shape :dtype dtype))
						    (result (nnl2.hli.ts:empty-like tensor)))
							
		  (nnl2.hli.ts.tests:assert-tensor-properties 
			:tensor result
			:expected-size expected-size
			:expected-dtype dtype
			:expected-shape shape)
			
		  (fiveam:is (= (nnl2.hli.ts:int-dtype tensor) int-dtype) "Checking if the data type matches the enum")
		  
		  (nnl2.tests.utils:end-log-for-test :function #'nnl2.hli.ts:empty-like :dtype dtype)))
		
	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function #'nnl2.hli.ts:empty-like :dtype dtype) 
	
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when creating an `empty-like` tensor"
		:error-type :error
		:message e 
		:function #'check-nnl2.hli.ts/empty-like 
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%int-dtype - ~d~%shape - ~a~%fill - ~a"
					dtype int-dtype shape fill)))))
					
(defun check-nnl2.hli.ts/full-like (&key dtype shape fill)
  "Checks that the `nnl2.hli.ts:full-like` function creates a tensor with the specified properties"
  
  (nnl2.tests.utils:start-log-for-test :function #'nnl2.hli.ts:full-like :dtype dtype) 
  
  (handler-case
      (let* ((expected-size (apply #'* shape))
		     (expected-shape (coerce shape 'vector))
			 (lisp-type (nnl2.hli.ts:type/nnl2->lisp dtype))
			 (filler (coerce fill lisp-type)))	 
		
		(nnl2.hli.ts:tlet* ((tensor (nnl2.hli.ts:zeros expected-shape :dtype dtype))
						    (result (nnl2.hli.ts:full-like tensor :filler filler)))
		 
		  (nnl2.hli.ts.tests:check-tensor-data 
		    :tensor result
		    :shape shape
		    :expected-value filler)
		  
		  (nnl2.tests.utils:end-log-for-test :function #'nnl2.hli.ts:full-like :dtype dtype)))
		
	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function #'nnl2.hli.ts:full-like :dtype dtype) 
	
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when creating an `full-like` tensor"
		:error-type :error
		:message e 
		:function #'check-nnl2.hli.ts/full-like 
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%shape - ~a~%fill - ~a"
					dtype shape fill)))))					

(defparameter *default-zeros-like-operation-shape* '(5 5))	
(defparameter *default-ones-like-operation-shape* '(5 5))
(defparameter *default-empty-like-operation-shape* '(5 5))	
(defparameter *default-full-like-operation-shape* '(5 5))	

;; -- `zeros-like` tests section --

(fiveam:test nnl2.hli.ts/zeros-like/float64
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-zeros-like-operation-shape* :val 3.0d0 :expected 0.0d0 :op #'nnl2.hli.ts:zeros-like))  
	
(fiveam:test nnl2.hli.ts/zeros-like/float32
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-zeros-like-operation-shape* :val 3.0s0 :expected 0.0s0 :op #'nnl2.hli.ts:zeros-like))

(fiveam:test nnl2.hli.ts/zeros-like/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-zeros-like-operation-shape* :val 3 :expected 0 :op #'nnl2.hli.ts:zeros-like))  
	  
;; -- `ones-like` tests section --	

(fiveam:test nnl2.hli.ts/ones-like/float64
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-ones-like-operation-shape* :val 4.0d0 :expected 1.0d0 :op #'nnl2.hli.ts:ones-like))  
	
(fiveam:test nnl2.hli.ts/ones-like/float32
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-ones-like-operation-shape* :val 4.0s0 :expected 1.0s0 :op #'nnl2.hli.ts:ones-like))

(fiveam:test nnl2.hli.ts/ones-like/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-ones-like-operation-shape* :val 4 :expected 1 :op #'nnl2.hli.ts:ones-like))

;; -- `empty-like` tests section --	

(fiveam:test nnl2.hli.ts/empty-like/float64
  (check-nnl2.hli.ts/empty-like :dtype :float64 :shape *default-empty-like-operation-shape* :fill 5.0d0 :int-dtype nnl2.hli.ts.tests:+enum-float64+))  
	
(fiveam:test nnl2.hli.ts/empty-like/float32
  (check-nnl2.hli.ts/empty-like :dtype :float32 :shape *default-empty-like-operation-shape* :fill 5.0s0 :int-dtype nnl2.hli.ts.tests:+enum-float32+))

(fiveam:test nnl2.hli.ts/empty-like/int32
  (check-nnl2.hli.ts/empty-like :dtype :int32 :shape *default-empty-like-operation-shape* :fill 5 :int-dtype nnl2.hli.ts.tests:+enum-int32+))  	
	
;; -- `full-like` tests section --	

(fiveam:test nnl2.hli.ts/full-like/float64
  (check-nnl2.hli.ts/full-like :dtype :float64 :shape *default-full-like-operation-shape* :fill 2.0d0))  
	
(fiveam:test nnl2.hli.ts/full-like/float32
  (check-nnl2.hli.ts/full-like :dtype :float32 :shape *default-full-like-operation-shape* :fill 2.0s0))

(fiveam:test nnl2.hli.ts/full-like/int32
  (check-nnl2.hli.ts/full-like :dtype :int32 :shape *default-full-like-operation-shape* :fill 2))    
		