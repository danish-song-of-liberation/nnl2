(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-basic-tensors.lisp
;; File: ts-tests-basic-tensors.lisp

;; The file contains tests for creating basic tensors using the :nnl2.hli.ts package, 
;; specifically tests for the `empty`, `zeros`, `ones`, and `full` functions

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/empty (&key dtype int-dtype shape)
  "Checks that the `nnl2.hli.ts:empty` function creates a tensor with the specified properties.
   It only checks the metadata because there is no tensor to check the contents of"
  
  (nnl2.tests.utils:start-log-for-test :function #'nnl2.hli.ts:empty :dtype dtype) 
  
  (handler-case
      (let ((expected-size (apply #'* shape))
		    (expected-shape (coerce shape 'vector)))
		
		(nnl2.hli.ts:tlet ((tensor (nnl2.hli.ts:empty expected-shape :dtype dtype)))
		  (nnl2.hli.ts.tests:assert-tensor-properties 
			:tensor tensor 
			:expected-size expected-size
			:expected-dtype dtype
			:expected-shape shape)
		 
		  (fiveam:is (= (nnl2.hli.ts:int-dtype tensor) int-dtype) "Checking if the data type matches the enum")
		  
		  (nnl2.tests.utils:end-log-for-test :function #'nnl2.hli.ts:empty :dtype dtype)))
		
	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function #'nnl2.hli.ts:empty :dtype dtype) 
	
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when creating an `empty` tensor"
		:error-type :error
		:message e 
		:function #'check-nnl2.hli.ts/empty 
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%int-dtype - ~d~%shape - ~a"
					dtype int-dtype shape)))))

(defun check-nnl2.hli.ts/check-value (&key dtype int-dtype shape expected-value tensor-func)
  "Checks that the `nnl2.hli.ts:zeros` and `nnl2.hli.ts:ones` function creates a tensor with the specified properties
   Checking `zeros` and `ones` has been moved to a single generalized function to avoid DRY"
  
  (nnl2.tests.utils:start-log-for-test :function tensor-func :dtype dtype) 
  
  (handler-case
      (let ((expected-size (apply #'* shape))
			(expected-shape (coerce shape 'vector)))
			
		(nnl2.hli.ts:tlet ((tensor (case tensor-func 
								     (:zeros (nnl2.hli.ts:zeros expected-shape :dtype dtype)) 
									 (:ones (nnl2.hli.ts:ones expected-shape :dtype dtype))
									 (t (error "~%tensor-funct Cannot be ~a~% It can be :zeros or :ones~%" tensor-func)))))
									 
		  (nnl2.hli.ts.tests:assert-tensor-properties
			:tensor tensor
			:expected-size expected-size
			:expected-dtype dtype
			:expected-shape shape)
			
		  (fiveam:is (= (nnl2.hli.ts:int-dtype tensor) int-dtype) "Checking if the data type matches the enum")

		  (nnl2.hli.ts.tests:check-tensor-data 
		    :tensor tensor
			:shape shape
			:expected-value expected-value)
			
		  (nnl2.tests.utils:end-log-for-test :function tensor-func :dtype dtype)))
	  
	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function tensor-func :dtype dtype) 
	
	  (nnl2.tests.utils:throw-error
	    :documentation "Throws an informative error if an error occurs when creating an `zeros` or `ones` tensor"
		:error-type :error
		:message e
		:function #'check-nnl2.hli.ts/check-value
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%int-dtype - ~d~%shape - ~a"
					dtype int-dtype shape)))))
					
(defun check-nnl2.hli.ts/full (&key dtype int-dtype shape filler)
  "Checks that the `nnl2.hli.ts:full` function creates a tensor with the specified properties"
  
  (nnl2.tests.utils:start-log-for-test :function #'nnl2.hli.ts:full :dtype dtype) 
  
  (handler-case
      (let ((expected-size (apply #'* shape))
			(expected-shape (coerce shape 'vector)))
			
		(nnl2.hli.ts:tlet ((tensor (nnl2.hli.ts:full expected-shape :filler filler :dtype dtype)))
		  (nnl2.hli.ts.tests:assert-tensor-properties
			:tensor tensor
			:expected-size expected-size
			:expected-dtype dtype
			:expected-shape shape)
			
		  (fiveam:is (= (nnl2.hli.ts:int-dtype tensor) int-dtype) "Checking if the data type matches the enum")

		  (nnl2.hli.ts.tests:check-tensor-data 
		    :tensor tensor
			:shape shape
			:expected-value filler)
			
		  (nnl2.tests.utils:end-log-for-test :function #'nnl2.hli.ts:full :dtype dtype)))
	  
	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function #'nnl2.hli.ts:full :dtype dtype) 
	
	  (nnl2.tests.utils:throw-error
	    :documentation "Throws an informative error if an error occurs when creating an `full` tensor"
		:error-type :error
		:message e
		:function #'check-nnl2.hli.ts/zeros 
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%int-dtype - ~d~%shape - ~a"
					dtype int-dtype shape)))))					
		
(defparameter *default-empty-test-shape* '(4 4))
(defparameter *default-zeros-test-shape* '(4 4))	
(defparameter *default-ones-test-shape* '(4 4))	
(defparameter *default-full-test-shape* '(4 4))	
		
;; -- `Empty` tests section --		
		
(fiveam:test nnl2.hli.ts/empty/float64
  (check-nnl2.hli.ts/empty :dtype :float64 :int-dtype nnl2.hli.ts.tests:+enum-float64+ :shape *default-empty-test-shape*))
	  
(fiveam:test nnl2.hli.ts/empty/float32	
  (check-nnl2.hli.ts/empty :dtype :float32 :int-dtype nnl2.hli.ts.tests:+enum-float32+ :shape *default-empty-test-shape*))
  
(fiveam:test nnl2.hli.ts/empty/int32
  (check-nnl2.hli.ts/empty :dtype :int32 :int-dtype nnl2.hli.ts.tests:+enum-int32+ :shape *default-empty-test-shape*)) 
  
(fiveam:test nnl2.hli.ts/empty/int64
  (check-nnl2.hli.ts/empty :dtype :int64 :int-dtype nnl2.hli.ts.tests:+enum-int64+ :shape *default-empty-test-shape*))   
  
;; -- `Zeros` tests section --

(fiveam:test nnl2.hli.ts/zeros/float64
  (check-nnl2.hli.ts/check-value :dtype :float64 :int-dtype nnl2.hli.ts.tests:+enum-float64+ :shape *default-zeros-test-shape* :expected-value 0.0d0 :tensor-func :zeros))
	  
(fiveam:test nnl2.hli.ts/zeros/float32
  (check-nnl2.hli.ts/check-value :dtype :float32 :int-dtype nnl2.hli.ts.tests:+enum-float32+ :shape *default-zeros-test-shape* :expected-value 0.0s0 :tensor-func :zeros))
  
(fiveam:test nnl2.hli.ts/zeros/int32
  (check-nnl2.hli.ts/check-value :dtype :int32 :int-dtype nnl2.hli.ts.tests:+enum-int32+ :shape *default-zeros-test-shape* :expected-value 0 :tensor-func :zeros))   
  
(fiveam:test nnl2.hli.ts/zeros/int64
  (check-nnl2.hli.ts/check-value :dtype :int64 :int-dtype nnl2.hli.ts.tests:+enum-int64+ :shape *default-zeros-test-shape* :expected-value 0 :tensor-func :zeros))     

;; -- `Ones` tests section --

(fiveam:test nnl2.hli.ts/ones/float64
  (check-nnl2.hli.ts/check-value :dtype :float64 :int-dtype nnl2.hli.ts.tests:+enum-float64+ :shape *default-ones-test-shape* :expected-value 1.0d0 :tensor-func :ones))
	  
(fiveam:test nnl2.hli.ts/ones/float32
  (check-nnl2.hli.ts/check-value :dtype :float32 :int-dtype nnl2.hli.ts.tests:+enum-float32+ :shape *default-ones-test-shape* :expected-value 1.0s0 :tensor-func :ones))
  
(fiveam:test nnl2.hli.ts/ones/int32
  (check-nnl2.hli.ts/check-value :dtype :int32 :int-dtype nnl2.hli.ts.tests:+enum-int32+ :shape *default-ones-test-shape* :expected-value 1 :tensor-func :ones))   
  
(fiveam:test nnl2.hli.ts/ones/int64
  (check-nnl2.hli.ts/check-value :dtype :int64 :int-dtype nnl2.hli.ts.tests:+enum-int64+ :shape *default-ones-test-shape* :expected-value 1 :tensor-func :ones))     

;; -- `Full` tests section --

(fiveam:test nnl2.hli.ts/full/float64
  (check-nnl2.hli.ts/full :dtype :float64 :int-dtype nnl2.hli.ts.tests:+enum-float64+ :shape *default-full-test-shape* :filler 2.0d0))
	  
(fiveam:test nnl2.hli.ts/full/float32
  (check-nnl2.hli.ts/full :dtype :float32 :int-dtype nnl2.hli.ts.tests:+enum-float32+ :shape *default-full-test-shape* :filler 2.0s0))
  
(fiveam:test nnl2.hli.ts/full/int32
  (check-nnl2.hli.ts/full :dtype :int32 :int-dtype nnl2.hli.ts.tests:+enum-int32+ :shape *default-full-test-shape* :filler 2)) 

(fiveam:test nnl2.hli.ts/full/int64
  (check-nnl2.hli.ts/full :dtype :int64 :int-dtype nnl2.hli.ts.tests:+enum-int64+ :shape *default-full-test-shape* :filler 2))   
  