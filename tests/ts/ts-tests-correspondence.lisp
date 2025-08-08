(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-correspondence.lisp
;; File: ts-tests-correspondence.lisp

;; Contains tests for operations such as incf, decf, etc. (they are 	
;; built into basic operations such as .+, .-, and are applied 
;; automatically, for example when (.+ foo 1))

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/correspondence-operation (&key dtype shape fill val op expected inplace)
  "Performs operations on tensors where one argument 
   is a scalar and the other is a tensor"
   
  (nnl2.tests.utils:start-log-for-test :function op :dtype dtype)

  (handler-case
      (let ((tensor-shape (coerce shape 'vector))
			(lisp-type (nnl2.hli.ts:ts-type-to-lisp dtype)))
			
	    (nnl2.hli.ts:tlet* ((first-tensor (nnl2.hli.ts:full shape :dtype dtype :filler (coerce fill lisp-type)))
							(result-tensor (funcall op first-tensor val)))
							
		  (nnl2.hli.ts.tests:check-tensor-data 
		    :tensor (if inplace first-tensor result-tensor)
		    :shape shape
		    :expected-value expected)
			
		  (nnl2.tests.utils:end-log-for-test :function op :dtype dtype)))	
			

    (error (e)
      (nnl2.tests.utils:fail-log-for-test :function op :dtype dtype)

	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when you call the function"
		:error-type :error
		:message e 
		:function op
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%shape - ~d~%val - ~a~%inplace - ~a"
					dtype shape val op inplace)))))
					
(defparameter *default-.+/incf-operation-shape* '(5 5))	
(defparameter *default-.-/decf-operation-shape* '(5 5))	
(defparameter *default-.*/mulf-operation-shape* '(5 5))	
(defparameter *default-.//divf-operation-shape* '(5 5))	
(defparameter *default-.^/powf-operation-shape* '(5 5))	
(defparameter *default-.max/maxf-operation-shape* '(5 5))
(defparameter *default-.min/minf-operation-shape* '(5 5))		

;; -- `.+/incf` tests section --

(fiveam:test nnl2.hli.ts/.+/incf/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.+/incf-operation-shape* :fill 1.0d0 :val 1.0d0 :expected 2.0d0 :op #'nnl2.hli.ts:.+))									
  
(fiveam:test nnl2.hli.ts/.+/incf/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.+/incf-operation-shape* :fill 1.0s0 :val 1.0s0 :expected 2.0s0 :op #'nnl2.hli.ts:.+))									
    
(fiveam:test nnl2.hli.ts/.+/incf/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.+/incf-operation-shape* :fill 1 :val 1 :expected 2 :op #'nnl2.hli.ts:.+))									
      
;; -- `.-/decf` tests section --	  

(fiveam:test nnl2.hli.ts/.-/decf/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.-/decf-operation-shape* :fill 2.0d0 :val 1.0d0 :expected 1.0d0 :op #'nnl2.hli.ts:.-))									
  
(fiveam:test nnl2.hli.ts/.-/decf/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.-/decf-operation-shape* :fill 2.0s0 :val 1.0s0 :expected 1.0s0 :op #'nnl2.hli.ts:.-))									
    
(fiveam:test nnl2.hli.ts/.-/decf/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.-/decf-operation-shape* :fill 2 :val 1 :expected 1 :op #'nnl2.hli.ts:.-))									
      	  
;; -- `.*/mulf` tests section --			  
		  
(fiveam:test nnl2.hli.ts/.*/mulf/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.*/mulf-operation-shape* :fill 4.0d0 :val 2.0d0 :expected 8.0d0 :op #'nnl2.hli.ts:.*))									
  
(fiveam:test nnl2.hli.ts/.*/mulf/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.*/mulf-operation-shape* :fill 4.0s0 :val 2.0s0 :expected 8.0s0 :op #'nnl2.hli.ts:.*))									
    
(fiveam:test nnl2.hli.ts/.*/mulf/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.*/mulf-operation-shape* :fill 4 :val 2 :expected 8 :op #'nnl2.hli.ts:.*))									  
		  
;; -- `.//divf` tests section --			  
		  
(fiveam:test nnl2.hli.ts/.//divf/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.//divf-operation-shape* :fill 4.0d0 :val 2.0d0 :expected 2.0d0 :op #'nnl2.hli.ts:./))									
  
(fiveam:test nnl2.hli.ts/.//divf/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.//divf-operation-shape* :fill 4.0s0 :val 2.0s0 :expected 2.0s0 :op #'nnl2.hli.ts:./))									
    
(fiveam:test nnl2.hli.ts/.//divf/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.//divf-operation-shape* :fill 4 :val 2 :expected 2 :op #'nnl2.hli.ts:./))									  		  
		  
;; -- `.^/powf` tests section --			  
		  
(fiveam:test nnl2.hli.ts/.^/powf/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.^/powf-operation-shape* :fill 4.0d0 :val 3.0d0 :expected 64.0d0 :op #'nnl2.hli.ts:.^))									
  
(fiveam:test nnl2.hli.ts/.^/powf/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.^/powf-operation-shape* :fill 4.0s0 :val 3.0s0 :expected 64.0s0 :op #'nnl2.hli.ts:.^))									
    
(fiveam:test nnl2.hli.ts/.^/powf/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.^/powf-operation-shape* :fill 4 :val 3 :expected 64 :op #'nnl2.hli.ts:.^))									  		  

;; -- `.max/maxf` tests section --		
	
(fiveam:test nnl2.hli.ts/.max/maxf/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.max/maxf-operation-shape* :fill 4.0d0 :val 5.0d0 :expected 5.0d0 :op #'nnl2.hli.ts:.max))									
  
(fiveam:test nnl2.hli.ts/.max/maxf/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.max/maxf-operation-shape* :fill 4.0s0 :val 5.0s0 :expected 5.0s0 :op #'nnl2.hli.ts:.max))									
    
(fiveam:test nnl2.hli.ts/.max/maxf/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.max/maxf-operation-shape* :fill 4 :val 5 :expected 5 :op #'nnl2.hli.ts:.max))									  		  
	
;; -- `.min/minf` tests section --		
	
(fiveam:test nnl2.hli.ts/.min/minf/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.min/minf-operation-shape* :fill 4.0d0 :val 5.0d0 :expected 4.0d0 :op #'nnl2.hli.ts:.min))									
  
(fiveam:test nnl2.hli.ts/.min/minf/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.min/minf-operation-shape* :fill 4.0s0 :val 5.0s0 :expected 4.0s0 :op #'nnl2.hli.ts:.min))									
    
(fiveam:test nnl2.hli.ts/.min/minff/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.min/minf-operation-shape* :fill 4 :val 5 :expected 4 :op #'nnl2.hli.ts:.min))									  		 		
	
;; A continuation of this file is `ts-tests-correspondence-inplace` 
;; with the implementation of the same thing but in-place				  
				  