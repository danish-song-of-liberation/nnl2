(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-broadcasting.lisp
;; File: ts-tests-broadcasting.lisp

;; Makes tests with /broadcasting operations (like .+/broadcasting)

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/broadcasting-operation (&key dtype shape-1 shape-2 fill-1 fill-2 op expected-shape expected-val inplace)
  "Implements a check operation for broadcasting"
  
  (nnl2.tests.utils:start-log-for-test :function op :dtype dtype)
  
  (handler-case
      (let ((tensor-shape-1 (coerce shape-1 'vector))
			(tensor-shape-2 (coerce shape-2 'vector))
			(lisp-type (nnl2.hli.ts:type/nnl2->lisp dtype)))
			
		(nnl2.hli.ts:tlet* ((first-tensor (nnl2.hli.ts:full tensor-shape-1 :dtype dtype :filler (coerce fill-1 lisp-type)))
							(second-tensor (nnl2.hli.ts:full tensor-shape-2 :dtype dtype :filler (coerce fill-2 lisp-type)))
							(result-tensor (funcall op first-tensor second-tensor)))
	
		  (nnl2.hli.ts.tests:check-tensor-data 
		    :tensor (if inplace first-tensor result-tensor) 
		    :shape shape-1
		    :expected-value expected-val)

		  (nnl2.tests.utils:end-log-for-test :function op :dtype dtype)))
	  
	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function op :dtype dtype)
	  
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when you call the function"
		:error-type :error
		:message e 
		:function op
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%dtype - ~a~%shape-1 - ~d~%shape-2 - ~a~%inplace - ~a~%fill-1 - ~a~%fill-2 - ~a~%op - ~a~%expected-val - ~a"
					dtype shape-1 shape-2 inplace fill-1 fill-2 op expected-val)))))
  
(defparameter *default-.+/broadcasting/a-operation-shape* '(4 4 4) "Shape for first (a) tensor in broadcasting")
(defparameter *default-.+/broadcasting/b-operation-shape* '(4) "Shape for second (b) tensor in broadcasting")	  
  
(defparameter *default-.-/broadcasting/a-operation-shape* '(4 4) "Shape for first (a) tensor in broadcasting")
(defparameter *default-.-/broadcasting/b-operation-shape* '(4) "Shape for second (b) tensor in broadcasting")	  

(defparameter *default-.*/broadcasting/a-operation-shape* '(3 3 2) "Shape for first (a) tensor in broadcasting")
(defparameter *default-.*/broadcasting/b-operation-shape* '(3 1) "Shape for second (b) tensor in broadcasting")	  

(defparameter *default-.//broadcasting/a-operation-shape* '(3 3) "Shape for first (a) tensor in broadcasting")
(defparameter *default-.//broadcasting/b-operation-shape* '(3) "Shape for second (b) tensor in broadcasting")	  

(defparameter *default-.^/broadcasting/a-operation-shape* '(3 3) "Shape for first (a) tensor in broadcasting")
(defparameter *default-.^/broadcasting/b-operation-shape* '(3) "Shape for second (b) tensor in broadcasting")	  

(defparameter *default-.max/broadcasting/a-operation-shape* '(3 4) "Shape for first (a) tensor in broadcasting")
(defparameter *default-.max/broadcasting/b-operation-shape* '(3) "Shape for second (b) tensor in broadcasting")	  

(defparameter *default-.min/broadcasting/a-operation-shape* '(3 4) "Shape for first (a) tensor in broadcasting")
(defparameter *default-.min/broadcasting/b-operation-shape* '(4) "Shape for second (b) tensor in broadcasting")	  

(defparameter *default-axpy/broadcasting/a-operation-shape* '(3 3) "Shape for first (a) tensor in broadcasting")
(defparameter *default-axpy/broadcasting/b-operation-shape* '(1) "Shape for second (b) tensor in broadcasting")	  

;; -- `.+/broadcasting` tests section --

(fiveam:test nnl2.hli.ts/.+/broadcasting/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.+/broadcasting/a-operation-shape* :shape-2 *default-.+/broadcasting/b-operation-shape* :fill-1 2.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:.+ :expected-val 5.0d0))									
  
(fiveam:test nnl2.hli.ts/.+/broadcasting/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.+/broadcasting/a-operation-shape* :shape-2 *default-.+/broadcasting/b-operation-shape* :fill-1 2.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:.+ :expected-val 5.0s0))									
    
(fiveam:test nnl2.hli.ts/.+/broadcasting/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.+/broadcasting/a-operation-shape* :shape-2 *default-.+/broadcasting/b-operation-shape* :fill-1 2 :fill-2 3 :op #'nnl2.hli.ts:.+ :expected-val 5))									
    	
;; -- `.-/broadcasting` tests section --

(fiveam:test nnl2.hli.ts/.-/broadcasting/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.-/broadcasting/a-operation-shape* :shape-2 *default-.-/broadcasting/b-operation-shape* :fill-1 2.0d0 :fill-2 1.0d0 :op #'nnl2.hli.ts:.- :expected-val 1.0d0))									
  
(fiveam:test nnl2.hli.ts/.-/broadcasting/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.-/broadcasting/a-operation-shape* :shape-2 *default-.-/broadcasting/b-operation-shape* :fill-1 2.0s0 :fill-2 1.0s0 :op #'nnl2.hli.ts:.- :expected-val 1.0s0))									
    
(fiveam:test nnl2.hli.ts/.-/broadcasting/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.-/broadcasting/a-operation-shape* :shape-2 *default-.-/broadcasting/b-operation-shape* :fill-1 2 :fill-2 1 :op #'nnl2.hli.ts:.- :expected-val 1))									
    			
;; -- `.*/broadcasting` tests section --

(fiveam:test nnl2.hli.ts/.*/broadcasting/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.*/broadcasting/a-operation-shape* :shape-2 *default-.*/broadcasting/b-operation-shape* :fill-1 2.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:.* :expected-val 6.0d0))									
  
(fiveam:test nnl2.hli.ts/.*/broadcasting/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.*/broadcasting/a-operation-shape* :shape-2 *default-.*/broadcasting/b-operation-shape* :fill-1 2.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:.* :expected-val 6.0s0))									
    
(fiveam:test nnl2.hli.ts/.*/broadcasting/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.*/broadcasting/a-operation-shape* :shape-2 *default-.*/broadcasting/b-operation-shape* :fill-1 2 :fill-2 3 :op #'nnl2.hli.ts:.* :expected-val 6))									
    			
;; -- `.//broadcasting` tests section --

(fiveam:test nnl2.hli.ts/.//broadcasting/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.//broadcasting/a-operation-shape* :shape-2 *default-.//broadcasting/b-operation-shape* :fill-1 8.0d0 :fill-2 2.0d0 :op #'nnl2.hli.ts:./ :expected-val 4.0d0))									
  
(fiveam:test nnl2.hli.ts/.//broadcasting/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.//broadcasting/a-operation-shape* :shape-2 *default-.//broadcasting/b-operation-shape* :fill-1 8.0s0 :fill-2 2.0s0 :op #'nnl2.hli.ts:./ :expected-val 4.0s0))									
    
(fiveam:test nnl2.hli.ts/.//broadcasting/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.//broadcasting/a-operation-shape* :shape-2 *default-.//broadcasting/b-operation-shape* :fill-1 8 :fill-2 2 :op #'nnl2.hli.ts:./ :expected-val 4))											
    			
;; -- `.^/broadcasting` tests section --

(fiveam:test nnl2.hli.ts/.^/broadcasting/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.^/broadcasting/a-operation-shape* :shape-2 *default-.^/broadcasting/b-operation-shape* :fill-1 4.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:.^ :expected-val 64.0d0))									
  
(fiveam:test nnl2.hli.ts/.^/broadcasting/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.^/broadcasting/a-operation-shape* :shape-2 *default-.^/broadcasting/b-operation-shape* :fill-1 4.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:.^ :expected-val 64.0s0))									
    
(fiveam:test nnl2.hli.ts/.^/broadcasting/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.^/broadcasting/a-operation-shape* :shape-2 *default-.^/broadcasting/b-operation-shape* :fill-1 4 :fill-2 3 :op #'nnl2.hli.ts:.^ :expected-val 64))									
    							
;; -- `.max/broadcasting` tests section --	

(fiveam:test nnl2.hli.ts/.max/broadcasting/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.max/broadcasting/a-operation-shape* :shape-2 *default-.max/broadcasting/b-operation-shape* :fill-1 4.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:.max :expected-val 4.0d0))									
  
(fiveam:test nnl2.hli.ts/.max/broadcasting/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.max/broadcasting/a-operation-shape* :shape-2 *default-.max/broadcasting/b-operation-shape* :fill-1 4.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:.max :expected-val 4.0s0))									
    
(fiveam:test nnl2.hli.ts/.max/broadcasting/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.max/broadcasting/a-operation-shape* :shape-2 *default-.max/broadcasting/b-operation-shape* :fill-1 4 :fill-2 3 :op #'nnl2.hli.ts:.max :expected-val 4))									
    							
;; -- `.min/broadcasting` tests section --							
	
(fiveam:test nnl2.hli.ts/.min/broadcasting/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.min/broadcasting/a-operation-shape* :shape-2 *default-.min/broadcasting/b-operation-shape* :fill-1 4.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:.min :expected-val 3.0d0))									
  
(fiveam:test nnl2.hli.ts/.min/broadcasting/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.min/broadcasting/a-operation-shape* :shape-2 *default-.min/broadcasting/b-operation-shape* :fill-1 4.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:.min :expected-val 3.0s0))									
    
(fiveam:test nnl2.hli.ts/.min/broadcasting/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.min/broadcasting/a-operation-shape* :shape-2 *default-.min/broadcasting/b-operation-shape* :fill-1 4 :fill-2 3 :op #'nnl2.hli.ts:.min :expected-val 3))									
    		
;; -- `axpy/broadcasting` tests section --							
	
(fiveam:test nnl2.hli.ts/axpy/broadcasting/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-axpy/broadcasting/a-operation-shape* :shape-2 *default-axpy/broadcasting/b-operation-shape* :fill-1 4.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:axpy :expected-val 7.0d0))									
  
(fiveam:test nnl2.hli.ts/axpy/broadcasting/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-axpy/broadcasting/a-operation-shape* :shape-2 *default-axpy/broadcasting/b-operation-shape* :fill-1 4.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:axpy :expected-val 7.0s0))									
    
(fiveam:test nnl2.hli.ts/axpy/broadcasting/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-axpy/broadcasting/a-operation-shape* :shape-2 *default-axpy/broadcasting/b-operation-shape* :fill-1 4 :fill-2 3 :op #'nnl2.hli.ts:axpy :expected-val 7))									
			
;; The file is continued by `ts-tests-broadcasting-inplace.lisp` 
;; in the same folder, which implements the same operations but in-place  				
				