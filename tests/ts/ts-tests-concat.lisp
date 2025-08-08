(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-concat.lisp
;; File: ts-tests-concat.lisp

;; contains tests for operations such as incf, decf, etc. (they are 
;; built into basic operations such as .+, .-, and are applied 
;; automatically, for example when (.+ foo 1))

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun stack-test (&key shape-1 shape-2 expected-shape dtype op)
  "Tests for hstack/vstack by comparing dimensions"
  
  (nnl2.tests.utils:start-log-for-test :function op :dtype dtype)
  
  (handler-case
      (let ((tensor-shape-1 (coerce shape-1 'vector))
			(tensor-shape-2 (coerce shape-2 'vector))
			(lisp-type (nnl2.hli.ts:ts-type-to-lisp dtype)))
			
		(nnl2.hli.ts:tlet* ((a (nnl2.hli.ts:ones tensor-shape-1 :dtype dtype))
							(b (nnl2.hli.ts:ones tensor-shape-2 :dtype dtype))
							(result (funcall op a b)))
							
		  (let ((result-shape (nnl2.hli.ts:shape result :as :list)))
		    (fiveam:is (equal result-shape expected-shape))
			
			(nnl2.hli.ts.tests:check-tensor-data 
		      :tensor result
		      :shape result-shape
		      :expected-value (coerce 1 lisp-type))
			  
			(nnl2.tests.utils:end-log-for-test :function op :dtype dtype))))
	  
	(error (e)
	  (nnl2.tests.utils:fail-log-for-test :function op :dtype dtype)
	  
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when you call the function"
		:error-type :error
		:message e 
		:function op
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%shape-1 - ~a~%shape-2 - ~a~%expected-shape - ~a~%dtype - ~a~%op - ~a"
					shape-1 shape-2 expected-shape dtype op)))))  
					
(defun concat-test (&key axis dtype shape-1 shape-2 expected-shape &aux (op #'nnl2.hli.ts:concat))
  "Test for concat (just simple shape check)"

  (nnl2.tests.utils:start-log-for-test :function op :dtype dtype) 

  (handler-case
      (let ((tensor-shape-1 (coerce shape-1 'vector))
		    (tensor-shape-2 (coerce shape-2 'vector))
		    (lisp-type (nnl2.hli.ts:ts-type-to-lisp dtype)))
		  
	    (nnl2.hli.ts:tlet* ((a (nnl2.hli.ts:ones tensor-shape-1 :dtype dtype))
							(b (nnl2.hli.ts:ones tensor-shape-2 :dtype dtype))
							(result (nnl2.hli.ts:concat axis a b)))
							
		  (let ((result-shape (nnl2.hli.ts:shape result :as :list)))
		    (fiveam:is (equal result-shape expected-shape))
			
			;; (nnl2.hli.ts.tests:check-tensor-data ;; <<== POTENTIAL ERROR HERE. MAY RESULT IN A SEGFAULT
		    ;;   :tensor result
		    ;;   :shape result-shape
		    ;;   :expected-value (coerce 1 lisp-type)
			;;   :debug t)
			
			(nnl2.tests.utils:end-log-for-test :function op :dtype dtype))))

    (error (e)
      (nnl2.tests.utils:fail-log-for-test :function op :dtype dtype)
	  
	  (nnl2.tests.utils:throw-error 
	    :documentation "Throws an informative error if an error occurs when you call the function"
		:error-type :error
		:message e 
		:function op
		:addition (format nil "The function caught an error when running the test with the passed arguments:~%shape-1 - ~a~%shape-2 - ~a~%expected-shape - ~a~%dtype - ~a"
					shape-1 shape-2 expected-shape dtype)))))

;; -- `hstack` tests section --	
  
(fiveam:test nnl2.hli.ts/hstack/float64
  (stack-test :dtype :float64 :shape-1 '(3 3) :shape-2 '(3 4) :expected-shape '(3 7) :op #'nnl2.hli.ts:hstack))		  
  
(fiveam:test nnl2.hli.ts/hstack/float32
  (stack-test :dtype :float32 :shape-1 '(2 5) :shape-2 '(2 4) :expected-shape '(2 9) :op #'nnl2.hli.ts:hstack))		  

(fiveam:test nnl2.hli.ts/hstack/int32
  (stack-test :dtype :int32 :shape-1 '(1 4) :shape-2 '(1 1) :expected-shape '(1 5) :op #'nnl2.hli.ts:hstack))		 

;; -- `vstack` tests section --

(fiveam:test nnl2.hli.ts/vstack/float64
  (stack-test :dtype :float64 :shape-1 '(5 4) :shape-2 '(3 4) :expected-shape '(8 4) :op #'nnl2.hli.ts:vstack))		   
  
(fiveam:test nnl2.hli.ts/vstack/float32
  (stack-test :dtype :float32 :shape-1 '(2 4) :shape-2 '(3 4) :expected-shape '(5 4) :op #'nnl2.hli.ts:vstack))

(fiveam:test nnl2.hli.ts/vstack/int32
  (stack-test :dtype :int32 :shape-1 '(1 3) :shape-2 '(2 3) :expected-shape '(3 3) :op #'nnl2.hli.ts:vstack))  
    
;; -- `concat` tests section --

(fiveam:test nnl2.hli.ts/concat/float64/1
  (concat-test :dtype :float64 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(5 5 10) :axis 2))		

(fiveam:test nnl2.hli.ts/concat/float64/2
  (concat-test :dtype :float64 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(5 10 5) :axis 1))	
  
(fiveam:test nnl2.hli.ts/concat/float64/3
  (concat-test :dtype :float64 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(10 5 5) :axis 0))	  
  
(fiveam:test nnl2.hli.ts/concat/float32/1
  (concat-test :dtype :float32 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(5 5 10) :axis 2))		

(fiveam:test nnl2.hli.ts/concat/float32/2
  (concat-test :dtype :float32 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(5 10 5) :axis 1))	
  
(fiveam:test nnl2.hli.ts/concat/float32/3
  (concat-test :dtype :float32 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(10 5 5) :axis 0))	

(fiveam:test nnl2.hli.ts/concat/int32/1
  (concat-test :dtype :int32 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(5 5 10) :axis 2))		

(fiveam:test nnl2.hli.ts/concat/int32/2
  (concat-test :dtype :int32 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(5 10 5) :axis 1))	
  
(fiveam:test nnl2.hli.ts/concat/int32/3
  (concat-test :dtype :int32 :shape-1 '(5 5 5) :shape-2 '(5 5 5) :expected-shape '(10 5 5) :axis 0))	    
  