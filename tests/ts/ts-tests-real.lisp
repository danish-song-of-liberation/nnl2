(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-real.lisp
;; File: ts-tests-real.lisp

;; File contains tests with real tensor operation that will be used 
;; by the user, rather than repetitive tests of a single function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defmacro make-test (dtype &body body &aux (msg "No Function. Just simple test"))
  "Advanced wrapper for creating a test"

  `(progn 
     (nnl2.tests.utils:start-log-for-test :function ,msg :dtype ,dtype)
	 
	 (handler-case
	     (progn 
		   ,@body
		   (nnl2.tests.utils:end-log-for-test :function ,msg :dtype ,dtype))
		   
	   (error (e)   
		 (nnl2.tests.utils:fail-log-for-test :function ,msg :dtype ,dtype)
		 
		 (nnl2.tests.utils:throw-error 
	       :documentation "Throws an informative error if an error occurs when you call the function"
		   :error-type :error
		   :message e 
		   :function ,msg)))))
		   


;; TODO

