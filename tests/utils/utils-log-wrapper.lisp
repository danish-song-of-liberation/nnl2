(in-package :nnl2.tests.utils)

;; NNL2

;; Filepath: nnl2/tests/utils/utils-log-wrapper.lisp
;; File: utils-log-wrapper.lisp

;; File with a wrapper function for logging

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defmacro make-test (dtype test-for &body body &aux (msg (format nil "No Function. Just a simple test. Addition: Test for ~a" test-for)))
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
		   