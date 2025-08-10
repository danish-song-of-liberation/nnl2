(in-package :nnl2.tests.utils)

;; NNL2

;; Filepath: nnl2/tests/utils/utils-trivial-errors.lisp
;; File: utils-trivial-errors.lisp

;; Declarations of auxiliary handler functions for further tests

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun throw-error (&key (error-type :error) (function "Unknown Function") (message "No Data") (addition "No Data") documentation)
  (let ((info (format nil "~%~%[nnl2]: { ~a from function `~a`: `~a`~%Additional Info: ~a }~%~%" (case error-type (:error "Error") (:warning "Warning") (otherwise "Error")) function message addition)))
    (with-open-file (stream nnl2.system:+nnl2-filepath-log-path+
					  :direction :output
					  :if-exists :append 
					  :if-does-not-exist :create)
					  
	  (print info stream))))
	  
(defun log-test-event (message &key (function "Unknown Function") (dtype "Unknown type") additional-newline)
  (with-open-file (stream nnl2.system:+nnl2-filepath-log-path+
					  :direction :output
					  :if-exists :append
					  :if-does-not-exist :create)
    (format stream "[nnl2]: ~a test for `~a` (type: ~a)~%" message function dtype)
	
	(when additional-newline
	  (format stream "~%"))))

(defun start-log-for-test (&key (function "Unknown Function") (dtype "Unknown type"))
  (log-test-event "Started" :function function :dtype dtype))

(defun end-log-for-test (&key (function "Unknown Function") (dtype "Unknown type"))
  (log-test-event "Succesfully ended" :function function :dtype dtype :additional-newline t))

(defun fail-log-for-test (&key (function "Unknown Function") (dtype "Unknown type"))
  (log-test-event "Failed" :function function :dtype dtype))
  