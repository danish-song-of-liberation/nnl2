(in-package :nnl2.tests.utils)

;; NNL2

;; Filepath: nnl2/tests/utils/utils-trivial-errors.lisp
;; File: utils-trivial-errors.lisp

;; Declarations of auxiliary functions for further tests

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun throw-error (&key (error-type :error) (function "Unknown Function") (message "No Data") (addition "No Data") documentation)
  (let ((info (format nil "~%[nnl2]: { Error from function `~a`: `~a`~%Additional Info: ~a }~%" function message addition)))
    (case error-type
	  (:error (error info))
	  (:warning (warn info)))))
