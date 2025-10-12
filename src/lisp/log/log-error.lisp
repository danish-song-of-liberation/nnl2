(in-package :nnl2.log)

;; NNL2

;; Filepath: nnl2/src/lisp/log/log-error.lisp
;; File: log-error.lisp

;; Contains a definition of #'nnl2.log:error function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun error (msg &rest args)
  "Displays an error in the console via the nnl2 logging system
   
   Args:
       msg: Message to format
	   args (&rest): Format args

   Example:
	   (nnl2.log:error \"Failed to create tensor\")
	   (nnl2.log:error \"~a Not equal to ~a\" 'foo 'bar)
	   
   Note:
	   Doesn't terminate the program. 
	   If an error occurs, the program continues to run"
	   
  (let ((text (apply #'format nil msg args)))
    (%error text)
	text))
  