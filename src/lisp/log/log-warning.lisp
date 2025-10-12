(in-package :nnl2.log)

;; NNL2

;; Filepath: nnl2/src/lisp/log/log-warning.lisp
;; File: log-warning.lisp

;; Contains a definition of #'nnl2.log:warning function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun warning (msg &rest args)
  "Displays a warning in the console via the nnl2 logging system
   
   Args:
       msg: Message to format
       args (&rest): Format args

   Example:
       (nnl2.log:warning \"Deprecated function called\")
       (nnl2.log:warning \"~a is deprecated, use ~a instead\" 'old-func 'new-func)
   
   Note:
       Warnings indicate potential issues that don't prevent execution.
       The program continues to run normally after displaying the warning"
  
  (let ((text (apply #'format nil msg args)))
    (%warning text)
    text))
	
