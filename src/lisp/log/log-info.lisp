(in-package :nnl2.log)

;; NNL2

;; Filepath: nnl2/src/lisp/log/log-info.lisp
;; File: log-info.lisp

;; Contains a definition of #'nnl2.log:info function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun info (msg &rest args)
  "Displays an info message in the console via the nnl2 logging system
   
   Args:
       msg: Message to format
       args (&rest): Format args

   Example:
       (nnl2.log:info \"Application started successfully\")
       (nnl2.log:info \"Loaded ~a records from database\" 1500)"
  
  (let ((text (apply #'format nil msg args)))
    (%info text)
    text))
	