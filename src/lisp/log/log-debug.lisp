(in-package :nnl2.log)

;; NNL2

;; Filepath: nnl2/src/lisp/log/log-debug.lisp
;; File: log-debug.lisp

;; Contains a definition of #'nnl2.log:debug function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun debug (msg &rest args)
  "Displays a debug message in the console via the nnl2 logging system
   
   Args:
       msg: Message to format
       args (&rest): Format args

   Example:
       (nnl2.log:debug \"Entering function process-data\")
       (nnl2.log:debug \"Variable ~a has value: ~a\" 'counter 42)"
  
  (let ((text (apply #'format nil msg args)))
    (%debug text)
    text))
	