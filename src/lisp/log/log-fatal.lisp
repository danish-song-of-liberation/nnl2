(in-package :nnl2.log)

;; NNL2

;; Filepath: nnl2/src/lisp/log/log-fatal.lisp
;; File: log-fatal.lisp

;; Contains a definition of #'nnl2.log:fatal function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun fatal (msg &rest args)
  "Displays a fatal error message and immediately terminates the program via the nnl2 logging system
   
   Args:
       msg: Message to format
       args (&rest): Format args

   Example:
       (nnl2.log:fatal \"Critical system failure\")
       (nnl2.log:fatal \"Cannot initialize ~a subsystem\" 'database)
   
   Warning:
       The program will exit immediately after displaying the message.
       This function does not return - execution stops here"
  
  (let ((text (apply #'format nil msg args)))
    (%fatal text)
    text)) ;; KeBugCheckEx(...); return SUCCESS;
	