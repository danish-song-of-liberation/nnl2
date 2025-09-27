(in-package :nnl2.system)

;; NNL2

;; Filepath: nnl2/src/lisp/system/system-utils.lisp
;; File: system-utils.lisp

;; File contains system-level helper functions 
;; for working with the configuration file

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defparameter *alist-true* 'true) 
(defparameter *alist-false* 'false)

(defun alist-symbol-to-bool (str)
  "Returns T if string is 'true
   Returns NIL if string is 'false
   Otherwise returns error"

  (cond
    ((eq str *alist-true*) t)
	((eq str *alist-false*) nil)
	(t (error "(~a): Unknown value: ~a~%" #'alist-symbol-to-bool str))))
	
(defun bool-to-alist (bool)
  "If T returns \"true\" otherwise \"false\""

  (if bool 
    *alist-true* 
	*alist-false*))
	
(defun bool-to-int (bool)
  "If T returns 1 otherwise 0"
  
  (if bool
	1
	0))

(defmacro assoc-key (list alist)
  "Returns a value by key from an associative list"
  
  `(cdr (assoc ,list ,alist)))	 
  
(defmacro alist-to-int (alist)
  "Converts symbol to integer ('true -> 1)"
  
  `(bool-to-int (alist-symbol-to-bool ,alist)))
