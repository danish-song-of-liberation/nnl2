(in-package :nnl2.system)

#| PLEASE READ ME

   Some parts of the code contain very strange solutions, 
   but please don't blame me. It's the cl-json library 
   that handles t, nil, and kebab-case in a very strange way,
   forcing me to resort to such measures.   |#

(defparameter *alist-true* 'true) 
(defparameter *alist-false* 'false)

(defun alist-symbol-to-bool (str)
  "returns T if string is 'true
   returns NIL if string is 'false
   otherwise returns error"

  (cond
    ((eq str *alist-true*) t)
	((eq str *alist-false*) nil)
	(t (error "(~a): Unknown value: ~a~%" #'alist-symbol-to-bool str))))
	
(defun bool-to-alist (bool)
  #| if T returns "true" otherwise "false" |#

  (if bool 
    *alist-true* 
	*alist-false*))
	
(defun bool-to-int (bool)
  "if T returns 1 otherwise 0"
  
  (if bool
	1
	0))

(defmacro assoc-key (list alist)
  `(cdr (assoc ,list ,alist)))	 
