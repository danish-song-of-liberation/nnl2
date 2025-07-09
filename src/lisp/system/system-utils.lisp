(in-package :nnl2.system)

#| PLEASE READ ME

   Some parts of the code contain very strange solutions, 
   but please don't blame me. It's the cl-json library 
   that handles t, nil, and kebab-case in a very strange way,
   forcing me to resort to such measures.   |#

(defparameter *json-true* "true") 
(defparameter *json-false* "false")

(defun json-string-to-bool (str)
  "returns T if string is \"true\"
   returns NIL if string is \"false\"
   otherwise returns error"

  (cond
    ((string= str *json-true*) t)
	((string= str *json-false*) nil)
	(t (error "(~a): Unknown value: ~a~%" #'json-string-to-bool str))))

(defun bool-to-json-string (bool)
  #| if T returns "true" otherwise "false" |#

  (if bool 
    *json-true* 
	*json-false*))
	
(defun bool-to-int (bool)
  "if T returns 1 otherwise 0"
  
  (if bool
	1
	0))
