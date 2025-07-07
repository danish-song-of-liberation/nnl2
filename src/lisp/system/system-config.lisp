(in-package :nnl2.system)

#| PLEASE READ ME

   Some parts of the code contain very strange solutions, 
   but please don't blame me. It's the cl-json library 
   that handles t, nil, and kebab-case in a very strange way,
   forcing me to resort to such measures.   |#

(defparameter *json-true* "true") 
(defparameter *json-false* "false")

(defparameter *first-launch* nil
  "flag indicating if this is the first system launch")

(defparameter *implementations* (list 
								  (cons 'openblas0330woa64static *json-false*))
								 
  "Contains an associative list for subsequent conversion 
   to json format in the format (implementation . *is-it-working-or-not*)")

(defun json-string-to-bool (str)
  (cond
    ((string= str *json-true*) t)
	((string= str *json-false*) nil)
	(t (error "(~a): Unknown value: ~a~%" #'json-string-to-bool str))))

(defun bool-to-json-string (bool)
  #| if T returns "true" otherwise "false" |#

  (if bool 
    *json-true* 
	*json-false*))
	
(defun get-config-path ()
  (uiop:merge-pathnames*
    (make-pathname :directory '(:relative "src" "lisp" "system")
				   :name "config"
				   :type "json")
				   
	nnl2.intern-system:*current-dir*))
	
(defun create-default-config (path)
  "creates json config file"
  
  (assert (not (probe-file path)) nil (format nil "(~a): File ~a does not exist" #'create-default-config path))

  (let ((default-config (list (cons "firstLaunch" *json-true*) (cons "implementations" *implementations*))))
  
	(with-open-file (out path :direction :output
							  :if-exists :supersede
							  :if-does-not-exist :create)
							  
	  (json:encode-json default-config out))))

(defun init-system ()
  "creates a json file if it doesn't exist and then reads it"

  (let* ((config-filepath (get-config-path)))
    (unless (probe-file config-filepath)
	  (create-default-config config-filepath)
	  (update-implementations-config))
	
	(handler-case
		(let ((data (with-open-file (in config-filepath :direction :input)
				(json:decode-json-from-source in))))
				  
		"todo"
				  
			)
	  
	  (error (e)
	    (error "(~a): ~a~%" e)))))
  
(init-system)  
  