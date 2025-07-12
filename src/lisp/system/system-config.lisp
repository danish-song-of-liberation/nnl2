(in-package :nnl2.system)

#| PLEASE READ ME

   Some parts of the code contain very strange solutions, 
   but please don't blame me. It's the cl-json library 
   that handles t, nil, and kebab-case in a very strange way,
   forcing me to resort to such measures.   |#

(defparameter *first-launch* nil
  "flag indicating if this is the first system launch")
  
(defparameter *openblas0330woa64static-available* nil)

(defparameter *silent-mode* nil)

(defparameter *implementations* (list 
								  (cons 'openblas0330woa64static *json-false*))
								 
  "Contains an associative list for subsequent conversion 
   to json format in the format (implementation . *is-it-working-or-not*)")

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
	  (update-implementations-config nnl2.intern-system:*current-dir* config-filepath))
	
	(handler-case
	    (let ((data (with-open-file (in config-filepath :direction :input)
					  (json:decode-json-from-source in))))
					  
		  (with-open-file (stream config-filepath :direction :io
											      :if-exists :supersede
												  :if-does-not-exist :create)
										
			(let* ((first-launch-json (cdar data))
				   (first-launch-p (json-string-to-bool first-launch-json))
				   (openblas0330woa64static-available-json (cdr (cadadr data))) ; im so sorry for that its cause cl-json is so buggy and cant correctly write key so assoc is broken
				   (openblas0330woa64static-available-p (json-string-to-bool openblas0330woa64static-available-json)))
				   
			  (setf *openblas0330woa64static-available* openblas0330woa64static-available-p)
				   
			  (setf *first-launch* first-launch-p)
				   
    		  (when first-launch-p ; first launch
		        (setf (cdar data) *json-false*)) ; to affect setter
			    
		      (unwind-protect
				(json:encode-json data stream)))))
	  
	  (error (e)
	    (error "(~a): ~a~%" e)))))
  
(init-system)  
  
(defparameter +architecture+ 
  (list
    (cons '*openblas0330woa64static* *openblas0330woa64static-available*)))
  