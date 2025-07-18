(in-package :nnl2.system)

#| PLEASE READ ME

   Some parts of the code contain very strange solutions, 
   but please don't blame me. It's the cl-json library 
   that handles t, nil, and kebab-case in a very strange way,
   forcing me to resort to such measures.   |#

(defparameter *first-launch* nil
  "flag indicating if this is the first system launch")
  
(defparameter *naive-available* t)  
(defparameter *openblas0330woa64static-available* nil)

(defparameter *default-tensor-type* :float64)

(defparameter *silent-mode* nil)

(defparameter *implementations* (list 
								  (cons 'naive *alist-true*)
								  (cons 'openblas0330woa64static *alist-false*))
								 
  "Contains an associative list for subsequent conversion 
   to json format in the format (implementation . *is-it-working-or-not*)")

(defun get-config-path ()
  (uiop:merge-pathnames*
    (make-pathname :directory '(:relative "src" "lisp" "system")
				   :name "config"
				   :type "alist")
				   
	nnl2.intern-system:*current-dir*))

(defun create-default-config (path)
  "creates json config file"
  
  (assert (not (probe-file path)) nil (format nil "(~a): File ~a does not exist" #'acreate-default-config path))

  (let ((default-config (list (cons 'first-launch *alist-true*) (cons 'implementations *implementations*))))
  
	(with-open-file (out path :direction :output
							  :if-exists :supersede
							  :if-does-not-exist :create)
							  
	  (format out "~a~%" default-config))))	  	  

(defun init-system ()
  "creates a json file if it doesn't exist and then reads it"

  (let* ((config-filepath (get-config-path)))
	  
	(unless (probe-file config-filepath)
	  (create-default-config config-filepath)
	  (update-implementations-config nnl2.intern-system:*current-dir* config-filepath))
	
	(handler-case
	    (let ((data (read-from-string (uiop:read-file-string config-filepath))))
					  		
		  (with-open-file (stream config-filepath :direction :io
												   :if-exists :supersede
												   :if-does-not-exist :create)
												   
			(let* ((first-launch-json (assoc-key 'first-launch data))
				   (first-launch-p (alist-symbol-to-bool first-launch-json))
				   (naive-available-alist (assoc-key 'naive (assoc-key 'implementations data)))
				   (naive-available-p (alist-symbol-to-bool naive-available-alist))
				   (openblas0330woa64static-available-alist (assoc-key 'openblas0330woa64static (assoc-key 'implementations data)))
				   (openblas0330woa64static-available-p (alist-symbol-to-bool openblas0330woa64static-available-alist)))
				   
			  (setf *naive-available* naive-available-p
					*openblas0330woa64static-available* openblas0330woa64static-available-p	     
					*first-launch* first-launch-p)
				   
    		  (when first-launch-p (setf (assoc-key 'first-launch data) *alist-false*)) 
			  
			  (unwind-protect
			    (format stream "~a~%" data)))))
	  
	  (error (e)
	    (error "(~a): ~a~%" e)))))
  
(init-system)  
  
(defparameter +architecture+ 
  (list
    (cons '*naive* *naive-available*)
	(cons '*openblas0330woa64static* *openblas0330woa64static-available*)))
  