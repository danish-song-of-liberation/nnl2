(in-package :nnl2.system)

(defparameter *first-launch* nil
  "Flag indicating if this is the first system launch")
  
(defparameter *naive-available* t)  
(defparameter *openblas0330woa64static-available* nil)
(defparameter *avx128-available* nil)
(defparameter *avx512-available* nil)

(defparameter *default-tensor-type* :float64)

(defparameter *silent-mode* nil)

(defparameter *implementations* (list 
								  (cons 'naive *alist-true*)
								  (cons 'openblas0330woa64static *alist-false*)
								  (cons 'avx128 *alist-false*)
								  (cons 'avx512 *alist-false*))
								 
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
  
  (assert (not (probe-file path)) nil (format nil "(~a): File ~a does not exist" #'create-default-config path))

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
												   
			(let* ((first-launch-alist (assoc-key 'first-launch data))
				   (first-launch-p (alist-symbol-to-bool first-launch-alist))
				   (naive-available-alist (assoc-key 'naive (assoc-key 'implementations data)))
				   (openblas0330woa64static-available-alist (assoc-key 'openblas0330woa64static (assoc-key 'implementations data)))
				   (avx128-available-alist (assoc-key 'avx128 (assoc-key 'implementations data)))
				   (avx512-available-alist (assoc-key 'avx512 (assoc-key 'implementations data))))
				   
			  (setf *naive-available* naive-available-alist
					*openblas0330woa64static-available* openblas0330woa64static-available-alist   
					*avx128-available* avx128-available-alist
					*avx512-available* avx512-available-alist
					*first-launch* first-launch-alist)
				   
    		  (when first-launch-p (setf (assoc-key 'first-launch data) *alist-false*)) 
			  
			  (unwind-protect
			    (format stream "~a~%" data)))))
	  
	  (error (e)
	    (error "(~a): ~a~%" e)))))
  
(init-system)  
  
(defparameter +architecture+ 
  (list
    (cons '*naive* *naive-available*)
	(cons '*openblas0330woa64static* *openblas0330woa64static-available*)
	(cons '*avx128* *avx128-available*)
	(cons '*avx512* *avx512-available*)))
  