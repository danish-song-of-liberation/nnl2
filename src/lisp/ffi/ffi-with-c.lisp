(in-package :nnl2.system)

(defun load-c-library (c-dir)
  (let ((dll-path (uiop:merge-pathnames* "libnnl.dll" c-dir)))
    
	(assert (probe-file dll-path) nil (format nil "File ~a is missing" dll-path))

	(cffi:load-foreign-library dll-path)))
		
(defun compile-makefile (project-path &optional debug)
  (let ((makefile-path (uiop:merge-pathnames* "Makefile" project-path)))
    
	(assert (probe-file makefile-path) nil (format nil "File ~a is missing" makefile-path))
	
	(multiple-value-bind (output error-output exit-code)
	  (uiop:run-program (format nil "make") 
	    :error-output :interactive 
		:output :string
		:directory project-path)
	  
	  (when debug
	    (format t "~%Exit code (~a): ~d~%Output (~a): ~a~%" #'compile-makefile exit-code output #'compile-makefile)))))
  
(let* ((c-dir (uiop:merge-pathnames* "src/c/" *current-dir*)))
  (compile-makefile *current-dir*)
  (load-c-library c-dir))
