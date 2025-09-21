(in-package :nnl2.ffi)

(defun load-c-library (c-dir)
  "loads C library `libnnl.dll (or libnnl.so)` from the directory"

  (let ((dll-path (uiop:merge-pathnames* "libnnl.dll" c-dir)))
	(assert (probe-file dll-path) nil (format nil "File ~a is missing" dll-path))

    (handler-case
		(cffi:load-foreign-library dll-path)
	  
	  (error (e)
	    (error "(~a): CFFI Returns an error: ~a" #'load-c-library e)))))
		
(defun compile-makefile (project-path &optional debug)
  "compiles project by running `make` in the directory
   if debug is true prints compilation output and exit code."

  (let ((makefile-path (uiop:merge-pathnames* "Makefile" project-path)))
    
	(assert (probe-file makefile-path) nil (format nil "File ~a is missing" makefile-path))
	
	(multiple-value-bind (output error-output exit-code)
	  (uiop:run-program (format nil "make openblas0330woa64static_available=~d avx256_available=~d" 
						  (nnl2.system:alist-to-int nnl2.system:*openblas0330woa64static-available*)
						  (nnl2.system:alist-to-int nnl2.system:*avx256-available*))
						  
	    :error-output :interactive 
		:output :string
		:directory project-path)
		
	  (unless (zerop exit-code)
	    (error "(~a): `make` Returned a bad value: ~d~%" #'compile-makefile exit-code))
	  
	  (when debug
	    (format t "~%Exit code (~a): ~d~%Output (~a): ~a~%" #'compile-makefile exit-code output #'compile-makefile)))))
  
;; compiles and loads the C library  
(let* ((c-dir (uiop:merge-pathnames* "src/c/" nnl2.intern-system:*current-dir*)))
  (assert (uiop:directory-exists-p c-dir) nil (format nil "Path ~a is not exists" c-dir))

  (compile-makefile nnl2.intern-system:*current-dir*)
  (load-c-library c-dir))
