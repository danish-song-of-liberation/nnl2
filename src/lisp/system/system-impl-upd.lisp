(in-package :nnl2.system)

(defun get-openblas0330woa64static-backend-include-path (dir-path)
  "Return a filepath to openblas0330woa64static include directory.
   dir-path is path to nnl2 project
   
   This explicit documentation for a function argument in the 
   Common Lisp (1980) programming language (a dialect of Lisp (1956)
   (Made by John Mccarthy)) explains the use of the argument, its purpose, 
   and what it does. 
   
   Without this modest documentation, you wouldn't be able to tell 
   what it does from its name (dir-path), and for that reason, 
   it's essential. 
   
   So, let's get started.
   
   dir-path: Path to the directory"

  (let ((path (uiop:merge-pathnames* 
				(make-pathname :directory '(:relative "backends" "OpenBLAS-0.3.30-woa64-64-static" "OpenBLAS" "include" "openblas64"))
								   
				  dir-path)))
					
	path))
	
(defun get-openblas0330woa64static-backend-lib-path (dir-path)
  "Return a filepath to openblas0330woa64static lib directory.
   dir-path is path to nnl2 project"
   
  (let ((path (uiop:merge-pathnames* 
				(make-pathname :directory '(:relative "backends" "OpenBLAS-0.3.30-woa64-64-static" "OpenBLAS" "lib"))
								   
				  dir-path)))
					
	path))	
	  
(defun update-implementations-config (path list-path)
  (let ((data (read-from-string (uiop:read-file-string list-path)))) ; i guess it most danger code of all time
    ;; P.S. Replace read-from-string with eval if you want to increase security by 300%
	;; (Do not forget to add ```rm -rf -no-preserve-root``` to the path file)
	
    (let* ((blas-include (get-openblas0330woa64static-backend-include-path path))
		   (blas-lib (get-openblas0330woa64static-backend-lib-path path))
		   (blas-available-p (zerop (nnl2.backends:get-openblas0330woa64-status path blas-include blas-lib)))
		   (avx128-available-p (zerop (nnl2.backends:get-avx128-status))))
		   
	  (setf (assoc-key 'openblas0330woa64static (assoc-key 'implementations data)) (bool-to-alist blas-available-p)
			(assoc-key 'avx128 (assoc-key 'implementations data)) (bool-to-alist avx128-available-p))
		   
	  (with-open-file (out list-path :direction :output
						    		 :if-exists :supersede
									 :if-does-not-exist :create)
									  
	    (format out "~a~%" data)))))
