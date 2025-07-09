(in-package :nnl2.system)

(defun get-openblas0330woa64static-backend-include-path (dir-path)
  "return a filepath to openblas0330woa64static include directory.
   dir-path is path to nnl2 project"

  (let ((path (uiop:merge-pathnames* 
				(make-pathname :directory '(:relative "backends" "OpenBLAS-0.3.30-woa64-64-static" "OpenBLAS" "include" "openblas64"))
								   
				  dir-path)))
					
	path))
	
(defun get-openblas0330woa64static-backend-lib-path (dir-path)
  "return a filepath to openblas0330woa64static lib directory.
   dir-path is path to nnl2 project"
   
  (let ((path (uiop:merge-pathnames* 
				(make-pathname :directory '(:relative "backends" "OpenBLAS-0.3.30-woa64-64-static" "OpenBLAS" "lib"))
								   
				  dir-path)))
					
	path))	

(defun update-implementations-config (path json-path)
  "updates the json file based on whether the backends are working or not"
  
  (let ((data (with-open-file (in json-path :direction :input) (json:decode-json-from-source in))))
    (with-open-file (out json-path :direction :output :if-exists :supersede)
	  (let* ((blas-include (get-openblas0330woa64static-backend-include-path path))
	    	 (blas-lib (get-openblas0330woa64static-backend-lib-path path))
			 (available-p (if (zerop (nnl2.backends:get-openblas0330woa64-status path blas-include blas-lib)) t nil)))
		 
	    (setf (cdr (cadadr data)) (bool-to-json-string available-p))) ; wtf is that 
		
	  (json:encode-json data out))))
 