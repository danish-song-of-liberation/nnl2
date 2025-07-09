(in-package :nnl2.backends)

(defun get-openblas0330woa64-path (dir-path)
  "returns a filepath to openblas0330woa64static_test
   dir-path is a directory to nnl2 project"

  (handler-case 
      (let ((path (uiop:merge-pathnames* 
					(make-pathname :directory '(:relative "src" "c" "backends_tests")
								   :name "openblas0330woa64static_test"
								   :type "c")
								   
					dir-path)))
					
		path)	
		
	(error (e)
	  (error "(~a): ~a~%" #'get-openblas0330woa64-path e))))

(defun get-openblas0330woa64-status (dir-path include-path lib-path)
  "compiles c file with openblas0330woa64static code
   
   if works correctly returns 0
   else returns 1"

  (handler-case
      (let* ((path nnl2.intern-system:*current-dir*)
			 (cpath (get-openblas0330woa64-path dir-path))
			 (include (concatenate 'string "-I" (namestring include-path)))
			 (lib (concatenate 'string "-L" (namestring lib-path)))
			 (shared "-lopenblas")
			 (space " ")
			 (compiler "gcc")
			 (command (concatenate 'string compiler space (namestring cpath) space include space lib space shared))) 
			 
		;; `command` must look like "gcc `path-to-backend` -I$(path-to-inclide) -L$(path-to-lib) -l$(shared)
					  
		(multiple-value-bind (output error-output exit-code) 
							 (uiop:run-program command :output :string
													   :error-output :string)										
		  
		  exit-code))
		  
	(error (e) 1)))
