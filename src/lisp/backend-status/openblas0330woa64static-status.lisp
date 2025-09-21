(in-package :nnl2.backends)

(defun get-openblas0330woa64-status (dir-path include-path lib-path)
  "compiles c file with openblas0330woa64static code
   
   if works correctly returns 0
   else returns 1"

  (handler-case
      (let* ((path nnl2.intern-system:*current-dir*)
			 (cpath (get-test-path dir-path "openblas0330woa64static_test"))
			 (include (concatenate 'string "-I" (namestring include-path)))
			 (lib (concatenate 'string "-L" (namestring lib-path)))
			 (shared "-lopenblas")
			 (space " ")
			 (compiler "gcc")
			 (command (concatenate 'string compiler space (namestring cpath) space include space lib space shared))) 
			 
		;; `command` must look like "gcc `path-to-backend` -I$(path-to-include) -L$(path-to-lib) -l$(shared)			  
					  
		(multiple-value-bind (output error-output exit-code) 
							 (uiop:run-program command :output :string
													   :error-output :string
													   :ignore-error-status t)										
		  
		  exit-code))
		  
	(error (e) 1)))	
