(in-package :nnl2.backends)

;; NNL2

;; Filepath: nnl2/src/lisp/backend-status/openblas0330woa64static-status.lisp
;; File: openblas0330woa64static-status.lisp

;; Contains the function for check openblas status

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun get-openblas0330woa64-status (dir-path include-path lib-path)
  "Compiles c file with openblas0330woa64static code
   
   If openblas is available returns 0
   Else returns 1"

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
