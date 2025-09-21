(in-package :nnl2.backends)

(defun get-avx256-status ()
  (handler-case
      (let* ((compiler "gcc")
			 (space " ")
			 (nnl2-path nnl2.intern-system:*current-dir*)
			 (avx256-status-path "avx256_test")
			 (avx256-flag "-mavx")
			 (full-path-to-avx256-status-c (get-test-path nnl2-path avx256-status-path))
			 (command (concatenate 'string compiler space (namestring full-path-to-avx256-status-c) space avx256-flag)))
			 
		(unless (probe-file full-path-to-avx256-status-c)
          (error (format nil "File not found: ~a" full-path-to-avx256-status-c)))	 
			 
		(multiple-value-bind (output error-output exit-code)
							 (uiop:run-program command :output :string
													   :error-output :string
													   :ignore-error-status t)

		  exit-code))														
	  
	(error (e) 1)))
  