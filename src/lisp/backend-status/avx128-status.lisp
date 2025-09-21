(in-package :nnl2.backends)

(defun get-avx128-status ()
  (handler-case
      (let* ((compiler "gcc")
			 (space " ")
			 (nnl2-path nnl2.intern-system:*current-dir*)
			 (avx128-status-path "avx128_test")
			 (full-path-to-avx128-status-c (get-test-path nnl2-path avx128-status-path))
			 (command (concatenate 'string compiler space (namestring full-path-to-avx128-status-c))))
			 
		(unless (probe-file full-path-to-avx128-status-c)
          (error (format nil "File not found: ~a" full-path-to-avx128-status-c)))	 
			 
		(multiple-value-bind (output error-output exit-code)
							 (uiop:run-program command :output :string
													   :error-output :string
													   :ignore-error-status t)

		  exit-code))														
	  
	(error (e) 1)))
  