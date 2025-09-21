(in-package :nnl2.backends)

(defun get-avx512-status ()
  (handler-case
      (let* ((compiler "gcc")
			 (space " ")
			 (nnl2-path nnl2.intern-system:*current-dir*)
			 (avx512-status-path "avx512_test")
			 (avx512-flag "-mavx512f")
			 (full-path-to-avx512-status-c (get-test-path nnl2-path avx512-status-path))
			 (command (concatenate 'string compiler space (namestring full-path-to-avx512-status-c) space avx512-flag)))
			 
		(unless (probe-file full-path-to-avx512-status-c)
          (error (format nil "File not found: ~a" full-path-to-avx512-status-c)))	 
			 
		(multiple-value-bind (output error-output exit-code)
							 (uiop:run-program command :output :string
													   :error-output :string
													   :ignore-error-status t)

		  exit-code))														
	  
	(error (e) 1)))
  