(in-package :nnl2.backends)

;; NNL2

;; Filepath: nnl2/src/lisp/backend-status/avx128-status.lisp
;; File: avx128-status.lisp

;; Checks if AVX128 is available

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun get-avx128-status ()
  "Checks if AVX128 is available
   Returns:
     0 if available
	 1 if not available"
	 
  (handler-case
      (let* ((compiler "gcc")
			 (space " ")
			 (nnl2-path nnl2.intern-system:*current-dir*)
			 (avx128-status-path "avx128_test")
			 (avx128-flag "-mavx")
			 (full-path-to-avx128-status-c (get-test-path nnl2-path avx128-status-path))
			 (command (concatenate 'string compiler space (namestring full-path-to-avx128-status-c) space avx128-flag)))
			 
		(unless (probe-file full-path-to-avx128-status-c)
          (error (format nil "File not found: ~a" full-path-to-avx128-status-c)))	 
			 
		(multiple-value-bind (output error-output exit-code)
							 (uiop:run-program command :output :string
													   :error-output :string
													   :ignore-error-status t)

		  exit-code))														
	  
	(error (e) 1)))
  