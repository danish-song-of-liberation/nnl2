(defpackage :nnl2.tests
  (:use :cl)
  (:export :run-all-tests :run-ffi-tests :run-system-tests :run-ts-tests))

(in-package :nnl2.tests)

(defun run-all-tests ()
  (fiveam:run-all-tests))

(defun run-ffi-tests ()
  (fiveam:run 'nnl2.ffi.tests:nnl2.ffi-suite))
  
(defun run-ts-tests ()
  (fiveam:run 'nnl2.hli.ts.tests:nnl2.hli.ts-suite)
	
  (format t "~%[nnl2]: All tests have been saved to a file ~a~%" nnl2.system:+nnl2-filepath-log-path+)) 	
  
(defun run-system-tests ()
  (fiveam:run 'nnl2.system.tests:nnl2.system-suite))  
  
(defun restart-tests ()
  (when (probe-file nnl2.system:+nnl2-filepath-log-path+)
    (delete-file nnl2.system:+nnl2-filepath-log-path+)))
	
(restart-tests)	
  