;; Filepath: nnl2/tests/tests.lisp
;; File: tests.lisp

;; General test file. Contains the :nnl2.tests package 
;; declaration and basic testing functions

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.tests
  (:use :cl)
  (:export 
    #:run-all-tests 
	#:run-ffi-tests 
	#:run-system-tests 
	#:run-ts-tests 
	#:run-llits-tests
	#:run-convert-tests))

(in-package :nnl2.tests)

(defun nnl2-run-test (suite-name explicit)
  (if explicit
    (fiveam:run! suite-name)
	(fiveam:run suite-name)))

(defun run-all-tests ()
  (fiveam:run-all-tests))

(defun run-ffi-tests (&key explicit)
  (nnl2-run-test 'nnl2.ffi.tests:nnl2.ffi-suite explicit))
  
(defun run-ts-tests (&key explicit)
  (nnl2-run-test 'nnl2.hli.ts.tests:nnl2.hli.ts-suite explicit)
  (terpri)	
  (nnl2.log:info "All tests have been saved to a file \"~a\"" nnl2.system:+nnl2-filepath-log-path+)) 

(defun run-llits-tests (&key explicit)
  (nnl2-run-test 'nnl2.lli.ts.tests:nnl2.lli.ts.tests-suite explicit))  
  
(defun run-system-tests (&key explicit)
  (nnl2-run-test 'nnl2.system.tests:nnl2.system-suite explicit))  
  
(defun run-convert-tests (&key explicit)
  (nnl2-run-test 'nnl2.convert.tests:nnl2.convert.tests-suite explicit))
  
(defun restart-tests ()
  (when (probe-file nnl2.system:+nnl2-filepath-log-path+)
    (delete-file nnl2.system:+nnl2-filepath-log-path+)))
	
(restart-tests)	
  