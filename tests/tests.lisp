(defpackage :nnl2.tests
  (:use :cl)
  (:export :run-all-tests :run-ffi-tests :run-system-tests :run-ts-tests))

(in-package :nnl2.tests)

(defun run-all-tests ()
  (fiveam:run-all-tests))

(defun run-ffi-tests ()
  (fiveam:run 'nnl2.ffi.tests:nnl2.ffi-suite))
  
(defun run-ts-tests (&key debug)
  (if debug
    (fiveam:run! 'nnl2.hli.ts.tests:nnl2.hli.ts-suite)
    (fiveam:run 'nnl2.hli.ts.tests:nnl2.hli.ts-suite)))
  
(defun run-system-tests ()
  (fiveam:run 'nnl2.system.tests:nnl2.system-suite))  
  