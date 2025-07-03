(defpackage :nnl2.tests
  (:use :cl)
  (:export :run-all-tests :run-ffi-tests))

(in-package :nnl2.tests)

(defun run-all-tests ()
  (fiveam:run-all-tests))

(defun run-ffi-tests ()
  (fiveam:run! 'nnl2.ffi.tests:nnl2.ffi-suite))
  