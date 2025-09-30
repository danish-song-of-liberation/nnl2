;; Filepath: nnl2/tests/system/system-tests-package.lisp
;; File: system-tests-package.lisp

;; Definition of :nnl2.system.tests package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.system.tests
  (:use :cl)
  (:export #:nnl2.system-suite))
  
(in-package :nnl2.system.tests)

(fiveam:def-suite nnl2.system-suite :description "Tests for nnl2.system")  
  