;; NNL2

;; Filepath: nnl2/tests/ffi/ffi-tests-package.lisp
;; File: ffi-tests-package.lisp

;; Definition of :nnl2.ffi.tests package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.ffi.tests
  (:use :cl)
  (:export #:nnl2.ffi-suite))

(in-package :nnl2.ffi.tests)

(fiveam:def-suite nnl2.ffi-suite :description "Tests for nnl2.ffi")  
  