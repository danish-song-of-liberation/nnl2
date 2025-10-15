;; Filepath: nnl2/tests/convert/convert-tests-package.lisp
;; File: convert-tests-package.lisp

;; Contains definition of the :nnl2.convert.tests package 
;; and fiveam suite :nnl2.convert.tests-suite

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.convert.tests
  (:use :cl)
  (:export
    #:nnl2.convert.tests-suite))
  
(in-package :nnl2.convert.tests)

(fiveam:def-suite nnl2.convert.tests-suite :description "Tests for :nnl2.convert")
