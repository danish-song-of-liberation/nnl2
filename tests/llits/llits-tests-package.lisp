;; Filepath: nnl2/tests/llits/llits-tests-package.lisp
;; File: llits-tests-package.lisp

;; Contains definition of the :nnl2.lli.ts.tests package 
;; and fiveam suite :nnl2.lli.ts.tests-suite

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.lli.ts.tests
  (:use :cl)
  (:export
    #:nnl2.lli.ts.tests-suite))
  
(in-package :nnl2.lli.ts.tests)

(fiveam:def-suite nnl2.lli.ts.tests-suite :description "Tests for nnl2.lli.ts")  
  
