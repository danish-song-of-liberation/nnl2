(in-package :nnl2.tests.utils)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-activation-functions.lisp
;; File: ts-tests-activation-functions.lisp

;; Implements the approximately-equal function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun approximately-equal (x y &key (tolerance 0.1))
   (<= (abs (- x y)) tolerance))
