;; NNL2

;; Filepath: nnl2/src/lisp/fusion/fusion-package.lisp
;; File: fusion-package.lisp

;; File contains definition of :nnl2.fusion package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.fusion
  (:use :cl)
  (:export
    #:*rules*
	#:with-fusion
	#:add-rule
	#:reset-rules))
  