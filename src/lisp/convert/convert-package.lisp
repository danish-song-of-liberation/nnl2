;; NNL2

;; Filepath: nnl2/src/lisp/convert/convert-package.lisp
;; File: convert-package.lisp

;; Definition of :nnl2.convert package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.convert
  (:use :cl)
  (:export
    #:nnl2->magicl
	#:nnl2->array
	#:nnl2->list))
  