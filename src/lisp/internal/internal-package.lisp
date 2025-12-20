;; NNL2

;; Filepath: nnl2/src/lisp/internal/internal-package.lisp
;; File: internal-package.lisp

;; Definition of :nnl2.internal package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.internal
  (:use #:cl)
  (:export
    #:ts-axpy-regional!
	#:ad-share-data
	#:ts-concat-vectors
	#:ad-concat-vectors
	#:ts-vector-as-parameter
	#:ts-assign-row
	#:ad-assign-row))
  