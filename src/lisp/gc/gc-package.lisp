;; NNL2

;; Filepath: nnl2/src/lisp/gc/gc-package.lisp
;; File: gc-package.lisp

;; Contains a definition of :nnl2.gc package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.gc
  (:use #:cl)
  (:shadow #:push)
  
  (:export
    #:*gc*
	#:*profile*
	#:gc
	#:push
	#:clear
	#:with-gc))
  