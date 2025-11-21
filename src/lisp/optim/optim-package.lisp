;; NNL2

;; Filepath: nnl2/src/lisp/optim/optim-package.lisp
;; File: optim-package.lisp

;; Contains :nnl2.optim package definition

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.optim
  (:use #:cl)
  
  (:export
    #:step!
	#:zero-grad!
	#:free
	#:gd
	#:leto
	#:leto*))
  