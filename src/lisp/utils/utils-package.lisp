;; NNL2

;; Filepath: nnl2/src/lisp/utils/utils-package.lisp
;; File: utils-package.lisp

;; Contains a :nnl2.utils package definition

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.utils
  (:use #:cl)
  (:export
    #:dataloader
	#:with-batch
	#:process))
  