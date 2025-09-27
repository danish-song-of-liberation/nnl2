;; NNL2

;; Filepath: nnl2/src/lisp/backend-status/backend-status-package.lisp
;; File: backend-status-package.lisp

;; Defining the :nnl2.backends package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.backends
  (:use :cl)
  (:export 
    #:get-test-path 
	#:get-openblas0330woa64-status 
	#:get-avx128-status 
	#:get-avx512-status
    #:get-avx256-status))
  