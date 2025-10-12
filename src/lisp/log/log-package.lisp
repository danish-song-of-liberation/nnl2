;; NNL2

;; Filepath: nnl2/src/lisp/log/log-package.lisp
;; File: log-package.lisp

;; Contains a definition of :nnl2.log package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.log
  (:use :cl)
  
  (:shadow 
    #:error
	#:warning
	#:debug)
	
  (:export
    #:error
	#:warning
	#:fatal
	#:debug
	#:info))
