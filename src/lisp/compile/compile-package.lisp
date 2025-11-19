;; NNL2

;; Filepath: nnl2/src/lisp/compile/compile-package.lisp
;; File: compile-package.lisp

;; Definition of :nnl2.compile and :nnl2.compile.aot package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.compile
  (:use #:cl)
  
  (:shadow 
    #:compile)
	
  (:export 
    #:compile
	#:clear-compile))
  
(defpackage :nnl2.compile.aot
  (:use #:cl)
  
  (:shadow 
	#:function)
	
  (:export 
	#:function))
  