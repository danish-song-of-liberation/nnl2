;; NNL2

;; Filepath: nnl2/src/lisp/threading/threading.lisp
;; File: system-vars.lisp

;; File contains definition of :nnl2.threading package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.threading
  (:use :cl)
  (:export
    #:pdotimes
	#:pmapcar
	#:pmap
	#:pmapc
	#:pmapcan
	#:pmaplist
	#:pmap-into
	#:pmapl
	#:plet
	#:plet-if
	#:por
	#:pand))
  