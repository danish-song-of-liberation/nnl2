;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/highlevel-accessors.lisp
;; File: highlevel-accessors.lisp

;; Definition of :nnl2.lli and :nnl2.lli.ts package where
;; lli stays for low-level-interface, ts stays for
;; tensor system

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.lli
  (:use :cl)
  (:export
    #:fast-mem-aref-setter
	#:fast-mem-aref-getter
    #:alignment))
  
(defpackage :nnl2.lli.ts
  (:use :cl)
  (:export 
    #:flat 
    #:trefw 
    #:data 
    #:mem-aref 
	#:piatd
	#:iatd
	#:iats
    #:iterate-across-tensor-data 
	#:parallel-iterate-across-tensor-data
    #:iterate-across-tensor-shape))
  