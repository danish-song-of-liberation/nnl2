;; NNL2

;; Filepath: nnl2/src/lisp/utils/highlevel-utils-package.lisp
;; File: highlevel-utils-package.lisp

;; Definition of :nnl2.hli.ts.utils, :nnl2.hli.ad.utils and :nnl2.hli.nn.utils packages

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage #:nnl2.hli.ts.utils
  (:use #:cl)
  
  (:export
    #:narrow
	#:swap-rows!))
	
(defpackage #:nnl2.hli.ad.utils
  (:use #:cl)
  
  (:export
    #:narrow))
		
(defpackage #:nnl2.hli.nn.utils		
  (:use #:cl)
  
  (:export
    #:get-param
	#:get-names))
  