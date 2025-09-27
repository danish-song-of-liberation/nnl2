;; NNL2

;; Filepath: nnl2/src/lisp/system/system-package.lisp
;; File: system-package.lisp

;; File contains definition on :nnl2.system package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.system
  (:use :cl)
  (:export 
    #:bool-to-int 
	#:*first-launch* 
	#:+architecture+ 
	#:*openblas0330woa64static-available* 
	#:*silent-mode* 
	#:assoc-key 
	#:greet-the-user
    #:*default-tensor-type* 
	#:+nnl2-filepath-log-path+ 
	#:*silent-mode* 
	#:*avx128-available* 
	#:alist-symbol-to-bool 
	#:*avx256-available*
    #:*avx512-available* 
	#:*avx128-available* 
	#:alist-to-int))
  