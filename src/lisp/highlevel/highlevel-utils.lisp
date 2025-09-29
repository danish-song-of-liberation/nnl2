(in-package :nnl2.hli)

;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/highlevel-utils.lisp
;; File: highlevel-utils.lisp

;; Contains utils for high-level-interface

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun make-foreign-pointer (value dtype)
  (let ((pntr (cffi:foreign-alloc dtype)))
    (setf (cffi:mem-ref pntr dtype) value)
	
	pntr))
