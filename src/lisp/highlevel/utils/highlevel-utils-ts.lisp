(in-package :nnl2.hli.ts.utils)

;; NNL2

;; Filepath: nnl2/src/lisp/utils/highlevel-utils-ts.lisp
;; File: highlevel-utils-ts.lisp

;; Contains high-level user interface for internal utils ts functions

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun narrow (tensor &key (dim 0) start len)
  (nnl2.ffi:%narrow tensor dim start len))
  
(defun swap-rows! (tensor row-1 row-2)
  (nnl2.ffi:%swap-rows! tensor row-1 row-2))  
  
(declaim (inline narrow swap-rows!))  
  