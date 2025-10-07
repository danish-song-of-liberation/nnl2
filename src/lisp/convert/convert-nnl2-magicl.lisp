(in-package :nnl2.convert)

;; NNL2

;; Filepath: nnl2/src/lisp/convert/convert-nnl2-magicl.lisp
;; File: convert-nnl2-magicl.lisp

;; Declares utilities for converting nnl2 tensors to MAGICL tensors

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun nnl2->magicl (nnl2-tensor)
  "Converts nnl2 tensor to MAGICL tensor
   
   Args:
      nnl2-tensor: Input nnl2 tensor
	  
   Example:
	  todo
		
  "
  
  (if (cl:find-package :magicl)
    nil ;; todo
    (error "MAGICL was not found. Did you happen to forget ```(ql:quickload :magicl)``` ?")))
