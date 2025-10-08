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
	  (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:zeros #(5 5))))
	    (let ((b (nnl2.convert:nnl2->magicl a)))
		  ...
		  ))"
  
  (let ((magicl-package (find-package :magicl)))
    (if magicl-package
      (let ((dims (nnl2.hli.ts:shape nnl2-tensor :as :list))
			(list-to-magicl-from-list-tensor (nnl2->list nnl2-tensor :flatten t)))
			
        (funcall (find-symbol "FROM-LIST" magicl-package) list-to-magicl-from-list-tensor dims))
		
      (if (nnl2.convert:auto-install-magicl-choice)
	    (nnl2->magicl nnl2-tensor)))))
