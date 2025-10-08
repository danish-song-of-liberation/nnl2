(in-package :nnl2.convert)

;; NNL2

;; Filepath: nnl2/src/lisp/convert/convert-magicl-nnl2.lisp
;; File: convert-magicl-nnl2.lisp

;; Declares utilities for converting MAGICL tensors to nnl2 tensor

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun magicl->nnl2 (magicl-tensor)
  "Converts MAGICL tensor to nnl2 tensor
  
   Args:
       magicl-tensor: Input MAGICL tensor

   Example:
	   (let ((a (magicl:ones '(3 3))))
	     (nnl2.hli.ts:tlet ((b (nnl2.convert:magicl->nnl2 a)))
		   (nnl2.hli.ts:print-tensor b)))"
  
  (let ((magicl-package (find-package :magicl)))
    (if magicl-package
        (let ((storage-fn (find-symbol "STORAGE" magicl-package))
              (shape-fn (find-symbol "SHAPE" magicl-package))
              (dimensions-fn (find-symbol "DIMENSIONS" magicl-package)))
          
          (when (or (null storage-fn) (not (fboundp storage-fn)))
            (error "magicl::storage Function not found in :magicl package"))
          
          (when (or (null shape-fn) (not (fboundp shape-fn)))
            (error "magicl:shape Function not found in :magicl package"))
          
          (let ((storage-data (funcall storage-fn magicl-tensor))
                (tensor-shape (if (fboundp shape-fn)
                                (funcall shape-fn magicl-tensor)
                                (funcall dimensions-fn magicl-tensor))))
            
            (nnl2.hli.ts:from-flatten storage-data tensor-shape)))
        
        (if (nnl2.convert:auto-install-magicl-choice)
	      (magicl->nnl2 nnl2-tensor)))))
		