(in-package :nnl2.convert)

;; NNL2

;; Filepath: nnl2/src/lisp/convert/convert-magicl-array.lisp
;; File: convert-magicl-array.lisp

;; Declares utilities for converting MAGICL tensors to lisp array

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun magicl->array (magicl-tensor &key flatten)
  "Convert MAGICL tensor to Lisp array
  
   Args:
       magicl-tensor: Input MAGICL tensor
	   flatten (&key): Should the resulting array be flat or not
	   
   Example 1:
       (let ((a (magicl:ones '(3 3))))
         (print (nnl2.convert:magicl->array a)))
		 
   Example 2 (Flatten):
       (let ((a (magicl:zeros '(2 4))))
         (print (nnl2.convert:magicl->array a :flatten t)))"
  
  (let ((magicl-package (find-package :magicl)))
    (unless magicl-package
      (if (nnl2.convert:auto-install-magicl-choice) ;; dont even want to know how you passed the tensor to magicl without loading it
	    (magicl->array magicl-tensor :flatten flatten)))
    
    (let ((storage-fn (find-symbol "STORAGE" magicl-package))
          (dims-fn (find-symbol "SHAPE" magicl-package)))
      
      (unless (and storage-fn (fboundp storage-fn)) (error "storage Function not found in :magicl"))	
      (unless (and dims-fn (fboundp dims-fn))       (error "shape Function not found in :magicl"))
      
      (let ((storage-vector (funcall storage-fn magicl-tensor))
            (dims (funcall dims-fn magicl-tensor)))
        
        (if flatten
            storage-vector  
            (make-array dims 
                       :element-type (array-element-type storage-vector)
                       :displaced-to storage-vector))))))
					   