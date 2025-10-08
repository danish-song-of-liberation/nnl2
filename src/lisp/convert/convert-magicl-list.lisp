(in-package :nnl2.convert)

;; NNL2

;; Filepath: nnl2/src/lisp/convert/convert-magicl-list.lisp
;; File: convert-magicl-list.lisp

;; Declares utilities for converting MAGICL tensors to lisp list

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun magicl->list (magicl-tensor &key flatten)
  "Convert MAGICL tensor to lisp list
  
   Args:
       magicl-tensor: Input MAGICL tensor
	   flatten (&key): Should the resulting list be flat or not
	   
   Return: if flatten returns result otherwise returns ```lisp(values result dims)```
       result: Conversion result
	   dims (values): Dimensions of input MAGICL tensor
	   
   Example 1:
       (let ((a (magicl:ones '(2 2))))
	     (print (nnl2.convert:magicl->list a))) ;; ((1.0 1.0) (1.0 1.0))
		 
   Example 2 (Flatten):
       (let ((a (magicl:zeros '(2 3))))
	     (print (nnl2.convert:magicl->list a :flatten t))) ;; (0.0 0.0 0.0 0.0 0.0 0.0)"
  
  (let ((magicl-package (find-package :magicl)))
    (unless magicl-package
      (error "MAGICL package not found"))
    
    (let ((storage-fn (find-symbol "STORAGE" magicl-package))
          (dims-fn (find-symbol "SHAPE" magicl-package)))
      
      (unless (and storage-fn (fboundp storage-fn))
        (error "storage Function not found in :magicl"))
		
      (unless (and dims-fn (fboundp dims-fn))
        (error "shape Function not found in :magicl"))
      
      (let ((storage-vector (funcall storage-fn magicl-tensor))
            (dims (funcall dims-fn magicl-tensor)))
        
        (if flatten
            (coerce storage-vector 'list)
            (let ((array (make-array dims 
                                    :element-type (array-element-type storage-vector)
                                    :displaced-to storage-vector)))
									
              (values (array->list array) dims)))))))