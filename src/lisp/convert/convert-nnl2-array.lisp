(in-package :nnl2.convert)

;; NNL2

;; Filepath: nnl2/src/lisp/convert/convert-nnl2-array.lisp
;; File: convert-nnl2-array.lisp

;; Declares utilities for converting nnl2 tensors to lisp array

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun nnl2->array (nnl2-tensor &key flatten)
  "Convert nnl2 tensor to lisp array
  
   Args:
       nnl2-tensor: Input tensor
	   flatten (&key): Should the resulting array be flat or not
	   
   Example 1:
       (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:ones #(5 5))))
	     (print (nnl2.convert:nnl2->array a)))
		 
   Example 2 (Flatten):
       (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:zeros #(3 3))))
	     (print (nnl2.convert:nnl2->array a :flatten t))) ;; #(0 0 0 0 0 0 0 0 0)"
  
  (let* ((raw-data (nnl2.lli.ts:data nnl2-tensor))
         (dims (nnl2.hli.ts:shape nnl2-tensor :as :list))  
         (total-size (nnl2.hli.ts:size nnl2-tensor))
         (nnl2-type (the keyword (nnl2.hli.ts:dtype nnl2-tensor)))
         (element-type (the symbol (nnl2.hli.ts:type/nnl2->lisp nnl2-type)))
         (flat-array (the simple-array (make-array total-size :element-type element-type))))
    
    (let ((reader (ecase nnl2-type
                    (:float64 (the function #'nnl2.ffi:mem-aref-getter-float64))
                    (:float32 (the function #'nnl2.ffi:mem-aref-getter-float32))
                    (:int32   (the function #'nnl2.ffi:mem-aref-getter-int32)))))
      
      (nnl2.threading:pdotimes (i total-size)
        (setf (row-major-aref flat-array i) (funcall reader raw-data i))))
    
    (if flatten
      flat-array
      (make-array dims 
        :element-type element-type
        :displaced-to flat-array))))
					