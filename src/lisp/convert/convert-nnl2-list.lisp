(in-package :nnl2.convert)

(defun nnl2->list (nnl2-tensor &key flatten)
  "Convert nnl2 tensor to lisp list
  
   Args:
       nnl2-tensor: Input nnl2 tensor
	   flatten (&key): Should the resulting list be flat or not
	   
   Return: if flatten returns result otherwise returns ```lisp(values result dims)```
       result: Conversion result
	   dims (values): Dimensions of input nnl2 tensor
	   
   Example 1:
       (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:ones #(2 2))))
	     (print (nnl2.convert:nnl2->list a))) ;; ((1.0 1.0) (1.0 1.0))
		 
   Example 2 (Flatten):
       (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:zeros #(2 3))))
	     (print (nnl2.convert:nnl2->list a :flatten t))) ;; (0.0 0.0 0.0 0.0 0.0 0.0)"		 
  
  (let ((array (nnl2->array nnl2-tensor :flatten flatten)))
    (if flatten
        (coerce array 'list)
        (let ((dims (array-dimensions array)))
          (values (array->nested-list array) dims)))))
		  