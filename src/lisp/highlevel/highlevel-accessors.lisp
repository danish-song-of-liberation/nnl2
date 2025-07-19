(in-package :nnl2.hli.ts)

(defun make-shape-pntr (shape)
  (declare (type vector shape))

  (let* ((len (the (integer 0 *) (length shape)))
		(shape-pntr (the cffi:foreign-pointer (cffi:foreign-alloc :int :count len))))
		
	(declare (type (integer 0 *) len))
	(declare (type cffi:foreign-pointer shape-pntr))
		
    (loop for i from 0 below len
          do (setf (cffi:mem-aref shape-pntr :int i) (aref shape i)))
		  
    (values shape-pntr len)))
	
(declaim (inline make-shape-pntr))	
	
(defmacro free (tensor)
  `(nnl2.ffi:free-tensor ,tensor))
	
(defmacro tlet ((&rest bindings) &body body)
  (let ((vars (mapcar #'car bindings)))
    `(let ,bindings
	   (unwind-protect
	       (progn ,@body)
		   
		 (progn 
		   ,@(loop for var in vars
		           collect `(free ,var)))))))

(defmacro empty (indices &key (dtype nnl2.system:*default-tensor-type*))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    `(nnl2.ffi:%empty ,shape ,rank ,dtype))) 
	
(defmacro empty-with-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  `(nnl2.ffi:%empty ,shape-pntr ,rank ,dtype))

(defmacro zeros (indices &key (dtype nnl2.system:*default-tensor-type*))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    `(nnl2.ffi:%zeros ,shape ,rank ,dtype))) 

(defmacro zeros-with-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  `(nnl2.ffi:%zeros ,shape-pntr ,rank ,dtype))

(defmacro ones (indices &key (dtype nnl2.system:*default-tensor-type*))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    `(nnl2.ffi:%ones ,shape ,rank ,dtype))) 

(defmacro ones-with-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  `(nnl2.ffi:%ones ,shape-pntr ,rank ,dtype))

(defmacro full (indices &key (dtype nnl2.system:*default-tensor-type*) (filler 0.0d0))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    (let ((filler-pntr (cffi:foreign-alloc :double)))
	  (setf (cffi:mem-ref filler-pntr :double) filler)
     `(nnl2.ffi:%full ,shape ,rank ,dtype ,filler-pntr))))
	 
(defmacro full-with-pntr (shape-pntr rank &key filler (dtype nnl2.system:*default-tensor-type*))
  `(nnl2.ffi:%full ,shape-pntr ,rank ,dtype ,filler))	 
