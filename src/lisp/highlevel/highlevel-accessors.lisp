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

(defmacro print-tensor (tensor)
  `(nnl2.ffi:print-tensor ,tensor))

(defmacro rank (tensor)
  `(nnl2.ffi:get-tensor-rank ,tensor))
  
(defmacro dtype (tensor)
  `(nnl2.ffi:get-tensor-dtype ,tensor))  

(defmacro int-dtype (tensor)
  `(nnl2.ffi:get-int-tensor-dtype ,tensor))    
  
(defmacro shape-pointer (tensor)
  `(nnl2.ffi:get-pointer-to-tensor-shape ,tensor))

(defun get-shape-as-list (tensor rank)
  (loop with rank-t = (if rank rank (rank tensor))
		with shape-pointer = (shape-pointer tensor)
		for i from 0 below rank-t
		collect (cffi:mem-aref shape-pointer :int i)))

(defun get-shape-as-vector (tensor rank)
  (let* ((rank-t (if rank rank (rank tensor)))
		 (vec (make-array rank-t))
		 (shape-pointer (shape-pointer tensor)))
		 
	(dotimes (i rank-t)
	  (setf (aref vec i) (cffi:mem-aref shape-pointer :int i)))
	  
	vec))
	
(defmacro shape (tensor &key (as :vector) (rank nil))
  (case as
    (:list `(get-shape-as-list ,tensor ,rank))
	(:vector `(get-shape-as-vector ,tensor ,rank))
	(otherwise (error "Unknown type: ~a~%" as))))

(defun gemm (a b &key (order :nnl2rowmajor) (transa :nnl2notrans) (transb :nnl2notrans) (alpha 1.0d0) (beta 0.0d0) m n k lda ldb)
  (let* ((shape-a (get-shape a :as :vector :rank 2))
		 (shape-b (get-shape b :as :vector :rank 2))
		 (m (if m m (aref shape-a 0)))
		 (n (if n n (aref shape-b 1)))
		 (k (if k k (aref shape-a 1)))
		 (lda (if lda lda k))
		 (ldb (if ldb ldb n)))
		 
	(nnl2.ffi:%gemm order transa transb m n k alpha a lda b ldb beta)))
  
(defun gemm! (a b &key out (order :nnl2rowmajor) (transa :nnl2notrans) (transb :nnl2notrans) (alpha 1.0d0) (beta 0.0d0) m n k lda ldb ldc)
  (let* ((shape-a (get-shape a :as :vector :rank 2))
		 (shape-b (get-shape b :as :vector :rank 2))
		 (shape-out (get-shape out :as :vector :rank 2))
		 (m (if m m (aref shape-a 0)))
		 (n (if n n (aref shape-b 1)))
		 (k (if k k (aref shape-a 1)))
		 (lda (if lda lda k))
		 (ldb (if ldb ldb n))
		 (ldc (if ldc ldc (aref shape-out 1))))
		 
	(nnl2.ffi:%gemm! order transa transb m n k alpha a lda b ldb beta out ldc)))  
  
(defmacro += (summand sumend) 
  `(nnl2.ffi:%+= ,summand ,sumend))  
  
(defmacro -= (summand sumend) 
  `(nnl2.ffi:%-= ,summand ,sumend))  ;sry for the naming like summand sumend i  remake the after
  
(defmacro + (summand addend)
  `(nnl2.ffi:%+ ,summand ,addend))  
  
(defmacro - (minuend subtrahend)
  `(nnl2.ffi:%- ,minuend ,subtrahend))     
    
(defmacro size (tensor)
  `(nnl2.ffi:%get-size ,tensor))

(defmacro size-in-bytes (tensor)
  `(nnl2.ffi:%get-size-in-bytes ,tensor))  
  