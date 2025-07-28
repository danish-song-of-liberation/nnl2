(in-package :nnl2.hli.ts)

(defun make-shape-pntr (shape)
  (let* ((len (the (integer 0 *) (length shape)))
	     (shape-pntr (the cffi:foreign-pointer (cffi:foreign-alloc :int :count len))))
		
	(declare (type (integer 0 *) len))
	(declare (type cffi:foreign-pointer shape-pntr))
		
	(if (typep shape 'vector)	
      (loop for i from 0 below len
            do (setf (cffi:mem-aref shape-pntr :int i) (aref shape i)))
			
	  (loop for i from 0 below len
		    do (setf (cffi:mem-aref shape-pntr :int i) (nth i shape))))
		  
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

(defmacro tlet* ((&rest bindings) &body body)
  (if (null bindings)
   `(progn ,@body)
    (let* ((binding (first bindings))
           (var (if (consp binding) (car binding) binding))
           (value (if (consp binding) (cadr binding) nil)))
     `(let (,binding)
        (unwind-protect
          (tlet* ,(rest bindings) ,@body)
          (free ,var))))))

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
  (declare (optimize (speed 3) (safety 0)))

  (let* ((shape-a (shape a :as :vector :rank 2))
		 (shape-b (shape b :as :vector :rank 2))
		 (m (if m m (aref shape-a 0)))
		 (n (if n n (aref shape-b 1)))
		 (k (if k k (aref shape-a 1)))
		 (lda (if lda lda k))
		 (ldb (if ldb ldb n)))
		 
	(nnl2.ffi:%gemm order transa transb m n k alpha a lda b ldb beta)))
  
(defun gemm! (a b &key out (order :nnl2rowmajor) (transa :nnl2notrans) (transb :nnl2notrans) (alpha 1.0d0) (beta 0.0d0) m n k lda ldb ldc)
  (let* ((shape-a (shape a :as :vector :rank 2))
		 (shape-b (shape b :as :vector :rank 2))
		 (shape-out (shape out :as :vector :rank 2))
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
  
(defmacro .+ (summand addend)
  `(nnl2.ffi:%+ ,summand ,addend))  
  
(defmacro .- (minuend subtrahend)
  `(nnl2.ffi:%- ,minuend ,subtrahend))    

(defmacro *= (multiplicand multiplier)
  `(nnl2.ffi:%*= ,multiplicand ,multiplier))  

(defmacro /! (dividend divisor)
  `(nnl2.ffi:%/= ,dividend ,divisor))  
  
(defmacro .* (multiplicand multiplier)
  `(nnl2.ffi:%* ,multiplicand ,multiplier))  
  
(defmacro ./ (dividend divisor)
  `(nnl2.ffi:%/ ,dividend ,divisor))  
  
(defmacro ^= (base exponent)
  `(nnl2.ffi:%^= ,base ,exponent))

(defmacro .^ (base exponent)
  `(nnl2.ffi:%.^ ,base ,exponent))

(defmacro .exp! (tensor)
  `(nnl2.ffi:%.exp! ,tensor))

(defmacro .exp (tensor)
  `(nnl2.ffi:%.exp ,tensor))  
  
(defmacro .log! (tensor)
  `(nnl2.ffi:%.log! ,tensor))  
  
(defmacro .log (tensor)
  `(nnl2.ffi:%.log ,tensor))    
    
(defmacro size (tensor)
  `(nnl2.ffi:%get-size ,tensor))

(defmacro size-in-bytes (tensor)
  `(nnl2.ffi:%get-size-in-bytes ,tensor))  
  
(defun tref (tensor &rest shape)
  (declare (optimize (speed 3)))

  (let* ((tensor-rank (rank tensor))
	     (tensor-dtype (dtype tensor))
		 (shape (make-shape-pntr shape))
		 (void-ptr (nnl2.ffi:%tref tensor shape tensor-rank)))
		 
	(case tensor-dtype
	  (:float64 (cffi:mem-ref void-ptr :double))	 
      (:float32 (cffi:mem-ref void-ptr :float))
	  (:int32 (cffi:mem-ref void-ptr :int))))) 
	  
(defun (setf tref) (change-to tensor &rest shape)
  (let* ((shape-rank (length shape))
		 (shape (make-shape-pntr (subst -1 '* shape)))
	     (tensor-dtype (dtype tensor)))
		 
	(case tensor-dtype
	  (:float64 
	    (let ((changer (cffi:foreign-alloc :double)))
		  (setf (cffi:mem-ref changer :double) change-to)
		  (nnl2.ffi:%tref-setter tensor shape shape-rank changer)))
		  
	  (:float32
	    (let ((changer (cffi:foreign-alloc :float)))
		  (setf (cffi:mem-ref changer :float) change-to)
		  (nnl2.ffi:%tref-setter tensor shape shape-rank changer)))

	  (:int32
	    (let ((changer (cffi:foreign-alloc :int)))
		  (setf (cffi:mem-ref changer :int) change-to)
		  (nnl2.ffi:%tref-setter tensor shape shape-rank changer))))))
		  
(defmacro scale! (tensor multiplier)
  `(nnl2.ffi:%scale! ,tensor (float ,multiplier)))

(defmacro scale (tensor multiplier)
  `(nnl2.ffi:%scale ,tensor (float ,multiplier)))  
  
(defmacro empty-like (tensor)
  `(nnl2.ffi:%empty-like ,tensor))  
  
(defmacro zeros-like (tensor)
  `(nnl2.ffi:%zeros-like ,tensor))  
	
(defmacro ones-like (tensor)
  `(nnl2.ffi:%ones-like ,tensor))  
		
(defmacro full-like (tensor &key (filler 0.0d0))
  (let ((filler-pntr (cffi:foreign-alloc :double)))
    (setf (cffi:mem-ref filler-pntr :double) filler)
	`(nnl2.ffi:%full-like ,tensor ,filler-pntr)))
	
(defmacro .max! (tensora tensorb)
  `(nnl2.ffi:%.max! ,tensora ,tensorb))	
	
(defmacro .min! (tensora tensorb)
  `(nnl2.ffi:%.min! ,tensora ,tensorb))
  
(defmacro .max (tensora tensorb)
  `(nnl2.ffi:%.max ,tensora ,tensorb))
  
(defmacro .min (tensora tensorb)
  `(nnl2.ffi:%.min ,tensora ,tensorb)) 
  
(defmacro .abs! (tensor)
  `(nnl2.ffi:%.abs! ,tensor))  
  
(defmacro .abs (tensor)
  `(nnl2.ffi:%.abs ,tensor))    
  
;; (setf (aref vec i) (cffi:mem-aref shape-pointer :int i)

(defun .map! (funct &rest tensors &aux (first-tensor (car tensors)))
  (let ((aggreg-data (mapcar #'nnl2.ffi:get-tensor-data tensors))
		(numel (size first-tensor))
		(type-t (case (dtype first-tensor)
				  (:float64 :double)
				  (:float32 :float)
				  (:int32 :int))))
		
	(loop for i from 0 below 9
		  do (setf
        	   (cffi:mem-aref (car aggreg-data) type-t i) (apply funct (loop for it in aggreg-data 
																			 collect (cffi:mem-aref it type-t i)))))))  
																			 
(defun .map (funct &rest tensors &aux (first-tensor (car tensors)))
  (let* ((aggreg-data (mapcar #'nnl2.ffi:get-tensor-data tensors))
		 (numel (size (car tensors)))
		
		 (type-t (case (dtype first-tensor)
				   (:float64 :double)
				   (:float32 :float)
				   (:int32 :int)))
				  
		 (new-tensor (empty-like first-tensor))
		 (new-tensor-data (nnl2.ffi:get-tensor-data new-tensor)))
		
	(loop for i from 0 below 9
		  do (setf
        	   (cffi:mem-aref new-tensor-data type-t i) (apply funct (loop for it in aggreg-data 
																           collect (cffi:mem-aref it type-t i)))))

	new-tensor))
	
(defun hstack (&rest tensors) (reduce #'nnl2.ffi:%hstack tensors))	
(defun vstack (&rest tensors) (reduce #'nnl2.ffi:%vstack tensors))	

(defmacro .relu! (tensor)
  `(nnl2.ffi:%.relu! ,tensor))
  
(defmacro .relu (tensor)
  `(nnl2.ffi:%.relu ,tensor))  
  
(defmacro .leaky-relu! (tensor &key (alpha 0.01))
  `(nnl2.ffi:%.leaky-relu! ,tensor ,alpha))  
  
(defmacro .leaky-relu (tensor &key (alpha 0.01))
  `(nnl2.ffi:%.leaky-relu ,tensor ,alpha))  
  
(defmacro .sigmoid! (tensor)
  `(nnl2.ffi:%.sigmoid! ,tensor))  
  
(defmacro .sigmoid (tensor)
  `(nnl2.ffi:%.sigmoid ,tensor))  
																				
(declaim (inline gemm))
(declaim (inline gemm!))																			 
																			 