(in-package :nnl2.hli.ts)

(deftype nnl2-tensor () 'sb-sys:system-area-pointer)

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

  (let* ((shape-rank (length shape))
		 (tensor-rank (rank tensor))
	     (tensor-dtype (dtype tensor))
		 (shape (make-shape-pntr (subst -1 '* shape)))
		 (void-ptr (nnl2.ffi:%tref tensor shape shape-rank)))	 
		 
	(if (= shape-rank tensor-rank)	 
	  (case tensor-dtype
	    (:float64 (cffi:mem-ref void-ptr :double))	 
        (:float32 (cffi:mem-ref void-ptr :float))
	    (:int32 (cffi:mem-ref void-ptr :int)))
	  
	  void-ptr)))
	  
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
		
	(loop for i from 0 below numel
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
		
	(loop for i from 0 below numel
		  do (setf
        	   (cffi:mem-aref new-tensor-data type-t i) (apply funct (loop for it in aggreg-data 
																           collect (cffi:mem-aref it type-t i)))))

	new-tensor))

	
(defun hstack (&rest tensors) (reduce #'nnl2.ffi:%hstack tensors))	
(defun vstack (&rest tensors) (reduce #'nnl2.ffi:%vstack tensors))	
(defun concat (axis &rest tensors) (reduce #'(lambda (acc tensor) (nnl2.ffi:%concat acc tensor axis)) tensors))

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
  
(defmacro .tanh! (tensor)
  `(nnl2.ffi:%.tanh! ,tensor))  
  
(defmacro .tanh (tensor)
  `(nnl2.ffi:%.tanh ,tensor))    
  
(defmacro randn (indices &key (dtype nnl2.system:*default-tensor-type*) (from 0.0d0) (to 1.0d0))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    (case dtype
	  (:float64 `(let ((from-pntr (cffi:foreign-alloc :double))
					   (to-pntr (cffi:foreign-alloc :double)))
					   
				   (setf (cffi:mem-ref from-pntr :double) ,from
						 (cffi:mem-ref to-pntr :double) ,to)
						 
				   (nnl2.ffi:%randn ,shape ,rank ,dtype from-pntr to-pntr)))
				   
      (:float32 `(let ((from-pntr (cffi:foreign-alloc :float))
					   (to-pntr (cffi:foreign-alloc :float)))
					   
				   (setf (cffi:mem-ref from-pntr :float) ,from
						 (cffi:mem-ref to-pntr :float) ,to)
						 
				   (nnl2.ffi:%randn ,shape ,rank ,dtype from-pntr to-pntr)))

	  (:int32 `(let ((from-pntr (cffi:foreign-alloc :int))
					 (to-pntr (cffi:foreign-alloc :int)))
					   
				 (setf (cffi:mem-ref from-pntr :int) ,from
					   (cffi:mem-ref to-pntr :int) ,to)
						 
				 (nnl2.ffi:%randn ,shape ,rank ,dtype from-pntr to-pntr))))))

(defun randn-like (tensor &key (from 0.0d0) (to 1.0d0) (dtype (dtype tensor)))
  (case dtype
    (:float64 (let ((from-pntr (cffi:foreign-alloc :double))
			    	(to-pntr (cffi:foreign-alloc :double)))
					   
			    (setf (cffi:mem-ref from-pntr :double) from
					  (cffi:mem-ref to-pntr :double) to)
						 
				(nnl2.ffi:%randn-like tensor from-pntr to-pntr)))
				
	(:float32 (let ((from-pntr (cffi:foreign-alloc :float))
			        (to-pntr (cffi:foreign-alloc :float)))
					   
			    (setf (cffi:mem-ref from-pntr :float) from
					  (cffi:mem-ref to-pntr :float) to)
						 
				(nnl2.ffi:%randn-like tensor from-pntr to-pntr)))
				
	(:int (let ((from-pntr (cffi:foreign-alloc :int))
			    (to-pntr (cffi:foreign-alloc :int)))
					   
		    (setf (cffi:mem-ref from-pntr :int) from
				  (cffi:mem-ref to-pntr :int) to)
						 
			(nnl2.ffi:%randn-like tensor from-pntr to-pntr)))))		
				 
(defmacro xavier (indices &key (dtype nnl2.system:*default-tensor-type*) (in 0) (out 0) (gain 1.0s0) (distribution :normal))
  (progn
    (assert (not (or (zerop in) (minusp in))) nil "Bad `in` was passed to xavier")
	(assert (not (or (zerop out) (minusp out))) nil "Bad `out` was passed to xavier")
  
    (multiple-value-bind (shape rank) (make-shape-pntr indices)
      (case distribution
	    (:normal `(nnl2.ffi:%xavier ,shape ,rank ,dtype ,in ,out ,gain 2.0s0))
	    (:uniform `(nnl2.ffi:%xavier ,shape ,rank ,dtype ,in ,out ,gain 6.0s0))
	    (otherwise (error "Unknown xavier-distribution: ~a%" distribution))))))
		
(defmacro transpose! (tensor)
  `(nnl2.ffi:%transpose! ,tensor))		
  
(defmacro transpose (tensor)
  `(nnl2.ffi:%transpose ,tensor))	  
  
(defun sum (tensor &key (axes #(0)) &aux (dtype (dtype tensor)))
  (let* ((type-t (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (axes-pntr (make-shape-pntr axes))
		 (axes-len (length axes))
		 (out (cffi:foreign-alloc type-t)))
					
	(nnl2.ffi:%sum tensor axes-pntr axes-len out)
				
	(cffi:mem-ref out type-t)))
	
(defun l2-norm (tensor &key (axes #(0)) &aux (dtype (dtype tensor)))
  (let* ((type-t (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (axes-pntr (make-shape-pntr axes))
		 (axes-len (length axes))
		 (out (cffi:foreign-alloc type-t)))
					
	(nnl2.ffi:%l2norm tensor axes-pntr axes-len out)
				
	(cffi:mem-ref out type-t)))
	
(defmacro norm (tensor &key (axes #(0)) (p :l2))
  (case p
    (:l2 `(l2-norm ,tensor :axes ,axes))
	(otherwise (error "Incorrect :p key in norm~%"))))
	
(defmacro copy (tensor)
  `(nnl2.ffi:%copy ,tensor))	
  
(defun .+/incf! (tensor increment)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
	
	(nnl2.ffi:%.+/incf! tensor incf-pntr)))

(defun .+/incf (tensor increment)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
	
	(nnl2.ffi:%.+/incf tensor incf-pntr)))
	
(defun .-/decf! (tensor decrement)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce decrement lisp-dtype))
	
	(nnl2.ffi:%.-/decf! tensor incf-pntr)))	
	
(defun .-/decf (tensor decrement)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce decrement lisp-dtype))
	
	(nnl2.ffi:%.-/decf tensor incf-pntr)))	
	
(defun .*/mulf! (tensor multiplier)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
	
	(nnl2.ffi:%.*/mulf! tensor incf-pntr)))		
	
(defun .*/mulf (tensor multiplier)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
	
	(nnl2.ffi:%.*/mulf tensor incf-pntr)))	

(defun .//divf! (tensor divf)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (divf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divf lisp-dtype))
	
	(nnl2.ffi:%.//divf! tensor divf-pntr)))		

(defun .//divf (tensor divf)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (divf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divf lisp-dtype))
	
	(nnl2.ffi:%.//divf tensor divf-pntr)))	

(defun .^/powf! (tensor powf)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (powf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce powf lisp-dtype))
	
	(nnl2.ffi:%.^/powf! tensor powf-pntr)))

(defun .^/powf (tensor powf)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (powf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce powf lisp-dtype))
	
	(nnl2.ffi:%.^/powf tensor powf-pntr)))	
	
(defun .max/maxf! (tensor maxf)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce maxf lisp-dtype))
	
	(nnl2.ffi:%.max/maxf! tensor maxf-pntr)))
	
(defun .max/maxf (tensor maxf)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce maxf lisp-dtype))
	
	(nnl2.ffi:%.max/maxf tensor maxf-pntr)))	
	
(defun .min/minf! (tensor minf)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (minf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce minf lisp-dtype))
	
	(nnl2.ffi:%.min/minf! tensor minf-pntr)))	
	
(defun .min/minf (tensor minf)
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (minf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce minf lisp-dtype))
	
	(nnl2.ffi:%.min/minf tensor minf-pntr)))	

(defun .+/gnrl (a b)
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.+/incf a b))
		(t (.+ a b))))
		
(defun .+/gnrl! (a b)
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.+/incf! a b))
		(t (+= a b))))		
		
(defun .-/gnrl (a b)
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.-/decf a b))
		(t (.- a b))))

(defun .-/gnrl! (a b)
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.-/decf! a b))
		(t (-= a b))))				
		
(defun .*/gnrl! (a b)
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.*/mulf! a b))
		(t (*= a b))))		

(defun .*/gnrl (a b)
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.*/mulf a b))
		(t (.* a b))))	

(defun .//gnrl! (a b)	
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.*/divf! a b))
		(t (/! a b))))	

(defun .//gnrl (a b)	
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.*/divf a b))
		(t (./ a b))))	

(defun .^/gnrl! (a b)	
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.^/powf! a b))
		(t (^= a b))))	
		
(defun .^/gnrl (a b)	
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.^/powf a b))
		(t (.^ a b))))			
			
(defun .max/gnrl! (a b)	
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.max/maxf! a b))
		(t (.max! a b))))		

(defun .max/gnrl (a b)	
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.max/maxf a b))
		(t (.max a b))))	

(defun .min/gnrl! (a b)	
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.min/minf! a b))
		(t (.min! a b))))		

(defun .min/gnrl (a b)	
  (cond ((and (typep a 'real) (typep b 'nnl2-tensor)) (error "You can't apply a tensor function to a scalar"))
		((and (typep a 'nnl2-tensor) (typep b 'real)) (.min/minf a b))
		(t (.min a b))))			
			
(declaim (inline gemm))
(declaim (inline gemm!))																			 
																			 