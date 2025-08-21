(in-package :nnl2.hli.ts)

(deftype nnl2-tensor () 'sb-sys:system-area-pointer)

(defparameter *nnl2-tensor-types* '((:float64 . double-float) (:float32 . single-float) (:int32 . integer)))

(defun ts-type-to-lisp (tensor-type)
  (case tensor-type (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))	
  
(defun lisp-type-to-ts (lisp-type)
  (cond ((eql lisp-type 'double-float) :float64) ((eql lisp-type 'single-float) :float32) ((eql lisp-type 'integer) :int32))) 

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
		           collect `(when (typep ,var 'nnl2-tensor) (free ,var))))))))

(defmacro tlet* ((&rest bindings) &body body)
  (if (null bindings)
   `(progn ,@body)
    (let* ((binding (first bindings))
           (var (if (consp binding) (car binding) binding))
           (value (if (consp binding) (cadr binding) nil)))
     `(let (,binding)
        (unwind-protect
          (tlet* ,(rest bindings) ,@body)
          (when (typep ,var 'nnl2-tensor) (free ,var)))))))

(defun empty (indices &key (dtype nnl2.system:*default-tensor-type*))
  (declare (speed 3))
  
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    (nnl2.ffi:%empty shape rank dtype))) 
	
(defmacro empty-with-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  `(nnl2.ffi:%empty ,shape-pntr ,rank ,dtype))

(defun zeros (indices &key (dtype nnl2.system:*default-tensor-type*))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    (nnl2.ffi:%zeros shape rank dtype)))	 

(defmacro zeros-with-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  `(nnl2.ffi:%zeros ,shape-pntr ,rank ,dtype))

(defun ones (indices &key (dtype nnl2.system:*default-tensor-type*))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    (nnl2.ffi:%ones shape rank dtype))) 

(defmacro ones-with-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  `(nnl2.ffi:%ones ,shape-pntr ,rank ,dtype))

(defun full (indices &key (dtype nnl2.system:*default-tensor-type*) (filler 0.0d0))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    (let* ((cffi-type (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		   (filler-pntr (cffi:foreign-alloc cffi-type)))
		  
	  (setf (cffi:mem-ref filler-pntr cffi-type) filler)
	  
      (nnl2.ffi:%full shape rank dtype filler-pntr))))
	  
(defun from-flatten (flatten-data indices &key (dtype nnl2.system:*default-tensor-type*))
  (multiple-value-bind (shape rank) (make-shape-pntr indices)
    (let* ((total-elems (length flatten-data))
		   (cffi-type (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		   (lisp-type (ts-type-to-lisp dtype))
		   (data-pntr (cffi:foreign-alloc cffi-type :count total-elems)))
	
      (cond
	    ((listp flatten-data)
	       (dotimes (i total-elems)	  
	         (setf (cffi:mem-aref data-pntr cffi-type i) (coerce (nth i flatten-data) lisp-type))))
			 
		((vectorp flatten-data)
		   (dotimes (i total-elems)	  
	         (setf (cffi:mem-aref data-pntr cffi-type i) (coerce (aref flatten-data i) lisp-type)))))
	  
      (nnl2.ffi:%make-tensor-from-flatten data-pntr total-elems shape rank dtype))))

(defun reduce-list-shape (lst)
  (if (atom lst)
    nil
	(cons (length lst) (reduce-list-shape (car lst)))))
	
(defun flatten (lst)
  (loop for elem in lst
        if (listp elem) append (flatten elem)
        else collect elem))	

(defun make-tensor (data &key (dtype nnl2.system:*default-tensor-type*))
  (declare (optimize (speed 3) (safety 0)))
  
  (etypecase data
    (array
	  (let* ((data-shape (array-dimensions data))
			 (flat-data (make-array (array-total-size data) :element-type (array-element-type data) :displaced-to data)))
		 
	    (from-flatten flat-data data-shape :dtype dtype)))
		
	(list
	  (let* ((data-shape (reduce-list-shape data))
			 (flat-data (coerce (flatten data) 'vector)))

	    (from-flatten flat-data data-shape)))))
  
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

(defun .exp! (tensor)
  (nnl2.ffi:%.exp! tensor))

(defun .exp (tensor)
  (nnl2.ffi:%.exp tensor))  
  
(defun .log! (tensor)
  (nnl2.ffi:%.log! tensor))  
  
(defun .log (tensor)
  (nnl2.ffi:%.log tensor))    
    
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
		 (void-ptr (nnl2.ffi:%tref-getter tensor shape shape-rank)))	 
		 
	(if (= shape-rank tensor-rank)	 
	  (case tensor-dtype
	    (:float64 (cffi:mem-ref void-ptr :double))	 
        (:float32 (cffi:mem-ref void-ptr :float))
	    (:int32 (cffi:mem-ref void-ptr :int)))
	  
	  void-ptr)))
	  
(defun (setf tref) (change-to tensor &rest shape)
  (let* ((shape-rank (length shape))
		 (shape (make-shape-pntr (subst -1 '* shape)))
	     (tensor-dtype (dtype tensor))
		 (is-tensor-p (typep change-to 'nnl2-tensor)))
		 
	(cond
	  (is-tensor-p (nnl2.ffi:%tref-setter tensor shape shape-rank change-to t))
	  
	  (t	 
		(case tensor-dtype
		  (:float64 
			(let ((changer (cffi:foreign-alloc :double)))
			  (setf (cffi:mem-ref changer :double) change-to)
			  (nnl2.ffi:%tref-setter tensor shape shape-rank changer nil)))
			  
		  (:float32
			(let ((changer (cffi:foreign-alloc :float)))
			  (setf (cffi:mem-ref changer :float) change-to)
			  (nnl2.ffi:%tref-setter tensor shape shape-rank changer nil)))

		  (:int32
			(let ((changer (cffi:foreign-alloc :int)))
			  (setf (cffi:mem-ref changer :int) change-to)
			  (nnl2.ffi:%tref-setter tensor shape shape-rank changer nil))))))))
		  
(defmacro scale! (tensor multiplier)
  `(nnl2.ffi:%scale! ,tensor (float ,multiplier)))

(defmacro scale (tensor multiplier)
  `(nnl2.ffi:%scale ,tensor (float ,multiplier)))  
  
(defun empty-like (tensor)
  (nnl2.ffi:%empty-like tensor))  
  
(defun zeros-like (tensor)
  (nnl2.ffi:%zeros-like tensor))  
	
(defun ones-like (tensor)
  (nnl2.ffi:%ones-like tensor))  
		
(defun full-like (tensor &key (filler 0.0d0))
  (let* ((filler-type (type-of filler))
		 (keyword-type (cond ((eql filler-type 'double-float) :double) ((eql filler-type 'single-float) :float) ((eql filler-type 'integer) :int)))
		 (filler-pntr (cffi:foreign-alloc keyword-type)))
		 
    (setf (cffi:mem-ref filler-pntr keyword-type) filler)
	  (nnl2.ffi:%full-like tensor filler-pntr)))
  
(defun .abs! (tensor)
  (nnl2.ffi:%.abs! tensor))  
  
(defun .abs (tensor)
  (nnl2.ffi:%.abs tensor))    

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
	
(defun /map! (funct &rest tensors &aux (first-tensor (car tensors)))
  (let ((first-shape (aref (shape first-tensor :as :vector) 0))
		(type-t (case (dtype first-tensor)
				  (:float64 :double)
				  (:float32 :float)
				  (:int32 :int))))
		
	(loop for i from 0 below first-shape
		  do (setf
        	   (tref first-tensor i) (apply funct (loop for it in tensors
														collect (tref it i)))))))
														
(defun /map (funct &rest tensors &aux (first-tensor (car tensors)))
  (let ((first-shape (aref (shape first-tensor :as :vector) 0))
		(type-t (case (dtype first-tensor)
				  (:float64 :double)
				  (:float32 :float)
				  (:int32 :int)))
		
		(new-tensor (empty-like first-tensor)))
		
	(loop for i from 0 below first-shape
		  do (setf
        	   (tref new-tensor i) (apply funct (loop for it in tensors
														collect (tref it i)))))

	new-tensor))

(defun hstack (&rest tensors) (reduce #'nnl2.ffi:%hstack tensors))	
(defun vstack (&rest tensors) (reduce #'nnl2.ffi:%vstack tensors))	
(defun concat (axis &rest tensors) (reduce #'(lambda (acc tensor) (nnl2.ffi:%concat acc tensor axis)) tensors))

(defun .relu! (tensor)
  (nnl2.ffi:%.relu! tensor))
  
(defun .relu (tensor)
  (nnl2.ffi:%.relu tensor))  
  
(defun .leaky-relu! (tensor &key (alpha 0.01))
  (nnl2.ffi:%.leaky-relu! tensor alpha))  
  
(defun .leaky-relu (tensor &key (alpha 0.01))
  (nnl2.ffi:%.leaky-relu tensor alpha))  
  
(defun .sigmoid! (tensor)
  (nnl2.ffi:%.sigmoid! tensor))  
  
(defun .sigmoid (tensor)
  (nnl2.ffi:%.sigmoid tensor))  
  
(defun .tanh! (tensor)
  (nnl2.ffi:%.tanh! tensor))  
  
(defun .tanh (tensor)
  (nnl2.ffi:%.tanh tensor))    
  
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

(defun axpy/axpf! (summand sumend alpha)
  (let* ((dtype (dtype summand))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (sumend-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref sumend-pntr cffi-dtype) (coerce sumend lisp-dtype))
	
	(nnl2.ffi:%axpy/axpf! summand sumend-pntr alpha)))	
	
(defun axpy/axpf (summand sumend alpha)
  (let* ((dtype (dtype summand))
		 (cffi-dtype (case dtype (:float64 :double) (:float32 :float) (:int32 :int)))
		 (lisp-dtype (case dtype (:float64 'double-float) (:float32 'single-float) (:int32 'integer)))
		 (sumend-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref sumend-pntr cffi-dtype) (coerce sumend lisp-dtype))
	
	(nnl2.ffi:%axpy/axpf summand sumend-pntr alpha)))		
	
(defmacro .+/broadcasting! (summand sumend)
  `(nnl2.ffi:%.+/broadcasting! ,summand ,sumend))
  
(defmacro .+/broadcasting (summand sumend)
  `(nnl2.ffi:%.+/broadcasting ,summand ,sumend))

(defmacro .-/broadcasting! (minuend subtrahend)
  `(nnl2.ffi:%.-/broadcasting! ,minuend ,subtrahend))
  
(defmacro .-/broadcasting (minuend subtrahend)
  `(nnl2.ffi:%.-/broadcasting ,minuend ,subtrahend))
  
(defmacro .*/broadcasting! (multiplicand multiplier)
  `(nnl2.ffi:%.*/broadcasting! ,multiplicand ,multiplier))  
  
(defmacro .*/broadcasting (multiplicand multiplier)
  `(nnl2.ffi:%.*/broadcasting ,multiplicand ,multiplier))   

(defmacro .//broadcasting! (dividend diviser)
  `(nnl2.ffi:%.//broadcasting! ,dividend ,diviser))    
  
(defmacro .//broadcasting (dividend diviser)
  `(nnl2.ffi:%.//broadcasting ,dividend ,diviser))  
  
(defmacro .^/broadcasting! (base exponent)
  `(nnl2.ffi:%.^/broadcasting! ,base ,exponent))

(defmacro .^/broadcasting (base exponent)
  `(nnl2.ffi:%.^/broadcasting ,base ,exponent))  
  
(defmacro .max/broadcasting! (tensora tensorb)
  `(nnl2.ffi:%.max/broadcasting! ,tensora ,tensorb)) 
  
(defmacro .max/broadcasting (tensora tensorb)
  `(nnl2.ffi:%.max/broadcasting ,tensora ,tensorb)) 

(defmacro .min/broadcasting! (tensora tensorb)
  `(nnl2.ffi:%.min/broadcasting! ,tensora ,tensorb))   
  
(defmacro .min/broadcasting (tensora tensorb)
  `(nnl2.ffi:%.min/broadcasting ,tensora ,tensorb))     
  
(defmacro axpy/broadcasting! (summand sumend alpha)
  `(nnl2.ffi:%axpy/broadcasting! ,summand ,sumend ,alpha))  
  
(defmacro axpy/broadcasting (summand sumend alpha)
  `(nnl2.ffi:%axpy/broadcasting ,summand ,sumend ,alpha))    

(defun axpy (a b &key (alpha 1.0))
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (+ a (* b alpha))
          (error "You can't apply a tensor function to a scalar"))
        (axpy/axpf a b alpha))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%axpy a b alpha)
          (if (> (rank a) (rank b))
            (axpy/broadcasting a b alpha)
            (axpy/broadcasting b a alpha)))))))  
			
(defun axpy! (a b &key (alpha 1.0))
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (setq a (+ a (* b alpha)))
          (error "You can't apply a tensor function to a scalar"))
        (axpy/axpf! a b alpha))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%axpy! a b alpha)
          (if (> (rank a) (rank b))
            (axpy/broadcasting! a b alpha)
            (axpy/broadcasting! b a alpha)))))))  			

(defun .+ (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (+ a b)
          (error "You can't apply a tensor function to a scalar"))
        (.+/incf a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%+ a b)
          (if (> (rank a) (rank b))
            (.+/broadcasting a b)
            (.+/broadcasting b a)))))))  

(defun += (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (incf a b)
          (error "You can't apply a tensor function to a scalar"))
        (.+/incf! a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%+= a b)
          (if (> (rank a) (rank b))
            (.+/broadcasting! a b)
            (.+/broadcasting! b a)))))))  

(defun .- (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (- a b)
          (error "You can't apply a tensor function to a scalar"))
        (.-/decf a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%- a b)
          (if (> (rank a) (rank b))
            (.-/broadcasting a b)
            (.-/broadcasting b a))))))) 

(defun -= (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (decf a b)
          (error "You can't apply a tensor function to a scalar"))
        (.-/decf! a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%-= a b)
          (if (> (rank a) (rank b))
            (.-/broadcasting! a b)
            (.-/broadcasting! b a)))))))	
			
(defun .* (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (* a b)
          (error "You can't apply a tensor function to a scalar"))
        (.*/mulf a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%* a b)
          (if (> (rank a) (rank b))
            (.*/broadcasting a b)
            (.*/broadcasting b a)))))))	 	
		
(defun *= (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (setq a (* a b))
          (error "You can't apply a tensor function to a scalar"))
        (.*/mulf! a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%*= a b)
          (if (> (rank a) (rank b))
            (.*/broadcasting! a b)
            (.*/broadcasting! b a)))))))
			
(defun ./ (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (/ a b)
          (error "You can't apply a tensor function to a scalar"))
        (.//divf a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%/ a b)
          (if (> (rank a) (rank b))
            (.//broadcasting a b)
            (.//broadcasting b a)))))))				
		
(defun /! (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (setq a (/ a b))
          (error "You can't apply a tensor function to a scalar"))
        (.//divf! a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%/= a b)
          (if (> (rank a) (rank b))
            (.//broadcasting! a b)
            (.//broadcasting! b a)))))))	  			
		
(defun .^ (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (setq a (expt a b))
          (error "You can't apply a tensor function to a scalar"))
        (.^/powf a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%.^ a b)
          (if (> (rank a) (rank b))
            (.^/broadcasting a b)
            (.^/broadcasting b a)))))))	  

(defun ^= (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (setq a (expt a b))
          (error "You can't apply a tensor function to a scalar"))
        (.^/powf! a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%^= a b)
          (if (> (rank a) (rank b))
            (.^/broadcasting! a b)
            (.^/broadcasting! b a)))))))	
			
(defun .max (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (max a b)
          (error "You can't apply a tensor function to a scalar"))
        (.max/maxf a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%.max a b)
          (if (> (rank a) (rank b))
            (.max/broadcasting a b)
            (.max/broadcasting b a))))))) 

(defun .max! (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (setq a (max a b))
          (error "You can't apply a tensor function to a scalar"))
        (.max/maxf! a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%.max! a b)
          (if (> (rank a) (rank b))
            (.max/broadcasting! a b)
            (.max/broadcasting! b a)))))))	  				

(defun .min (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (min a b)
          (error "You can't apply a tensor function to a scalar"))
        (.min/minf a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%.min a b)
          (if (> (rank a) (rank b))
            (.min/broadcasting a b)
            (.min/broadcasting b a)))))))	

(defun .min! (a b)
  (let ((scalar-a-p (typep a 'real)))
    (if (or scalar-a-p (typep b 'real))
      (if scalar-a-p
        (if (typep b 'real)
          (setq a (min a b))
          (error "You can't apply a tensor function to a scalar"))
        (.min/minf! a b))
      (let ((shapea (shape a :as :list))
            (shapeb (shape b :as :list)))
        (if (equal shapea shapeb)
		  (nnl2.ffi:%.min! a b)
          (if (> (rank a) (rank b))
            (.min/broadcasting! a b)
            (.min/broadcasting! b a)))))))	  		

(declaim (inline gemm))
(declaim (inline gemm!))																			 
																			 