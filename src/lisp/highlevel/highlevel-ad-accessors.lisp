(in-package :nnl2.hli.ad)

(deftype nnl2-ad-tensor () 
  #+sbcl      'sb-sys:system-area-pointer
  #+clisp     'fi:foreign-data
  #+ccl       'ccl:macptr
  #+ecl       'si:foreign-data
  #+abcl      'system:foreign-pointer
  #+lispworks 'fli:pointer
  #+allegro   'excl:foreign-pointer)  

(defun ad-get-shape-as-list (ad-tensor rank)
  "Gets the shape of AD tensor as a list"
  (loop with rank-t = (if rank rank (rank ad-tensor))
        with shape-pointer = (nnl2.ffi:%ad-shape ad-tensor)
        for i from 0 below rank-t
        collect (cffi:mem-aref shape-pointer :int i)))

(defun ad-get-shape-as-vector (ad-tensor rank)
  "Gets the shape of AD tensor as a vector"
  (let* ((rank-t (if rank rank (rank ad-tensor)))
         (vec (make-array rank-t))
         (shape-pointer (nnl2.ffi:%ad-shape ad-tensor)))
		 
    (dotimes (i rank-t)
      (setf (aref vec i) (cffi:mem-aref shape-pointer :int i)))
	  
    vec))

(defun rank (ad-tensor)
  "Gets rank of AD tensor"
  (nnl2.ffi:%ad-rank ad-tensor))

(defun dtype (ad-tensor &key (from :data))
  (ecase from
    (:data (nnl2.ffi:%ad-dtype-as-data ad-tensor))
	(:grad (nnl2.ffi:%ad-dtype-as-grad ad-tensor))))

(defun int-dtype (ad-tensor &key (from :data))
  (ecase from
    (:data (nnl2.ffi:%ad-dtype-as-data-int ad-tensor))
	(:grad (nnl2.ffi:%ad-dtype-as-grad-int ad-tensor))))

(declaim (ftype (function (nnl2-ad-tensor (integer 0 *)) list) ad-get-shape-as-list)
         (ftype (function (nnl2-ad-tensor (integer 0 *)) vector) ad-get-shape-as-vector)
         (ftype (function (nnl2-ad-tensor) integer) ad-rank))

(defun shape (ad-tensor &key (as :vector))
  "Function for getting the shape of an AD tensor"
  (let ((rank (rank ad-tensor)))
    (case as
      (:list    (ad-get-shape-as-list ad-tensor rank))
      (:vector  (ad-get-shape-as-vector ad-tensor rank))
      (:pointer (nnl2.ffi:%ad-shape ad-tensor))
      (otherwise (error "Unknown type: ~a~%" as)))))

(defun shapes-equal-p (ad-tensor-a ad-tensor-b)
  "Compares shapes of two AD tensors"
  (let ((shape-a (shape ad-tensor-a :as :vector))
        (shape-b (shape ad-tensor-b :as :vector)))
		
    (equalp shape-a shape-b)))

(cffi:defcfun ("nnl2_ad_get_num_roots" num-roots) :int
  (ad-tensor :pointer))
  
(defun get-roots-as-list (roots-pointer num-roots)
  (loop for i from 0 below num-roots
		collect (cffi:mem-aref roots-pointer :pointer i)))
  
(defun roots (ad-tensor &key (as :list))
  (ecase as 
    (:pointer (nnl2.ffi:%ad-roots ad-tensor))
	(:list (get-roots-as-list (nnl2.ffi:%ad-roots ad-tensor) (num-roots ad-tensor)))))
	
(defun (setf roots) (ad-tensors-list self)
  (let* ((new-len (length ad-tensors-list)) 
         (tensors-pool (cffi:foreign-alloc :pointer :count new-len)))
		 
	(dotimes (i new-len)
	  (setf (cffi:mem-aref tensors-pool :pointer i) (nth i ad-tensors-list)))
	  
	(nnl2.ffi:%ad-roots-setter self tensors-pool new-len)))

(defun higher-rank-tensor (a b)
  "Returns a pair (higher lower) depending on the rank"
  (nnl2.hli:fastcall (if (> (rank a) (rank b))
					   (values a b)
					   (values b a))))  

(defmacro with-tensor-dispatch ((a b) tensor-case same-shape-case broadcast-case)
  "Universal dispatcher for tensor operations"
  
  (let ((a-sym (gensym "A"))
        (b-sym (gensym "B")))
		
    `(let ((,a-sym ,a)
           (,b-sym ,b))
   
       (cond
	     ((typep ,b-sym 'real) ,tensor-case)
		 ((typep ,a-sym 'real) (error "You can't apply a tensor function to a scalar"))
         ((shapes-equal-p ,a-sym ,b-sym) ,same-shape-case)
		 (t (multiple-value-bind (higher lower) (higher-rank-tensor ,a-sym ,b-sym) ,broadcast-case))))))

(defun empty (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  (nnl2.ffi:%ad-empty shape rank dtype requires-grad name))))

(defun zeros (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  (nnl2.ffi:%ad-zeros shape rank dtype requires-grad name))))
	  
(defun ones (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  (nnl2.ffi:%ad-ones shape rank dtype requires-grad name))))
	  	  
(defun full (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name "") (filler 1.0))
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
		   
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  
	  (let* ((cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype))
	         (lisp-type (nnl2.hli.ts:type/nnl2->lisp dtype))
			 (filler-pntr (cffi:foreign-alloc cffi-type)))
			 
	    (setf (cffi:mem-ref filler-pntr cffi-type) (coerce filler lisp-type))
		
		(nnl2.ffi:%ad-full shape rank dtype requires-grad name filler-pntr)))))
	  
(defun %internal-randn (indices dtype requires-grad name from to)
  (nnl2.hli:fastcall
    (let* ((cffi-type    (nnl2.hli.ts:type/nnl2->cffi dtype))
		   (lisp-type    (nnl2.hli.ts:type/nnl2->lisp dtype))
		   (from-pntr    (cffi:foreign-alloc cffi-type))
		   (to-pntr      (cffi:foreign-alloc cffi-type))
		   (coerced-to   (coerce to lisp-type))
		   (coerced-from (coerce from lisp-type)))
		   
	 (setf (cffi:mem-ref from-pntr cffi-type) coerced-from
		   (cffi:mem-ref to-pntr cffi-type) coerced-to)
		   
     (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	   (declare (type integer rank))
	   (nnl2.ffi:%ad-randn shape rank dtype requires-grad name from-pntr to-pntr)))))
	
(defun rand (indices &key (from 0) (to 1) (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  (%internal-randn indices dtype requires-grad name from to))
	
(defun randn (indices &key (from -1) (to 1) (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  (%internal-randn indices dtype requires-grad name from to))
	 
(defun xavier (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name "") (in 0) (out 0) (gain 1.0s0) (distribution :normal))
  (assert (not (or (zerop in) (minusp in))) nil "Bad `in` was passed to xavier (AD)")
  (assert (not (or (zerop out) (minusp out))) nil "Bad `out` was passed to xavier (AD)")
  
  (nnl2.hli:fastcall
    (let ((dist (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))))
      (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	    (declare (type integer rank))
        (nnl2.ffi:%ad-xavier shape rank dtype requires-grad name in out gain dist))))) 

(defun make-tensor (data &key (dtype nnl2.system:*default-tensor-type*))
  (let* ((shape (array-dimensions data))
		 (ts-tensor (nnl2.hli.ts:make-tensor data :dtype dtype :shape-hint shape))
		 (ad-tensor (empty shape :dtype dtype)))
		 
	(nnl2.ffi:%data-pntr-share-setter ad-tensor ts-tensor)
	
	ad-tensor))
	
(defun from-flatten (flatten-data indices &key (dtype nnl2.system:*default-tensor-type*))
  (let ((ad-tensor (empty indices :dtype dtype))
		(ts-tensor (nnl2.hli.ts:from-flatten flatten-data indices :dtype dtype)))
		
	(nnl2.ffi:%data-pntr-share-setter ad-tensor ts-tensor)
	
	ad-tensor))
	
(defun transposition! (ad-tensor &key (track-graph t))
  (nnl2.ffi:%ad-transposition-inplace ad-tensor track-graph))
  
(defun transpose! (ad-tensor &key (track-graph t) force)
  (nnl2.ffi:%ad-transpose-inplace ad-tensor track-graph force))  
  
(cffi:defcfun ("nnl2_ad_zero_grad" zero-grad!) :void
  (ad-tensor :pointer))	  
	  
(cffi:defcfun ("nnl2_ad_get_data" data) :pointer
  (ad-tensor :pointer))  
  
(cffi:defcfun ("nnl2_ad_get_leaf" is-leaf) :bool
  (ad-tensor :pointer)) 

(cffi:defcfun ("nnl2_ad_get_requires_grad" requires-grad) :bool
  (ad-tensor :pointer))   
  
(cffi:defcfun ("nnl2_ad_get_grad" grad) :pointer
  (ad-tensor :pointer))
  
(cffi:defcfun ("nnl2_free_ad_tensor" free) :void
  (ad-tensor :pointer))  
  
(defun bp (ad-tensor &key retain-graph)
  (nnl2.ffi:%backpropagation ad-tensor retain-graph))  

(defun backpropagation (ad-tensor &key retain-graph)
  (nnl2.ffi:%backpropagation ad-tensor retain-graph))  
  
(defun bptt (ad-tensor &key retain-graph)
  (nnl2.ffi:%bptt ad-tensor retain-graph))    
  
(defun backpropagation-through-time (ad-tensor &key retain-graph)
  (nnl2.ffi:%bptt ad-tensor retain-graph))    
  
(defmacro tlet ((&rest bindings) &body body)
  (let ((vars (mapcar #'car bindings)))
    `(let ,bindings
	   (unwind-protect
	       (progn ,@body)
		   
		 (progn 
		   ,@(loop for var in vars
		           collect `(when (typep ,var 'nnl2-ad-tensor) (free ,var))))))))  
  
(defmacro tlet* ((&rest bindings) &body body)
  "The eigenform of let* for nnl2 ad tensors"
  
  (if (null bindings)
   `(progn ,@body)
    (let* ((binding (first bindings))
           (var (if (consp binding) (car binding) binding))
           (value (if (consp binding) (cadr binding) nil)))
     `(let (,binding)
        (unwind-protect
          (tlet* ,(rest bindings) ,@body)
          (when (typep ,var 'nnl2-ad-tensor) (free ,var)))))))  
  
(defun print-data (ad-tensor)
  (nnl2.hli.ts:print-tensor (data ad-tensor)))
  
(defun print-grad (ad-tensor)
  (nnl2.hli.ts:print-tensor (grad ad-tensor)))
  
(defun step-ts (ad-tensor &key (lr 1.0))
  (nnl2.ffi:%ad-step ad-tensor (coerce lr 'single-float)))
  
(defun step! (ad-tensor &key (lr 1.0))
  (nnl2.ffi:%ad-step! ad-tensor (coerce lr 'single-float)))
  
(defun .+/ad/incf! (tensor increment mode track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
		 (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
		 (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
	
	(nnl2.ffi:%ad-add-correspondence tensor incf-pntr mode track-graph)))  
  
(defun +=/ad/incf! (tensor increment track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (incf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
    
    (nnl2.ffi:%ad-add-incf-inplace tensor incf-pntr track-graph)))  

(defun .*/ad/mulf! (tensor multiplier mode track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (mulf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref mulf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
    
    (nnl2.ffi:%ad-mul-correspondence tensor mulf-pntr mode track-graph)))

(defun *=/ad/mulf! (tensor multiplier track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (mulf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref mulf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
    
    (nnl2.ffi:%ad-mul-mulf-inplace tensor mulf-pntr track-graph)))	
  
(defun .-/ad/decf! (tensor decrement mode track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (decf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref decf-pntr cffi-dtype) (coerce decrement lisp-dtype))
    
    (nnl2.ffi:%ad-sub-correspondence tensor decf-pntr mode track-graph)))

(defun -=/ad/decf! (tensor decrement track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (decf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref decf-pntr cffi-dtype) (coerce decrement lisp-dtype))
    
    (nnl2.ffi:%ad-sub-decf-inplace tensor decf-pntr track-graph)))
	
(defun ./ad/divf! (tensor divisor mode track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (divf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divisor lisp-dtype))
    
    (nnl2.ffi:%ad-div-correspondence tensor divf-pntr mode track-graph)))	
	
(defun /!/ad/divf! (tensor divisor track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (divf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divisor lisp-dtype))
    
    (nnl2.ffi:%ad-div-divf-inplace tensor divf-pntr track-graph)))
	
(defun .^/ad/powf! (tensor exponent mode track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (powf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce exponent lisp-dtype))
    
    (nnl2.ffi:%ad-pow-correspondence tensor powf-pntr mode track-graph)))	

(defun ^=/ad/powf! (tensor exponent track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (powf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce exponent lisp-dtype))
    
    (nnl2.ffi:%ad-pow-powf-inplace tensor powf-pntr track-graph)))
	
(defun .min/ad/minf! (tensor value mode track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (minf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-min-correspondence tensor minf-pntr mode track-graph)))

(defun .min!/ad/minf! (tensor value track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (minf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-min-minf-inplace tensor minf-pntr track-graph)))	

(defun .max/ad/maxf! (tensor value mode track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-max-correspondence tensor maxf-pntr mode track-graph)))
	
(defun .max!/ad/maxf! (tensor value track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-max-maxf-inplace tensor maxf-pntr track-graph)))
	
(defun axpy/ad/axpf! (tensor other alpha mode track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-axpf tensor other-pntr (coerce alpha 'single-float) mode track-graph)))	
	
(defun axpy!/ad/axpf! (tensor other alpha track-graph)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-axpf-inplace tensor other-pntr (coerce alpha 'single-float) track-graph)))	
	
(defmacro with-notrack (&body body)
  `(progn
     (let ((nnl2.system:*ad-default-track-graph* nil))
       ,@body)))
	
(cffi:defcfun ("nnl2_ad_neg_inplace" .neg!) :void
  (ad-tensor :pointer))		
  
(defun += (a b &key (track-graph t))
  "In-place addition"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (+=/ad/incf! a b track-graph)	
      (nnl2.ffi:%ad-+= a b track-graph)
      (nnl2.ffi:%ad-add-broadcasting-inplace a b track-graph))))  
	  
(defun -= (a b &key (track-graph t))
  "In-place subtraction"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (-=/ad/decf! a b track-graph)
      (nnl2.ffi:%ad--= a b track-graph)
      (nnl2.ffi:%ad-sub-broadcasting-inplace a b track-graph))))	  
  
(defun *= (a b &key (track-graph t))
  "In-place multiplication"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (*=/ad/mulf! a b track-graph)
      (nnl2.ffi:%ad-*= a b track-graph)
      (nnl2.ffi:%ad-mul-broadcasting-inplace a b track-graph))))
  
(defun /! (a b &key (track-graph t))
  "In-place division"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (/!/ad/divf! a b track-graph)
      (nnl2.ffi:%ad-/! a b track-graph)
      (nnl2.ffi:%ad-div-broadcasting-inplace a b track-graph))))

(defun ^= (a b &key (track-graph t))
  "In-place pow"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (^=/ad/powf! a b track-graph)
      (nnl2.ffi:%ad-^= a b track-graph)
      (nnl2.ffi:%ad-pow-broadcasting-inplace a b track-graph))))	
	  
(defun .abs! (ad-tensor &key (track-graph t))
  (nnl2.ffi:%ad-.abs! ad-tensor track-graph))		  
	  
(defun .min! (a b &key (track-graph t))
  "In-place min"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.min!/ad/minf! a b track-graph)
      (nnl2.ffi:%ad-.min! a b track-graph)
      (nnl2.ffi:%ad-min-broadcasting-inplace a b track-graph))))	  
	  
(defun .max! (a b &key (track-graph t))
  "In-place max"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.max!/ad/maxf! a b track-graph)
      (nnl2.ffi:%ad-.max! a b track-graph)
      (nnl2.ffi:%ad-max-broadcasting-inplace a b track-graph))))
	  
(defun axpy! (a b &key (alpha 1.0) (track-graph t))
  "In-place a+b*c"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:axpy!/ad/axpf! a b alpha track-graph)
      (nnl2.ffi:%ad-axpy! a b alpha track-graph)
      (nnl2.ffi:%ad-axpy-broadcasting-inplace a b alpha track-graph))))	  
	  
(defun scale! (a b &key (track-graph t))
  (nnl2.ffi:%ad-scale! a (coerce b 'single-float) track-graph))
  
(defun .exp! (ad-tensor &key (track-graph t))
  (nnl2.ffi:%ad-.exp! ad-tensor track-graph))  

(defun .log! (ad-tensor &key (track-graph t))
  (nnl2.ffi:%ad-.log! ad-tensor track-graph))    
  
(defun .relu! (ad-tensor &key (track-graph t))
  (nnl2.ffi:%ad-.relu! ad-tensor track-graph))  	 

(defun .leaky-relu! (ad-tensor &key (alpha 0.01) (track-graph t))
  (nnl2.ffi:%ad-.leaky-relu! ad-tensor alpha track-graph))  	
  
(defun .sigmoid! (ad-tensor &key (approx t) (track-graph t))
  (nnl2.ffi:%ad-.sigmoid! ad-tensor approx track-graph))  
  
(defun .tanh! (ad-tensor &key (approx t) (track-graph t))
  (nnl2.ffi:%ad-.tanh! ad-tensor approx track-graph))    
  
(defun .sqrt! (tensor &key (track-graph t))
  (nnl2.ffi:%ad-sqrt-inplace tensor track-graph))
  
(defun copy (tensor &key (dtype (dtype tensor)))
  (nnl2.ffi:%ad-copy tensor dtype))   
  
(cffi:defcfun ("nnl2_ad_empty_like" empty-like) :pointer
  (ad-tensor :pointer))    
  
(cffi:defcfun ("nnl2_ad_zeros_like" zeros-like) :pointer
  (ad-tensor :pointer))  
  
(cffi:defcfun ("nnl2_ad_ones_like" ones-like) :pointer
  (ad-tensor :pointer))    

(cffi:defcfun ("nnl2_ad_rand_like" rand-like) :pointer
  (ad-tensor :pointer))    

(cffi:defcfun ("nnl2_ad_randn_like" randn-like) :pointer
  (ad-tensor :pointer))    
  
(defun full-like (ad-tensor &key (filler 0))
  (let* ((dtype (dtype ad-tensor))
		 (cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype))
		 (lisp-type (nnl2.hli.ts:type/nnl2->lisp dtype))
		 (filler-pntr (cffi:foreign-alloc cffi-type)))
		 
	(setf (cffi:mem-ref filler-pntr cffi-type) (coerce filler lisp-type))
	
	(let ((result (nnl2.ffi:%ad-full-like ad-tensor filler-pntr)))
	  (cffi:foreign-free filler-pntr)
	  result)))

(defun xavier-like (ad-tensor &key in out (gain 1.0s0) (distribution :normal))
  (assert (and in out gain distribution) nil "Incorrect keys was passed in xavier-like")
  (nnl2.ffi:%ad-xavier-like ad-tensor in out gain (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))))
  
(cffi:defcfun ("nnl2_ad_ncast" ncast) :pointer
  (ad-tensor :pointer)
  (cast-to nnl2.ffi:tensor-type)) 

(cffi:defcfun ("nnl2_ad_detach_inplace" detach!) :void
  (ad-tensor :pointer))  

(cffi:defcfun ("nnl2_ad_detach" detach) :pointer
  (ad-tensor :pointer))  
  
(in-package :nnl2.hli.ad.r)

(defun .+ (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise addition"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.+/ad/incf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-.+ a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-add-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))
  
(defun .* (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise multiplication"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.*/ad/mulf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-.* a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-mul-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))	  
	  
(defun gemm (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-gemm a b nnl2.ffi:ad-reverse-mode track-graph))

(defun .- (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise subtraction"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.-/ad/decf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-.- a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-sub-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))
	  
(defun ./ (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise division"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:./ad/divf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-./ a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-div-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))
	
(defun .^ (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise pow"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.^/ad/powf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-.^ a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-pow-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))    
	  
(defun .abs (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-.abs ad-tensor nnl2.ffi:ad-reverse-mode track-graph))	  
	
(defun .min (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise min"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.min/ad/minf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-.min a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-min-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))	  	  
	
(defun .max (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise max"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.max/ad/maxf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-.max a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-max-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))	  	  
	
(defun scale (a b &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-scale a (coerce b 'single-float) save-type nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .log (ad-tensor &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-.log ad-tensor save-type nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .exp (ad-tensor &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-.exp ad-tensor save-type nnl2.ffi:ad-reverse-mode  track-graph))  
  
(defun axpy (a b &key (alpha 1.0) (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise a+b*c"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:axpy/ad/axpf! a b alpha nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-axpy a b alpha nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-axpy-broadcasting a b alpha nnl2.ffi:ad-reverse-mode track-graph))))

(defun .relu (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-.relu ad-tensor nnl2.ffi:ad-reverse-mode track-graph)) 

(defun .leaky-relu (ad-tensor &key save-type (alpha 0.01) (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-.leaky-relu ad-tensor alpha save-type nnl2.ffi:ad-reverse-mode track-graph)) 
    
(defun .sigmoid (ad-tensor &key (approx t) (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-.sigmoid ad-tensor approx nnl2.ffi:ad-reverse-mode track-graph))  
  	
(defun .tanh (ad-tensor &key (approx t) (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-.tanh ad-tensor approx nnl2.ffi:ad-reverse-mode track-graph)) 	
	
(defun .neg (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%.neg ad-tensor nnl2.ffi:ad-reverse-mode track-graph))
	
(defun transposition (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-transposition ad-tensor nnl2.ffi:ad-reverse-mode track-graph))
  
(defun transpose (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*) force)
  (nnl2.ffi:%ad-transpose ad-tensor nnl2.ffi:ad-reverse-mode track-graph force))    

(defun reshape (tensor new-shape &key force (track-graph nnl2.system:*ad-default-track-graph*))
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr new-shape)
    (nnl2.ffi:%ad-reshape tensor shape rank force nnl2.ffi:ad-reverse-mode track-graph)))

(defun reinterpret (tensor new-shape &key force (track-graph nnl2.system:*ad-default-track-graph*))
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr new-shape)
    (nnl2.ffi:%ad-reinterpret tensor shape rank force nnl2.ffi:ad-reverse-mode track-graph)))	
	
(defun .sqrt (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-sqrt tensor nnl2.ffi:ad-reverse-mode track-graph))
  
(defun slice (tensor &key from to (track-graph nnl2.system:*ad-default-track-graph*))
  (let* ((tensor-shape (nnl2.hli.ad:shape tensor :as :vector))
         (from (if from from (make-array (list (length tensor-shape)) :initial-element 0)))
         (to (if to to tensor-shape))
         (processed-to (nnl2.hli.ts:process-to-indices to tensor-shape))
         (pntr-from (nnl2.hli:make-shape-pntr from))
         (pntr-to (nnl2.hli:make-shape-pntr processed-to)))
    
    (nnl2.ffi:%ad-slice tensor pntr-from pntr-to nnl2.ffi:ad-reverse-mode track-graph)))	  
  
(defun l2-norm (tensor &key force (axes #(0)) (track-graph t) &aux (dtype (nnl2.hli.ad:dtype tensor)))
  (declare (ignore axes))
   
  (let ((out (nnl2.ffi:%ad-l2norm tensor force nnl2.ffi:ad-reverse-mode track-graph)))
    (if force 
	  (let ((cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype)))
	    (cffi:mem-ref out cffi-type))
		
	  out)))
  
(defun norm (tensor &key force (axes #(0)) (p :l2) (track-graph t))
  "WARNING: YET DOES NOT SUPPORT AXES (W.I.P.)
   
   Applies passed norm to passed norm (available: (:l2))
   
   tensor: Input tensor
   axes (&key): Axes to apply the norm. DOES NOT FULLY WORK YET"
   
  (case p
    (:l2 (l2-norm tensor :axes axes :force force :track-graph track-graph))
	(otherwise (error "Incorrect :p key in norm~%"))))
  