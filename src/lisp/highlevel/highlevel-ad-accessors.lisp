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
	  
(cffi:defcfun ("nnl2_ad_zero_grad" zero-grad) :void
  (ad-tensor :pointer))	  
	  
(cffi:defcfun ("nnl2_ad_get_data" data) :pointer
  (ad-tensor :pointer))  
  
(cffi:defcfun ("nnl2_ad_get_leaf" is-leaf) :bool
  (ad-tensor :pointer)) 

(cffi:defcfun ("nnl2_ad_get_requires_grad" requires-grad) :bool
  (ad-tensor :pointer))   
  
(cffi:defcfun ("nnl2_ad_backpropagation" backpropagation) :void
  (ad-tensor :pointer))
  
(cffi:defcfun ("nnl2_ad_backpropagation" bp) :void
  (ad-tensor :pointer))  

(cffi:defcfun ("nnl2_ad_backpropagation_through_time" backpropagation-through-time) :void
  (ad-tensor :pointer))  
  
(cffi:defcfun ("nnl2_ad_backpropagation_through_time" bptt) :void
  (ad-tensor :pointer))    
  
(cffi:defcfun ("nnl2_ad_get_grad" grad) :pointer
  (ad-tensor :pointer))
  
(cffi:defcfun ("nnl2_free_ad_tensor" free) :void
  (ad-tensor :pointer))  
  
(defmacro tlet ((&rest bindings) &body body)
  (let ((vars (mapcar #'car bindings)))
    `(let ,bindings
	   (unwind-protect
	       (progn ,@body)
		   
		 (progn 
		   ,@(loop for var in vars
		           collect `(when (typep ,var 'nnl2-ad-tensor) (free ,var))))))))  
  
(defun print-data (ad-tensor)
  (nnl2.hli.ts:print-tensor (data ad-tensor)))
  
(defun print-grad (ad-tensor)
  (nnl2.hli.ts:print-tensor (grad ad-tensor)))

(defun .+/ad/incf! (tensor increment mode)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
		 (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
		 (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
	
	(nnl2.ffi:%ad-add-correspondence tensor incf-pntr mode)))  
  
(defun +=/ad/incf! (tensor increment)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (incf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
    
    (nnl2.ffi:%ad-add-incf-inplace tensor incf-pntr)))  

(defun .*/ad/mulf! (tensor multiplier mode)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (mulf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref mulf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
    
    (nnl2.ffi:%ad-mul-correspondence tensor mulf-pntr mode)))

(defun *=/ad/mulf! (tensor multiplier)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (mulf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref mulf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
    
    (nnl2.ffi:%ad-mul-mulf-inplace tensor mulf-pntr)))	
  
(defun .-/ad/decf! (tensor decrement mode)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (decf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref decf-pntr cffi-dtype) (coerce decrement lisp-dtype))
    
    (nnl2.ffi:%ad-sub-correspondence tensor decf-pntr mode)))

(defun -=/ad/decf! (tensor decrement)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (decf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref decf-pntr cffi-dtype) (coerce decrement lisp-dtype))
    
    (nnl2.ffi:%ad-sub-decf-inplace tensor decf-pntr)))
	
(defun ./ad/divf! (tensor divisor mode)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (divf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divisor lisp-dtype))
    
    (nnl2.ffi:%ad-div-correspondence tensor divf-pntr mode)))	
	
(defun /!/ad/divf! (tensor divisor)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (divf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divisor lisp-dtype))
    
    (nnl2.ffi:%ad-div-divf-inplace tensor divf-pntr)))
	
(defun .^/ad/powf! (tensor exponent mode)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (powf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce exponent lisp-dtype))
    
    (nnl2.ffi:%ad-pow-correspondence tensor powf-pntr mode)))	

(defun ^=/ad/powf! (tensor exponent)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (powf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce exponent lisp-dtype))
    
    (nnl2.ffi:%ad-pow-powf-inplace tensor powf-pntr)))
	
(defun .min/ad/minf! (tensor value mode)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (minf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-min-correspondence tensor minf-pntr mode)))

(defun .min!/ad/minf! (tensor value)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (minf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-min-minf-inplace tensor minf-pntr)))	

(defun .max/ad/maxf! (tensor value mode)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-max-correspondence tensor maxf-pntr mode)))
	
(defun .max!/ad/maxf! (tensor value)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-max-maxf-inplace tensor maxf-pntr)))
	
(defun axpy/ad/axpf! (tensor other alpha mode)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-axpf tensor other-pntr (coerce alpha 'single-float) mode)))	
	
(defun axpy!/ad/axpf! (tensor other alpha)
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-axpf-inplace tensor other-pntr (coerce alpha 'single-float))))	
	
(cffi:defcfun ("nnl2_ad_neg_inplace" .neg!) :void
  (ad-tensor :pointer))		
  
(defun += (a b)
  "In-place addition"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (+=/ad/incf! a b)
      (nnl2.ffi:%ad-+= a b)
      (nnl2.ffi:%ad-add-broadcasting-inplace a b))))  
	  
(defun -= (a b)
  "In-place subtraction"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (-=/ad/decf! a b)
      (nnl2.ffi:%ad--= a b)
      (nnl2.ffi:%ad-sub-broadcasting-inplace a b))))	  
  
(defun *= (a b)
  "In-place multiplication"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (*=/ad/mulf! a b)
      (nnl2.ffi:%ad-*= a b)
      (nnl2.ffi:%ad-mul-broadcasting-inplace a b))))
  
  (defun /! (a b)
  "In-place division"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (/!/ad/divf! a b)
      (nnl2.ffi:%ad-/! a b)
      (nnl2.ffi:%ad-div-broadcasting-inplace a b))))

(defun ^= (a b)
  "In-place pow"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (^=/ad/powf! a b)
      (nnl2.ffi:%ad-^= a b)
      (nnl2.ffi:%ad-pow-broadcasting-inplace a b))))	
	  
(defun .abs! (ad-tensor)
  (nnl2.ffi:%ad-.abs! ad-tensor))		  
	  
(defun .min! (a b)
  "In-place min"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.min!/ad/minf! a b)
      (nnl2.ffi:%ad-.min! a b)
      (nnl2.ffi:%ad-min-broadcasting-inplace a b))))	  
	  
(defun .max! (a b)
  "In-place max"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.max!/ad/maxf! a b)
      (nnl2.ffi:%ad-.max! a b)
      (nnl2.ffi:%ad-max-broadcasting-inplace a b))))
	  
(defun axpy! (a b &key (alpha 1.0))
  "In-place a+b*c"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:axpy!/ad/axpf! a b alpha)
      (nnl2.ffi:%ad-axpy! a b alpha)
      (nnl2.ffi:%ad-axpy-broadcasting-inplace a b alpha))))	  
	  
(defun scale! (a b)
  (nnl2.ffi:%ad-scale! a (coerce b 'single-float)))
  
(defun .exp! (ad-tensor)
  (nnl2.ffi:%ad-.exp! ad-tensor))  

(defun .log! (ad-tensor)
  (nnl2.ffi:%ad-.log! ad-tensor))    
  
(defun .relu! (ad-tensor)
  (nnl2.ffi:%ad-.relu! ad-tensor))  	 

(defun .leaky-relu! (ad-tensor &key (alpha 0.01))
  (nnl2.ffi:%ad-.leaky-relu! ad-tensor alpha))  	
  
(defun .sigmoid! (ad-tensor &key (approx t))
  (nnl2.ffi:%ad-.sigmoid! ad-tensor approx))  
  
(defun .tanh! (ad-tensor &key (approx t))
  (nnl2.ffi:%ad-.tanh! ad-tensor approx))    
  
(in-package :nnl2.hli.ad.r)

(defun .+ (a b)
  "Element-wise addition"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.+/ad/incf! a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-.+ a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-add-broadcasting a b nnl2.ffi:ad-reverse-mode))))
  
(defun .* (a b)
  "Element-wise multiplication"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.*/ad/mulf! a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-.* a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-mul-broadcasting a b nnl2.ffi:ad-reverse-mode))))	  
	  
(defun gemm (a b)
  (nnl2.ffi:%ad-gemm a b nnl2.ffi:ad-reverse-mode))

(defun .- (a b)
  "Element-wise subtraction"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.-/ad/decf! a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-.- a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-sub-broadcasting a b nnl2.ffi:ad-reverse-mode))))
	  
(defun ./ (a b)
  "Element-wise division"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:./ad/divf! a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-./ a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-div-broadcasting a b nnl2.ffi:ad-reverse-mode))))
	
(defun .^ (a b)
  "Element-wise pow"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.^/ad/powf! a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-.^ a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-pow-broadcasting a b nnl2.ffi:ad-reverse-mode))))    
	  
(defun .abs (ad-tensor)
  (nnl2.ffi:%ad-.abs ad-tensor nnl2.ffi:ad-reverse-mode))	  
	
(defun .min (a b)
  "Element-wise min"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.min/ad/minf! a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-.min a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-min-broadcasting a b nnl2.ffi:ad-reverse-mode))))	  	  
	
(defun .max (a b)
  "Element-wise max"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.max/ad/maxf! a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-.max a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-max-broadcasting a b nnl2.ffi:ad-reverse-mode))))	  	  
	
(defun scale (a b &key save-type)
  (nnl2.ffi:%ad-scale a (coerce b 'single-float) save-type nnl2.ffi:ad-reverse-mode))
  
(defun .log (ad-tensor &key save-type)
  (nnl2.ffi:%ad-.log ad-tensor save-type nnl2.ffi:ad-reverse-mode))
  
(defun .exp (ad-tensor &key save-type)
  (nnl2.ffi:%ad-.exp ad-tensor save-type nnl2.ffi:ad-reverse-mode))  
  
(defun axpy (a b &key (alpha 1.0))
  "Element-wise a+b*c"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:axpy/ad/axpf! a b alpha nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-axpy a b alpha nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-axpy-broadcasting a b alpha nnl2.ffi:ad-reverse-mode))))

(defun .relu (ad-tensor)
  (nnl2.ffi:%ad-.relu ad-tensor nnl2.ffi:ad-reverse-mode)) 

(defun .leaky-relu (ad-tensor &key save-type (alpha 0.01))
  (nnl2.ffi:%ad-.leaky-relu ad-tensor alpha save-type nnl2.ffi:ad-reverse-mode)) 
    
(defun .sigmoid (ad-tensor &key (approx t))
  (nnl2.ffi:%ad-.sigmoid ad-tensor approx nnl2.ffi:ad-reverse-mode))  
  	
(defun .tanh (ad-tensor &key (approx t))
  (nnl2.ffi:%ad-.tanh ad-tensor approx nnl2.ffi:ad-reverse-mode)) 	
	
(defun .neg (ad-tensor)
  (nnl2.ffi:%.neg ad-tensor nnl2.ffi:ad-reverse-mode))
	