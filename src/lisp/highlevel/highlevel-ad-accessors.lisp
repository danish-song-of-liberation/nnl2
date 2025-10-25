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
  
(in-package :nnl2.hli.ad.r)

(defun .+ (a b)
  "Element-wise addition"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      nil ;; correspondence
      (nnl2.ffi:%ad-.+ a b nnl2.ffi:ad-reverse-mode)
      (nnl2.ffi:%ad-.+/broadcasting a b nnl2.ffi:ad-reverse-mode))))
  
(defun .* (a b)
  "Element-wise multiplication"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      nil ;; correspondence
      (nnl2.ffi:%ad-.* a b nnl2.ffi:ad-reverse-mode)
      nil))) ;; broadcasting
	  
(defun gemm (a b)
  (nnl2.ffi:%ad-gemm a b nnl2.ffi:ad-reverse-mode))

(defun .- (a b)
  "Element-wise subtraction"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      nil ;; correspondence
      (nnl2.ffi:%ad-.- a b nnl2.ffi:ad-reverse-mode)
      nil))) ;; broadcasting
	  
(defun ./ (a b)
  "Element-wise division"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      nil ;; correspondence
      (nnl2.ffi:%ad-./ a b nnl2.ffi:ad-reverse-mode)
      nil))) ;; broadcasting
	  
(defun .^ (a b)
  "Element-wise pow"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      nil ;; correspondence
      (nnl2.ffi:%ad-.^ a b nnl2.ffi:ad-reverse-mode)
      nil))) ;; broadcasting	  
	  
(defun .abs (ad-tensor)
  (nnl2.ffi:%ad-.abs ad-tensor nnl2.ffi:ad-reverse-mode))	  
	  