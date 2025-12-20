(in-package :nnl2.hli.ad)

;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/highlevel-ad-accessors.lisp
;; File: highlevel-ad-accessors.lisp

;; Contains a high-level interface for all the main AD functions in ffi-c-core.lisp

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(deftype nnl2-ad-tensor () 
  #+sbcl      'sb-sys:system-area-pointer
  #+clisp     'fi:foreign-data
  #+ccl       'ccl:macptr
  #+ecl       'si:foreign-data
  #+abcl      'system:foreign-pointer
  #+lispworks 'fli:pointer
  #+allegro   'excl:foreign-pointer)  

(defun ad-get-shape-as-list (ad-tensor rank)
  "Gets the shape of AD tensor as a list
  
   Args:
       ad-tensor: Input tensor
	   rank: Tensor number dimensions
	   
   Returns:
       Shape of ad tensor as a list
	   
   Example:
       (nnl2.hli.ad:tlet ((foo (nnl2.hli.ad:ones #(5 5)))) ;; #(5 5) - 2 dimensions
	     (nnl2.hli.ad::ad-get-shape-as-list foo 2)) ;; '(5 5)"
	   
  (loop with rank-t = (if rank rank (rank ad-tensor))
        with shape-pointer = (nnl2.ffi:%ad-shape ad-tensor)
        for i from 0 below rank-t
        collect (cffi:mem-aref shape-pointer :int i)))

(defun ad-get-shape-as-vector (ad-tensor rank)
  "Gets the shape of AD tensor as a vector
  
   Args:
       ad-tensor: Input AD tensor
       rank: Tensor number of dimensions
       
   Returns:
       Shape of AD tensor as a vector
       
   Example:
       (nnl2.hli.ad:tlet ((foo (nnl2.hli.ad:ones #(3 4)))) ;; #(3 4) - 2 dimensions
         (nnl2.hli.ad::ad-get-shape-as-vector foo 2)) ;; #(3 4)"
  
  (let* ((rank-t (if rank rank (rank ad-tensor)))
         (vec (make-array rank-t))
         (shape-pointer (nnl2.ffi:%ad-shape ad-tensor)))
		 
	;; Copying data	 
    (dotimes (i rank-t)
      (setf (aref vec i) (cffi:mem-aref shape-pointer :int i)))
	  
    vec))

(defun rank (ad-tensor)
  "Gets rank (number of dimensions) of an AD tensor
  
   Args:
       ad-tensor: Input AD tensor
       
   Returns:
       Integer rank (number of dimensions) of the tensor
       
   Example:
       (nnl2.hli.ad:tlet ((foo (nnl2.hli.ad:ones #(2 3 4)))) ;; rank = 3
         (nnl2.hli.ad:rank foo)) ;; 3"
		 
  (nnl2.ffi:%ad-rank ad-tensor))

(defun dtype (ad-tensor &key (from :data))
  "Gets the dtype of an AD tensor as a keyword symbol 
  
   Args:
       ad-tensor: Input AD tensor
       from (&key): Which dtype to return (:data or :grad)
       
   Returns:
       Keyword symbol representing dtype of the tensor's data or gradient
       
   Example:
       (nnl2.hli.ad:tlet ((foo (nnl2.hli.ad:ones #(5 5) :dtype :float64)))
         (nnl2.hli.ad:dtype foo)) ;; :float64
		 
       (nnl2.hli.ad:dtype foo :from :grad)   ;; dtype of gradient tensor"
		 
  (ecase from
    (:data (nnl2.ffi:%ad-dtype-as-data ad-tensor))
    (:grad (nnl2.ffi:%ad-dtype-as-grad ad-tensor))))
  
(defun int-dtype (ad-tensor &key (from :data))
  "Gets the dtype of an AD tensor as an integer enum value (C enum)
  
   Args:
       ad-tensor: Input AD tensor
       from (&key): Which dtype to return (:data or :grad)
       
   Returns:
       Integer value representing dtype enum (matches C backend)
       
   Example:
       (nnl2.hli.ad:tlet ((foo (nnl2.hli.ad:ones #(5 5) :dtype :float64)))
         (nnl2.hli.ad:int-dtype foo)) ;; nnl2 C Enum value of float64
		 
       (nnl2.hli.ad:int-dtype foo :from :grad)"

  (ecase from
    (:data (nnl2.ffi:%ad-dtype-as-data-int ad-tensor))
	(:grad (nnl2.ffi:%ad-dtype-as-grad-int ad-tensor))))

(declaim (ftype (function (nnl2-ad-tensor (integer 0 *)) list) ad-get-shape-as-list)
         (ftype (function (nnl2-ad-tensor (integer 0 *)) vector) ad-get-shape-as-vector)
         (ftype (function (nnl2-ad-tensor) integer) ad-rank))

(defun shape (ad-tensor &key (as :vector))
  "Gets the shape of an AD tensor
  
   Args:
       ad-tensor: Input AD tensor
       as (&key): Return type (:vector, :list, or :pointer)
       
   Returns:
       Shape of the tensor in the specified format:
         :vector - simple-vector of dimensions
         :list - list of dimensions
         :pointer - raw CFFI pointer to shape array
       
   Example:
       (nnl2.hli.ad:tlet ((foo (nnl2.hli.ad:ones #(2 3))))
         (nnl2.hli.ad:shape foo)) ;; #(2 3)
		 
       (nnl2.hli.ad:shape foo :as :list) ;; '(2 3)
       (nnl2.hli.ad:shape foo :as :pointer) ;; Depends on your lisp implementation. Usually something like `<sb-sys:int-sap ...>` (sbcl), `fi:foreign-data ...` (clisp)"
	   
  (let ((rank (rank ad-tensor)))
    (case as
      (:list    (ad-get-shape-as-list ad-tensor rank))
      (:vector  (ad-get-shape-as-vector ad-tensor rank))
      (:pointer (nnl2.ffi:%ad-shape ad-tensor))
      (otherwise (error "Unknown type: ~a~%" as)))))

(defun shapes-equal-p (ad-tensor-a ad-tensor-b)
  "Checks whether shapes of two AD tensors are equal
   Needs for internal tensor dispatch
  
   Args:
       ad-tensor-a: First tensor
       ad-tensor-b: Second tensor
       
   Returns:
       T if shapes are equal, NIL otherwise
       
   Example:
       (nnl2.hli.ad:tlet ((a (nnl2.hli.ad:ones #(3 4))) (b (nnl2.hli.ad:rand #(3 4))))
         (nnl2.hli.ad::shapes-equal-p a b)) ;; T"
		 
  (let ((shape-a (shape ad-tensor-a :as :vector))
        (shape-b (shape ad-tensor-b :as :vector)))
		
    (equalp shape-a shape-b)))

(cffi:defcfun ("nnl2_ad_get_num_roots" num-roots) :int
  (ad-tensor :pointer))
  
(defun get-roots-as-list (roots-pointer num-roots)
  "Converts roots pointer to a Lisp list of pointers"
  (loop for i from 0 below num-roots
		collect (cffi:mem-aref roots-pointer :pointer i)))
  
(defun roots (ad-tensor &key (as :list))
  "Gets roots of an AD tensor
  
   Args:
       ad-tensor: Input tensor
       as (&key): Return type (:list or :pointer)
	   
   Returns:
       List of root tensors or raw pointer"
	   
  (ecase as 
    (:pointer (nnl2.ffi:%ad-roots ad-tensor))
	(:list (get-roots-as-list (nnl2.ffi:%ad-roots ad-tensor) (num-roots ad-tensor)))))
	
(defun (setf roots) (ad-tensors-list self)
  "Sets roots of an AD tensor
  
   Args:
       ad-tensors-list: List of AD tensors
       self: Target tensor"
	   
  (let* ((new-len (length ad-tensors-list)) 
         (tensors-pool (cffi:foreign-alloc :pointer :count new-len)))
		 
	(dotimes (i new-len)
	  (setf (cffi:mem-aref tensors-pool :pointer i) (nth i ad-tensors-list)))
	  
	(nnl2.ffi:%ad-roots-setter self tensors-pool new-len)))

(defun strides (ad-tensor &key (from :data) (as :vector))
  "Gets strides of an AD tensor
  
   Args:
       ad-tensor: Input tensor
       from (&key): Source (:data or :grad)
       as (&key): Return type (:vector or :list)
	   
   Returns:
       Strides of the tensor"
	   
  (ecase from
    (:data (nnl2.hli.ts:strides (data ad-tensor) :as as))
	(:grad (nnl2.hli.ts:strides (grad ad-tensor) :as as))))

(defun higher-rank-tensor (a b)
  "Returns two (with `(values ...)`) tensors (higher . lower) depending on their rank"
  (nnl2.hli:fastcall (if (> (rank a) (rank b)) (values a b) (values b a))))  

(defmacro with-tensor-dispatch ((a b) tensor-case same-shape-case broadcast-case)
  "Dispatcher for binary tensor operations
  
   Args:
       a: First tensor
       b: Second tensor
       tensor-case: Form executed when b is a scalar
       same-shape-case: Form executed when shapes match
       broadcast-case: Form executed when broadcasting is required
	   
   Returns:
       Expanded conditional form selecting one of the given cases"
  
  (let ((a-sym (gensym "A"))
        (b-sym (gensym "B")))
		
    `(let ((,a-sym ,a)
           (,b-sym ,b))
   
       (cond
	   
		 ;; a=ad-tensor, b=number
	     ((typep ,b-sym 'real) ,tensor-case)
		 
		 ;; a=number, b=ad-tensor
		 ((typep ,a-sym 'real) (error "You can't apply a tensor function to a scalar"))
		 
		 ;; a=ad-tensor[5, 5], b=ad-tensor[5, 5]
         ((shapes-equal-p ,a-sym ,b-sym) ,same-shape-case)
		 
		 ;; a=ad-tensor[5, 5], b=ad-tensor[5]
		 (t (multiple-value-bind (higher lower) (higher-rank-tensor ,a-sym ,b-sym) ,broadcast-case))))))

(defun empty (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  "Creates an uninitialized AD tensor
  
   Args:
       indices: Shape specification
       dtype (&key): Tensor dtype
       requires-grad (&key): Whether the tensor requires gradients
       name (&key): Optional tensor name
	   
   Returns:
       New AD tensor with uninitialized data"
	   
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  (nnl2.ffi:%ad-empty shape rank dtype requires-grad name))))

(defun zeros (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  "Creates an AD tensor filled with zeros
  
   Args:
       indices: Shape specification
       dtype (&key): Tensor dtype
       requires-grad (&key): Whether the tensor requires gradients
       name (&key): Optional tensor name
	   
   Returns:
       New AD tensor filled with zeros"
	   
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  (nnl2.ffi:%ad-zeros shape rank dtype requires-grad name))))
	  
(defun ones (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  "Creates an AD tensor filled with ones
  
   Args:
       indices: Shape specification
       dtype (&key): Tensor dtype
       requires-grad (&key): Whether the tensor requires gradients
       name (&key): Optional tensor name
	   
   Returns:
       New AD tensor filled with ones"
	   
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  (nnl2.ffi:%ad-ones shape rank dtype requires-grad name))))
	  	  
(defun full (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name "") (filler 1.0))
  "Creates an AD tensor filled with a constant value
  
   Args:
       indices: Shape specification
       dtype (&key): Tensor dtype
       requires-grad (&key): Whether the tensor requires gradients
       name (&key): Optional tensor name
       filler (&key): Constant value to fill with
	   
   Returns:
       New AD tensor filled with a constant value"
	   
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
	  
(defun uniform-like (tensor &key (from 0) (to 1))
  "Creates a random tensor with uniform distribution matching another tensor's shape
  
   Args:
       tensor: Reference tensor whose shape will be used
       from (&key) (default 0): Lower bound of uniform distribution 
       to (&key) (default 1): Upper bound of uniform distribution 
       
   Returns:
       New AD tensor with same shape as input, filled with random values"
       
  (nnl2.hli:fastcall
    (let* ((dtype (dtype tensor))
		   (lisp-type    (nnl2.hli.ts:type/nnl2->lisp dtype))
		   (cffi-type    (nnl2.hli.ts:type/nnl2->cffi dtype))
		   (to-pntr      (cffi:foreign-alloc cffi-type))
		   (from-pntr    (cffi:foreign-alloc cffi-type))
		   (coerced-to   (coerce to lisp-type))
		   (coerced-from (coerce from lisp-type)))
		   
	  (setf (cffi:mem-ref from-pntr cffi-type) coerced-from
			(cffi:mem-ref to-pntr cffi-type)   coerced-to)
			
	  (nnl2.ffi:%ad-uniform-like tensor from-pntr to-pntr))))		
	  
(defun %internal-uniform (indices dtype requires-grad name from to)
  "Internal helper for creating random AD tensors in a given range"
  
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
	   (nnl2.ffi:%ad-uniform shape rank dtype requires-grad name from-pntr to-pntr)))))
	
(defun uniform (indices &key (from 0) (to 1) (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  "Creates an AD tensor with random values in the given range (default [0, 1])"
  (%internal-uniform indices dtype requires-grad name from to))
	 
(defun xavier (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name "") (in 0) (out 0) (gain 1.0s0) (distribution :normal))
  "Creates an AD tensor initialized with Xavier initialization
  
   Args:
       indices: Shape specification
       dtype (&key): Tensor dtype
       requires-grad (&key): Whether gradients are required
       name (&key): Optional tensor name
       in (&key): Number of input neurons
       out (&key): Number of output neurons
       gain (&key): Gain factor
       distribution (&key): :normal or :uniform
	   
   Returns:
       AD tensor initialized using Xavier method"
	   
  (assert (not (or (zerop in) (minusp in))) nil "Bad `in` was passed to xavier (AD)")
  (assert (not (or (zerop out) (minusp out))) nil "Bad `out` was passed to xavier (AD)")
  
  (nnl2.hli:fastcall
    (let ((dist (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))))
      (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	    (declare (type integer rank))
        (nnl2.ffi:%ad-xavier shape rank dtype requires-grad name in out gain dist))))) 

(defun kaiming (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name "") (in 0) (out 0) (gain (sqrt 2.0s0)) (distribution :normal) (mode :fan-in))
  "Creates an AD tensor initialized with Kaiming (He) initialization
  
   Args:
       indices: Shape specification
       dtype (&key): Tensor dtype
       requires-grad (&key): Whether gradients are required
       name (&key): Optional tensor name
       in (&key): Number of input neurons
       out (&key): Number of output neurons
       gain (&key): Gain factor (usually sqrt(2.0) for ReLU)
       distribution (&key): :normal or :uniform
       mode (&key): :fan-in, :fan-out, or :fan-avg
       
   Returns:
       AD tensor initialized using Kaiming method"
  
  (assert (not (or (zerop in) (minusp in))) nil "Bad `in` was passed to kaiming (AD)")
  (assert (not (or (zerop out) (minusp out))) nil "Bad `out` was passed to kaiming (AD)")
  
  (nnl2.hli:fastcall
    (let ((dist (ecase distribution (:normal 2.0s0) (:uniform 6.0s0)))
          (mode-val (ecase mode (:fan-in 0) (:fan-out 1) (:fan-avg 2))))
		  
      (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	    (declare (type integer rank))
        (nnl2.ffi:%ad-kaiming shape rank dtype requires-grad name in out gain dist mode-val)))))

(defun rand (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  "Creates an AD tensor of the specified shape filled with uniform random values [0, 1]"
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%ad-rand shape rank dtype requires-grad name)))

(defun randn (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name "") (mean 0.0d0) (std 1.0d0))
  "Creates an AD tensor of the specified shape filled with random numbers from normal distribution N(mean, std^2)"
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%ad-randn shape rank dtype requires-grad name (coerce mean 'double-float) (coerce std 'double-float))))

(cffi:defcfun ("nnl2_ad_rand_like" rand-like) :pointer  
  (ad-tensor :pointer))
  
(defun randn-like (ad-tensor &key (mean 0.0d0) (std 1.0d0))
  "Creates a new AD tensor with the same shape as the input tensor, filled with random numbers from N(mean, std^2)
   
   ad-tensor: Input AD tensor to copy shape from
   mean (&key) (default: 0.0): Mean of the normal distribution
   std (&key) (default: 1.0): Standard deviation of the normal distribution
   
   Returns: A new AD tensor with the same shape as input, filled with random values from N(mean, std^2)
   
   Examples:
     (randn-like my-ad-tensor) ; Creates AD tensor like my-ad-tensor with values from N(0, 1)
     (randn-like my-ad-tensor :mean 0.0 :std 0.1) ; Creates AD tensor with values from N(0, 0.01)"
  
  (nnl2.ffi:%ad-randn-like ad-tensor (coerce mean 'double-float) (coerce std 'double-float)))
  
(defun make-tensor (data &key requires-grad (dtype nnl2.system:*default-tensor-type*))
  "Creates an AD tensor from a Lisp array
  
   Args:
       data: Lisp array
       dtype (&key): Tensor dtype
	   
   Returns:
       AD tensor sharing memory with a TS tensor"
	   
  (let* ((shape (array-dimensions data))
		 (ts-tensor (nnl2.hli.ts:make-tensor data :dtype dtype :shape-hint shape))
		 (ad-tensor (empty shape :dtype dtype :requires-grad requires-grad)))
		 
	(nnl2.ffi:%data-pntr-share-setter ad-tensor ts-tensor)
	
	ad-tensor))
	
(defun from-flatten (flatten-data indices &key requires-grad (dtype nnl2.system:*default-tensor-type*))
  "Creates an AD tensor from a flat list and target shape
  
   Args:
       flatten-data: Flat list of values
       indices:      Shape specification
       dtype (&key):        Tensor dtype
	   
   Returns:
       AD tensor reconstructed from flat data"
	   
  (let ((ad-tensor (empty indices :dtype dtype :requires-grad requires-grad))
		(ts-tensor (nnl2.hli.ts:from-flatten flatten-data indices :dtype dtype)))
		
	(nnl2.ffi:%data-pntr-share-setter ad-tensor ts-tensor)
	
	ad-tensor))
	
(defun transposition! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "WARNING. 
  
   This function intentionally returns mathematically 
   incorrect results for maximum performance. 
   
   For correctness, use transpose! with the :force t flag
  
   Performs in-place transposition of an AD tensor (O(1))
  
   Args:
       ad-tensor: Input tensor
       track-graph: Whether to track operation in autograd graph
	   
   Returns:
       AD tensor with transposed shape (in-place)"
	   
  (nnl2.ffi:%ad-transposition-inplace ad-tensor track-graph))
  
(defun transpose! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*) force)
  "WARNING. 
  
   For maximum performance, the function intentionally 
   returns incorrect mathematical results. 
   
   For correctness, add the :force t flag
   
   Performs in-place transpose of an AD tensor (O(n))
  
   Args:
       ad-tensor: Input tensor
       track-graph (&key): Whether to track operation in autograd graph
       force (&key): Optional flag to mathematical correctness
	   
   Returns:
       AD tensor transposed in-place"
	   
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
  "Performs backpropagation on an AD tensor
  
   Args:
       ad-tensor: Input tensor
       retain-graph (&key): Whether to retain the computational graph
	   
   Returns:
       AD tensor after backpropagation"
	   
  (nnl2.ffi:%backpropagation ad-tensor retain-graph))  

(defun backpropagation (ad-tensor &key retain-graph)
  "Alias for `bp`; performs backpropagation on an AD tensor
  
   Args:
       ad-tensor: Input tensor
       retain-graph (&key): Whether to retain the computational graph
	   
   Returns:
       AD tensor after backpropagation"
	   
  (nnl2.ffi:%backpropagation ad-tensor retain-graph))  
  
(defun bptt (ad-tensor &key retain-graph)
  "Performs simplified backpropagation through time (BPTT) on an AD tensor
   Needs for RNN
  
   Args:
       ad-tensor: Input tensor
       retain-graph (&key): Whether to retain the computational graph
	   
   Returns:
       AD tensor after BPTT"
	   
  (nnl2.ffi:%bptt ad-tensor retain-graph))    
  
(defun backpropagation-through-time (ad-tensor &key retain-graph)
  "Alias for `bptt`; performs backpropagation through time on an AD tensor
  
   Args:
       ad-tensor: Input tensor
       retain-graph (&key): Whether to retain the computational graph
	   
   Returns:
       AD tensor after BPTT"
	   
  (nnl2.ffi:%bptt ad-tensor retain-graph))    
  
(defmacro tlet ((&rest bindings) &body body)
  "Like `let` but automatically frees AD tensors after use
  
   Args:
       bindings ((&rest)): List of variable bindings
       body (&body): Forms to evaluate
	   
   Returns:
       Result of body; frees any nnl2 AD tensors in bindings afterwards"
	   
  (let ((vars (mapcar #'car bindings)))
    `(let ,bindings
	   (unwind-protect
	       (progn ,@body)
		   
		 (progn 
		   ,@(loop for var in vars
		           collect `(when (typep ,var 'nnl2-ad-tensor) (free ,var))))))))  
  
(defmacro tlet* ((&rest bindings) &body body)
  "Like `let*` but automatically frees AD tensors after use
  
   Args:
       bindings: List of variable bindings
       body:     Forms to evaluate
	   
   Returns:
       Result of body; frees any nnl2 AD tensors in bindings afterwards"
  
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
  "Prints the data of an AD tensor"
  (nnl2.hli.ts:print-tensor (data ad-tensor)))

(defun print-grad (ad-tensor)
  "Prints the gradient of an AD tensor"
  (nnl2.hli.ts:print-tensor (grad ad-tensor)))
  
(defun step-ts (ad-tensor &key (lr 1.0))
  "Performs a gradient descent step on the tensor's data and returns a TS tensor
  
   Args:
       ad-tensor: AD tensor whose data will be updated
       lr (&key): Learning rate
	   
   Returns:
       TS tensor with updated data (original AD tensor remains unchanged)"
	   
  (nnl2.ffi:%ad-step ad-tensor (coerce lr 'single-float)))
  
(defun step! (ad-tensor &key (lr 1.0))
  "Performs an in-place gradient descent step on an AD tensor's data
  
   Args:
       ad-tensor: AD tensor whose data will be updated in-place
       lr (&key): Learning rate
	   
   Returns:
       AD tensor with data updated by subtracting lr * grad (in-place)"
	   
  (nnl2.ffi:%ad-step! ad-tensor (coerce lr 'single-float)))
  
(defun .+/ad/incf! (tensor increment mode track-graph)
  "Adds increment to tensor"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
		 (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
		 (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
	
	(nnl2.ffi:%ad-add-correspondence tensor incf-pntr mode track-graph)))  
  
(defun +=/ad/incf! (tensor increment track-graph)
  "Adds increment to tensor in-place"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (incf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
    
    (nnl2.ffi:%ad-add-incf-inplace tensor incf-pntr track-graph)))  

(defun .*/ad/mulf! (tensor multiplier mode track-graph)
  "Multiplies tensor by multiplier"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (mulf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref mulf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
    
    (nnl2.ffi:%ad-mul-correspondence tensor mulf-pntr mode track-graph)))

(defun *=/ad/mulf! (tensor multiplier track-graph)
  "Multiplies tensor by multiplier in-place"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (mulf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref mulf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
    
    (nnl2.ffi:%ad-mul-mulf-inplace tensor mulf-pntr track-graph)))	
  
(defun .-/ad/decf! (tensor decrement mode track-graph)
  "Subtracts decrement from tensor"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (decf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref decf-pntr cffi-dtype) (coerce decrement lisp-dtype))
    
    (nnl2.ffi:%ad-sub-correspondence tensor decf-pntr mode track-graph)))

(defun -=/ad/decf! (tensor decrement track-graph)
  "Subtracts decrement from tensor in-place"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (decf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref decf-pntr cffi-dtype) (coerce decrement lisp-dtype))
    
    (nnl2.ffi:%ad-sub-decf-inplace tensor decf-pntr track-graph)))
	
(defun ./ad/divf! (tensor divisor mode track-graph)
  "Divides tensor by divisor"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (divf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divisor lisp-dtype))
    
    (nnl2.ffi:%ad-div-correspondence tensor divf-pntr mode track-graph)))	
	
(defun /!/ad/divf! (tensor divisor track-graph)
  "Divides tensor by divisor in-place"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (divf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divisor lisp-dtype))
    
    (nnl2.ffi:%ad-div-divf-inplace tensor divf-pntr track-graph)))
	
(defun .^/ad/powf! (tensor exponent mode track-graph)
  "Raises tensor to exponent"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (powf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce exponent lisp-dtype))
    
    (nnl2.ffi:%ad-pow-correspondence tensor powf-pntr mode track-graph)))	

(defmacro .square! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Raises the tensor to the power of 2 in-place with automatic differentiation support
   tensor: Input tensor to be modified in-place
   track-graph: Controls whether the operation is recorded in the AD graph"
   
  `(nnl2.hli.ad:^=/ad/powf! ,tensor 2.0s0 ,track-graph))  
  
(defmacro .cube! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Raises the tensor to the power of 3 in-place with automatic differentiation support
   tensor: Input tensor to be modified in-place
   track-graph: Controls whether the operation is recorded in the AD graph"
   
  `(nnl2.hli.ad:^=/ad/powf! ,tensor 3.0s0 ,track-graph))  	
		  
(defun ^=/ad/powf! (tensor exponent track-graph)
  "Raises tensor to exponent in-place"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (powf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce exponent lisp-dtype))
    
    (nnl2.ffi:%ad-pow-powf-inplace tensor powf-pntr track-graph)))
	
(defun .min/ad/minf! (tensor value mode track-graph)
  "Applies elementwise minimum with value"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (minf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-min-correspondence tensor minf-pntr mode track-graph)))

(defun .min!/ad/minf! (tensor value track-graph)
  "Applies elementwise minimum with value in-place"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (minf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-min-minf-inplace tensor minf-pntr track-graph)))	

(defun .max/ad/maxf! (tensor value mode track-graph)
  "Applies elementwise maximum with value"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-max-correspondence tensor maxf-pntr mode track-graph)))
	
(defun .max!/ad/maxf! (tensor value track-graph)
  "Applies elementwise maximum with value in-place"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce value lisp-dtype))
    
    (nnl2.ffi:%ad-max-maxf-inplace tensor maxf-pntr track-graph)))
	
(defun axpy/ad/axpf! (tensor other alpha mode track-graph)
  "Performs axpy: tensor + alpha * other"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-axpf tensor other-pntr (coerce alpha 'single-float) mode track-graph)))	
	
(defun axpy!/ad/axpf! (tensor other alpha track-graph)
  "Performs axpy in-place: tensor += alpha * other"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-axpf-inplace tensor other-pntr (coerce alpha 'single-float) track-graph)))	
	
(defun .atan2/ad/correspondence! (tensor other alpha mode track-graph)
  "Performs elementwise atan2 in place(internal realization)"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-atan2-correspondence-inplace tensor other-pntr (coerce alpha 'single-float) mode track-graph)))	
	
(defun .atan2/ad/correspondence (tensor other alpha mode track-graph)
  "Performs elementwise atan2 (internal realization)"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-atan2-correspondence tensor other-pntr (coerce alpha 'single-float) mode track-graph)))		
	
(defun axpy!/ad/axpf! (tensor other alpha track-graph)
  "Performs axpy in-place: tensor += alpha * other"
  (let* ((dtype (nnl2.ffi:%ad-dtype-as-data tensor))
         (cffi-dtype (nnl2.hli.ts:type/nnl2->cffi dtype))
         (lisp-dtype (nnl2.hli.ts:type/nnl2->lisp dtype))
         (other-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref other-pntr cffi-dtype) (coerce other lisp-dtype))
    
    (nnl2.ffi:%ad-axpf-inplace tensor other-pntr (coerce alpha 'single-float) track-graph)))		
	
(defmacro with-notrack (&body body)
  "Temporarily disables autograd graph tracking for the enclosed body"
  `(progn
     (let ((nnl2.system:*ad-default-track-graph* nil))
       ,@body)))
	
(cffi:defcfun ("nnl2_ad_neg_inplace" .neg!) :void
  (ad-tensor :pointer))		
  
(defun += (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place addition"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (+=/ad/incf! a b track-graph)	
      (nnl2.ffi:%ad-+= a b track-graph)
      (nnl2.ffi:%ad-add-broadcasting-inplace a b track-graph))))  
	  
(defun -= (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place subtraction"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (-=/ad/decf! a b track-graph)
      (nnl2.ffi:%ad--= a b track-graph)
      (nnl2.ffi:%ad-sub-broadcasting-inplace a b track-graph))))	  
  
(defun *= (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place multiplication"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (*=/ad/mulf! a b track-graph)
      (nnl2.ffi:%ad-*= a b track-graph)
      (nnl2.ffi:%ad-mul-broadcasting-inplace a b track-graph))))
  
(defun /! (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place division"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (/!/ad/divf! a b track-graph)
      (nnl2.ffi:%ad-/! a b track-graph)
      (nnl2.ffi:%ad-div-broadcasting-inplace a b track-graph))))

(defun ^= (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place pow"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (^=/ad/powf! a b track-graph)
      (nnl2.ffi:%ad-^= a b track-graph)
      (nnl2.ffi:%ad-pow-broadcasting-inplace a b track-graph))))	
	  
(defun .abs! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place abs"
  (nnl2.ffi:%ad-.abs! ad-tensor track-graph))		  
	  
(defun .min! (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place min"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.min!/ad/minf! a b track-graph)
      (nnl2.ffi:%ad-.min! a b track-graph)
      (nnl2.ffi:%ad-min-broadcasting-inplace a b track-graph))))	 

(defun .atan2! (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place atan2"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.atan2/ad/correspondence! a b track-graph)
      (nnl2.ffi:%ad-atan2-inplace a b track-graph)
      (nnl2.ffi:%ad-atan2-broadcasting-inplace a b track-graph))))	 	  
	  
(defun .max! (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place max"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.max!/ad/maxf! a b track-graph)
      (nnl2.ffi:%ad-.max! a b track-graph)
      (nnl2.ffi:%ad-max-broadcasting-inplace a b track-graph))))
	  
(defun axpy! (a b &key (alpha 1.0) (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place a+b*c"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:axpy!/ad/axpf! a b alpha track-graph)
      (nnl2.ffi:%ad-axpy! a b alpha track-graph)
      (nnl2.ffi:%ad-axpy-broadcasting-inplace a b alpha track-graph))))	  
	  
(defun scale! (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Scales tensor a by scalar b in-place"
  (nnl2.ffi:%ad-scale! a (coerce b 'single-float) track-graph))
  
(defun .exp! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise exponential"
  (nnl2.ffi:%ad-.exp! ad-tensor track-graph))  

(defun .log! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise natural logarithm"
  (nnl2.ffi:%ad-.log! ad-tensor track-graph))    

(defun .log1p! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Applies the natural logarithm of (1 + x) to the tensor in place"
  (nnl2.ffi:%ad-.log1p! ad-tensor track-graph))      
  
(defun .log10! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise base-10 logarithm"
  (nnl2.ffi:%ad-.log10! ad-tensor track-graph))      
  
(defun .log2! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise base-2 logarithm"
  (nnl2.ffi:%ad-.log2! ad-tensor track-graph))      
    
(defun .relu! (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place ReLU activation"
  (nnl2.ffi:%ad-.relu! ad-tensor track-graph))  	 

(defun .leaky-relu! (ad-tensor &key (alpha nnl2.system:*leaky-relu-default-shift*) (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place Leaky ReLU activation with slope alpha"
  (nnl2.ffi:%ad-.leaky-relu! ad-tensor alpha track-graph))  	
  
(defun .sigmoid! (ad-tensor &key (approx t) (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place sigmoid activation; approx enables fast approximation"
  (nnl2.ffi:%ad-.sigmoid! ad-tensor approx track-graph))  
  
(defun .tanh! (ad-tensor &key (approx t) (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place tanh activation; approx enables fast approximation"
  (nnl2.ffi:%ad-.tanh! ad-tensor approx track-graph))    
  
(defun .sqrt! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise square root"
  (nnl2.ffi:%ad-sqrt-inplace tensor track-graph))

(defun .cos! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise ad .cos"
  (nnl2.ffi:%ad-.cos! tensor track-graph))
  
(defun .sin! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise ad .sin"
  (nnl2.ffi:%ad-.sin! tensor track-graph))

(defun .acos! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise ad .acos"
  (nnl2.ffi:%ad-.acos! tensor track-graph))
  
(defun .asin! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise ad .asin"
  (nnl2.ffi:%ad-.asin! tensor track-graph))
  
(defun .atan! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise ad .atan"
  (nnl2.ffi:%ad-.atan! tensor track-graph))
  
(defun .tan! (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "In-place elementwise ad .tan"
  (nnl2.ffi:%ad-.tan! tensor track-graph))  
  
(defun copy (tensor &key (dtype (dtype tensor)))
  "Returns a copy of the tensor, optionally casting to a different dtype"
  (nnl2.ffi:%ad-copy tensor dtype))   
  
(cffi:defcfun ("nnl2_ad_empty_like" empty-like) :pointer
  (ad-tensor :pointer))    
  
(cffi:defcfun ("nnl2_ad_zeros_like" zeros-like) :pointer
  (ad-tensor :pointer))  
  
(cffi:defcfun ("nnl2_ad_ones_like" ones-like) :pointer
  (ad-tensor :pointer))    

(defun full-like (ad-tensor &key (filler 0))
  "Creates a tensor like ad-tensor, filled with the given value"
  (let* ((dtype (dtype ad-tensor))
		 (cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype))
		 (lisp-type (nnl2.hli.ts:type/nnl2->lisp dtype))
		 (filler-pntr (cffi:foreign-alloc cffi-type)))
		 
	(setf (cffi:mem-ref filler-pntr cffi-type) (coerce filler lisp-type))
	
	(let ((result (nnl2.ffi:%ad-full-like ad-tensor filler-pntr)))
	  (cffi:foreign-free filler-pntr)
	  result)))

(defun xavier-like (ad-tensor &key in out (gain 1.0s0) (distribution :normal))
  "Fills a tensor like ad-tensor using Xavier initialization"
  (assert (and in out gain distribution) nil "Incorrect keys was passed in xavier-like")
  (nnl2.ffi:%ad-xavier-like ad-tensor in out gain (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))))
  
(defun kaiming-like (ad-tensor &key in out (gain (sqrt 2.0s0)) (distribution :normal) (mode :fan-in))
  "Fills a tensor like ad-tensor using Kaiming (He) initialization"
  (assert (and in out gain distribution mode) nil "Incorrect keys was passed in ad-kaiming-like")
  (nnl2.ffi:%ad-kaiming-like ad-tensor in out gain 
    (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))
    (ecase mode (:fan-in 0) (:fan-out 1) (:fan-avg 2))))
							
(cffi:defcfun ("nnl2_ad_ncast" ncast) :pointer
  (ad-tensor :pointer)
  (cast-to nnl2.ffi:tensor-type)) 

(cffi:defcfun ("nnl2_ad_detach_inplace" detach!) :void
  (ad-tensor :pointer))  

(cffi:defcfun ("nnl2_ad_detach" detach) :pointer
  (ad-tensor :pointer))  
  
(cffi:defcfun ("nnl2_ad_tensor_name_getter" name) :string
  (ad-tensor :pointer))

(defun (setf name) (new-name ad-tensor)
  "Sets the name of an AD tensor"
  (nnl2.ffi:%nnl2-ad-name-setter ad-tensor new-name))      
  
(defun nrows (ad-tensor)
  "Returns the number of rows of the AD tensor's data"	
  (nnl2.hli.ts:nrows (data ad-tensor)))	

(defun ncols (ad-tensor)
  "Returns the number of columns of the AD tensor's data"
  (nnl2.hli.ts:ncols (data ad-tensor)))	
  
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
  "Performs matrix multiplication (a @ b) with AD tracking"
  (nnl2.ffi:%ad-gemm a b nnl2.ffi:ad-reverse-mode track-graph))

(defun gemmvp (a b vector &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Performs matrix multiplication with bias (a @ b + vector) with AD tracking"
  (nnl2.ffi:%ad-gemmvp a b vector nnl2.ffi:ad-reverse-mode track-graph))

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
  "Computes elementwise absolute value with AD tracking"
  (nnl2.ffi:%ad-.abs ad-tensor nnl2.ffi:ad-reverse-mode track-graph))	  
	
(defun .min (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise min"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.min/ad/minf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-.min a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-min-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))	  	  
	
(defun .atan2 (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise atan2"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.atan2/ad/correspondence a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-atan2 a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-atan2-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))		
	
(defun .max (a b &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise max"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:.max/ad/maxf! a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-.max a b nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-max-broadcasting a b nnl2.ffi:ad-reverse-mode track-graph))))	  	  
	
(defun scale (a b &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  "Scales tensor a by scalar b with AD tracking, returns a new tensor"
  (nnl2.ffi:%ad-scale a (coerce b 'single-float) save-type nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .log (ad-tensor &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  "Computes elementwise natural logarithm with AD tracking, returns a new tensor"
  (nnl2.ffi:%ad-.log ad-tensor save-type nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .log1p (ad-tensor &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  "Applies the natural logarithm of (1 + x) to the tensor with AD tracking, returns a new tensor"
  (nnl2.ffi:%ad-.log1p ad-tensor save-type nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .log10 (ad-tensor &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  "Computes elementwise base-10 logarithm with AD tracking, returns a new tensor"
  (nnl2.ffi:%ad-.log10 ad-tensor save-type nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .log2 (ad-tensor &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  "Computes elementwise base-2 logarithm with AD tracking, returns a new tensor"
  (nnl2.ffi:%ad-.log2 ad-tensor save-type nnl2.ffi:ad-reverse-mode track-graph))  
  
(defun .exp (ad-tensor &key save-type (track-graph nnl2.system:*ad-default-track-graph*))
  "Computes elementwise exponential with AD tracking, returns a new tensor"
  (nnl2.ffi:%ad-.exp ad-tensor save-type nnl2.ffi:ad-reverse-mode  track-graph))  
  
(defun axpy (a b &key (alpha 1.0) (track-graph nnl2.system:*ad-default-track-graph*))
  "Element-wise a+b*c"
  
  (nnl2.hli:fastcall   
    (nnl2.hli.ad:with-tensor-dispatch (a b)
      (nnl2.hli.ad:axpy/ad/axpf! a b alpha nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-axpy a b alpha nnl2.ffi:ad-reverse-mode track-graph)
      (nnl2.ffi:%ad-axpy-broadcasting a b alpha nnl2.ffi:ad-reverse-mode track-graph))))

(defun .relu (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with ReLU activation applied"
  (nnl2.ffi:%ad-.relu ad-tensor nnl2.ffi:ad-reverse-mode track-graph)) 

(defun .leaky-relu (ad-tensor &key save-type (alpha nnl2.system:*leaky-relu-default-shift*) (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with Leaky ReLU activation applied, slope alpha"
  (nnl2.ffi:%ad-.leaky-relu ad-tensor alpha save-type nnl2.ffi:ad-reverse-mode track-graph)) 	
    
(defun .sigmoid (ad-tensor &key (approx t) (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with sigmoid activation applied"
  (nnl2.ffi:%ad-.sigmoid ad-tensor approx nnl2.ffi:ad-reverse-mode track-graph))  
  	
(defun .tanh (ad-tensor &key (approx t) (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with tanh activation applied"
  (nnl2.ffi:%ad-.tanh ad-tensor approx nnl2.ffi:ad-reverse-mode track-graph)) 	
	
(defun .neg (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with elementwise negation"
  (nnl2.ffi:%.neg ad-tensor nnl2.ffi:ad-reverse-mode track-graph))

(defun .sin (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with elementwise sine"
  (nnl2.ffi:%ad-.sin ad-tensor nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .cos (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with elementwise cos"
  (nnl2.ffi:%ad-.cos ad-tensor nnl2.ffi:ad-reverse-mode track-graph))

(defun .tan (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with elementwise tan"
  (nnl2.ffi:%ad-.cos ad-tensor nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .asin (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with elementwise arcsine"
  (nnl2.ffi:%ad-.asin ad-tensor nnl2.ffi:ad-reverse-mode track-graph))
  
(defun .acos (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with elementwise arccos"
  (nnl2.ffi:%ad-.acos ad-tensor nnl2.ffi:ad-reverse-mode track-graph))

(defun .atan (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with elementwise arctan"
  (nnl2.ffi:%ad-.atan ad-tensor nnl2.ffi:ad-reverse-mode track-graph))
  
(defun transposition (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a transposed view of the tensor (O(1))"
  (nnl2.ffi:%ad-transposition ad-tensor nnl2.ffi:ad-reverse-mode track-graph))
  
(defun transpose (ad-tensor &key (track-graph nnl2.system:*ad-default-track-graph*) force)
  "Returns a new transposed tensor (O(n))"
  (nnl2.ffi:%ad-transpose ad-tensor nnl2.ffi:ad-reverse-mode track-graph force))    

(defun reshape (tensor new-shape &key force (track-graph nnl2.system:*ad-default-track-graph*))
  "Reshapes tensor to new shape, optionally forcing a copy"
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr new-shape)
    (nnl2.ffi:%ad-reshape tensor shape rank force nnl2.ffi:ad-reverse-mode track-graph)))

(defun reinterpret (tensor new-shape &key force (track-graph nnl2.system:*ad-default-track-graph*))
  "Reinterprets tensor as new shape without copying, optionally forcing"
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr new-shape)
    (nnl2.ffi:%ad-reinterpret tensor shape rank force nnl2.ffi:ad-reverse-mode track-graph)))	
	
(defun .sqrt (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with elementwise square root"
  (nnl2.ffi:%ad-sqrt tensor nnl2.ffi:ad-reverse-mode track-graph))
  
(defun slice (tensor &key from to (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a sliced view of the tensor from `from` to `to` indices"
  
  (let* ((tensor-shape (nnl2.hli.ad:shape tensor :as :vector))
         (from (if from from (make-array (list (length tensor-shape)) :initial-element 0)))
         (to (if to to tensor-shape))
         (processed-to (nnl2.hli.ts:process-to-indices to tensor-shape))
         (pntr-from (nnl2.hli:make-shape-pntr from))
         (pntr-to (nnl2.hli:make-shape-pntr processed-to)))
    
    (nnl2.ffi:%ad-slice tensor pntr-from pntr-to nnl2.ffi:ad-reverse-mode track-graph)))	  
  
(defun l2-norm (tensor &key force (axes #(0)) (track-graph nnl2.system:*ad-default-track-graph*) &aux (dtype (nnl2.hli.ad:dtype tensor)))
  "Computes the L2 norm of the tensor, optionally returning a raw scalar when `force` is true"
  
  (declare (ignore axes))
   
  (let ((out (nnl2.ffi:%ad-l2norm tensor force nnl2.ffi:ad-reverse-mode track-graph)))
    (if force 
	  (let ((cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype)))
	    (cffi:mem-ref out cffi-type))
		
	  out)))
	  
(defun sum (ad-tensor &key axis keepdim force (track-graph nnl2.system:*ad-default-track-graph*) &aux (dtype (nnl2.hli.ad:dtype ad-tensor)))
  "Computes the sum of elements along a given axis or the entire tensor, optionally forcing a scalar result"
  
  (if axis
    (nnl2.ffi:%ad-sum-with-axis ad-tensor axis keepdim nnl2.ffi:ad-reverse-mode track-graph)
	(let ((result (nnl2.ffi:%ad-sum-without-axis ad-tensor force nnl2.ffi:ad-reverse-mode track-graph)))
	  (if force
	    (let ((cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype)))
	      (cffi:mem-ref result cffi-type))
		  
		result))))
  
(defun norm (tensor &key force (axes #(0)) (p :l2) (track-graph nnl2.system:*ad-default-track-graph*))
  "WARNING: YET DOES NOT SUPPORT AXES (W.I.P.)
   
   Applies passed norm to passed norm (available: (:l2))
   
   tensor: Input tensor
   axes (&key): Axes to apply the norm. DOES NOT FULLY WORK YET"
   
  (case p
    (:l2 (l2-norm tensor :axes axes :force force :track-graph track-graph))
	(otherwise (error "Incorrect :p key in norm~%"))))
  
(defun vstack (tensora tensorb &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Vertically stacks two tensors along the first axis"
  (nnl2.ffi:%ad-vstack tensora tensorb nnl2.ffi:ad-reverse-mode track-graph))
  
(defun hstack (tensora tensorb &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Horizontally stacks two tensors along the last axis"
  (nnl2.ffi:%ad-hstack tensora tensorb nnl2.ffi:ad-reverse-mode track-graph))
  
(defun concat (axis tensora tensorb &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Concatenates two tensors along a specified axis"
  (nnl2.ffi:%ad-concat tensora tensorb axis nnl2.ffi:ad-reverse-mode track-graph))
  
(defun view (tensor indices &key (track-graph nnl2.system:*ad-default-track-graph*) force)
  "Returns a reshaped view of the tensor according to `indices`"
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%ad-view tensor shape rank nnl2.ffi:ad-reverse-mode track-graph force)))

(defun tref (tensor indices &key (track-graph nnl2.system:*ad-default-track-graph*) force)
  "Returns a tensor copy at specified indices"
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%ad-tref tensor shape rank nnl2.ffi:ad-reverse-mode track-graph force)))
	
(defun (setf tref) (change-to tensor &rest shape)
  "Sets the values at the specified tref indices to `change-to`"
  (setf (apply #'nnl2.hli.ts:tref (nnl2.hli.ad:data tensor) shape) change-to))		  
  
(defmacro .square (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with each element raised to the power of 2 with automatic differentiation support
   tensor: Input tensor
   track-graph: Controls whether the operation is recorded in the AD graph"
   
  `(nnl2.hli.ad:.^/ad/powf! ,tensor 2.0s0 nnl2.ffi:ad-reverse-mode ,track-graph))  
  
(defmacro .cube (tensor &key (track-graph nnl2.system:*ad-default-track-graph*))
  "Returns a new tensor with each element raised to the power of 3 with automatic differentiation support
   tensor: Input tensor
   track-graph: Controls whether the operation is recorded in the AD graph"

  `(nnl2.hli.ad:.^/ad/powf! ,tensor 3.0s0 nnl2.ffi:ad-reverse-mode ,track-graph))  	  
  
(in-package :nnl2.hli.ad.r.loss)  
  
(defun mse (prediction target &key force (track-graph nnl2.system:*ad-default-track-graph*))
  "Computes Mean-Squared error
  
   Example:
       (nnl2.hli.ts.loss:mse prediction target) -> loss (scalar)"
   
  (let ((out (nnl2.ffi:%ad-mse prediction target force nnl2.ffi:ad-reverse-mode track-graph)))
    (if force 
	  (cffi:mem-ref out :double)
	  out)))  

(defun mae (prediction target &key force (track-graph nnl2.system:*ad-default-track-graph*))
  "Computes Mean-Absolute error
  
   Example:
       (nnl2.hli.ts.loss:mae prediction target) -> loss (scalar)"
   
  (let ((out (nnl2.ffi:%ad-mse prediction target force nnl2.ffi:ad-reverse-mode track-graph)))
    (if force 
	  (cffi:mem-ref out :double)
	  out)))  
  
  