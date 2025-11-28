(in-package :nnl2.lli)

;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/lowlevel-accessors.lisp
;; File: lowlevel-accessors.lisp

;; Contains a low-level interface for all the main functions in ffi-c-core.lisp

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(cffi:defcfun ("nnl2_free_ad_tensor" free) :void
  (ad-tensor :pointer))  
  
(defmacro fast-mem-aref-getter (data index dtype)
  "Fast memory reference getter with type specialization.
   
   data: pointer to tensor data
   index: Memory index to access
   dtype: Data type"
   
  (case dtype
    (:float64 `(nnl2.ffi:mem-aref-getter-float64 ,data ,index))
    (:float32 `(nnl2.ffi:mem-aref-getter-float32 ,data ,index))
    (:int32   `(nnl2.ffi:mem-aref-getter-int32   ,data ,index))))

(defmacro fast-mem-aref-setter (value data index dtype)
  "Fast memory reference setter with type specialization.
   
   value: Value to set
   data: Pointer to tensor data
   index: Memory index to access
   dtype: Data type"
   
  (case dtype
    (:float64 `(nnl2.ffi:mem-aref-setter-float64 ,data ,index ,value))
    (:float32 `(nnl2.ffi:mem-aref-setter-float32 ,data ,index ,value))
    (:int32   `(nnl2.ffi:mem-aref-setter-int32   ,data ,index ,value))))

(defun alignment ()
  "Gets memory allignment"
  (nnl2.ffi:%get-mem-alignment))						  
			  
(declaim (ftype (function () (integer 0 65)) alignment)
		 (inline alignment))

(in-package :nnl2.lli.ts)

(defun flat (tensor at)
  "Retrieves the tensor element value by linear index
   
   tensor: Input tensor
   at: Linear index of the element"
   
  (let* ((dtype (nnl2.hli.ts:type/nnl2->cffi (nnl2.hli.ts:dtype tensor)))
		 (elem (nnl2.ffi:%lowlevel-tref tensor at)))
		 
	(if (cffi:null-pointer-p elem)
	  (error "Pointer can't be NULL")
	  (cffi:mem-ref elem dtype))))
	  
(defun (setf flat) (with tensor at)
  "Sets the tensor element value by linear index
   
   with: New value to set
   tensor: Input tensor 
   at: Linear index of the element"
   
  (let* ((dtype (nnl2.hli.ts:dtype tensor))
         (cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype))    
	     (lisp-type (nnl2.hli.ts:type/nnl2->lisp dtype))
		 (filler-pntr (cffi:foreign-alloc cffi-type)))

   (setf (cffi:mem-ref filler-pntr cffi-type) (coerce with lisp-type))
   
   (nnl2.ffi:%lowlevel-tref-setter tensor at filler-pntr)))
  
(defun trefw (tensor &rest at)
  "Retrieves the tensor element value by coordinate indices
   
   tensor: Input tensor
   at: Variable number of coordinate indices"
   
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr at)
    (let ((elem (nnl2.ffi:%lowlevel-tref-with-coords tensor shape rank)))
	  (if (cffi:null-pointer-p elem)
	    (error "Pointer can't be NULL")
		(cffi:mem-ref elem (nnl2.hli.ts:type/nnl2->cffi (nnl2.hli.ts:dtype tensor)))))))
	    
(defun (setf trefw) (with tensor &rest at)
  "Sets the tensor element value by coordinate indices
   
   with: New value to set
   tensor: Input tensor
   at: Variable number of coordinate indices"
   
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr at)
    (let* ((dtype (nnl2.hli.ts:dtype tensor))
		   (cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype))
		   (lisp-type (nnl2.hli.ts:type/nnl2->lisp dtype))
		   (filler-pntr (cffi:foreign-alloc cffi-type)))

     (setf (cffi:mem-ref filler-pntr cffi-type) (coerce with lisp-type))
	 
	 (nnl2.ffi:%lowlevel-tref-with-coords-setter tensor shape rank filler-pntr))))
		
(defun data (tensor)
  "Returns the raw data pointer of the tensor
   tensor: Input tensor"
   
  (nnl2.ffi:get-tensor-data tensor))
  
(defun mem-aref (tensor-data index &key (dtype :float64))
  "Accesses memory at the specified index with given data type
   
   tensor-data: Pointer to tensor data
   index: Memory index to access
   dtype: Data type"
   
  (cffi:mem-aref tensor-data (nnl2.hli.ts:type/nnl2->cffi dtype)))
  
(defmacro iterate-across-tensor-data ((iterator tensor) &body body)
  "Iterates across all elements of tensor data
   
   iterator: Variable to bind each element value
   tensor: Tensor object
   body: Forms to execute for each element"
   
  `(loop for i from 0 below (nnl2.hli.ts:size ,tensor)
	     do (let ((,iterator (flat ,tensor i)))
		      (progn ,@body))))
			  
(defmacro iatd ((iterator tensor) &body body)
  "Short name for iterate-across-tensor-data (see docstring)"
  `(iterate-across-tensor-data (,iterator ,tensor) ,@body))			

(defmacro parallel-iterate-across-tensor-data ((iterator tensor) &body body)
  "Parallel iterates across all elements of tensor data
   
   iterator: Variable to bind each element value
   tensor: Tensor object
   body: Forms to execute for each element"
   
  `(nnl2.threading:pdotimes (i (nnl2.hli.ts:size ,tensor))
	 (let ((,iterator (flat ,tensor i)))
	   (progn ,@body))))  
	   
(defmacro piatd ((iterator tensor) &body body)
  "Short name for parallel-iterate-across-tensor-data (see docstring)"
  `(parallel-iterate-across-tensor-data (,iterator ,tensor) ,@body))			   
			  
(defmacro iterate-across-tensor-shape ((iterator tensor) &body body)
  "Iterates across all dimensions of tensor shape
   
   iterator: Variable to bind each dimension size
   tensor: Tenosr object
   body: Forms to execute for each dimension"
   
  `(loop for i from 0 below (nnl2.hli.ts:rank ,tensor)
		 do (let ((,iterator (nnl2.ffi:shape-at ,tensor i)))
			  (progn ,@body))))
			  
(defmacro iats ((iterator tensor) &body body)
  "Short name for iterate-across-tensor-shape (see docstring)"
  `(iterate-across-tensor-shape (,iterator ,tensor) ,@body))					  

(defmacro iterate-across-tensor-strides ((iterator tensor) &body body)
  "Iterates across all dimensions of tensor strides
   
   iterator: Variable to bind each dimension size
   tensor: Tenosr object
   body: Forms to execute for each dimension"
   
  `(loop for i from 0 below (nnl2.hli.ts:rank ,tensor)
		 do (let ((,iterator (nnl2.ffi:strides-at ,tensor i)))
			  (progn ,@body))))
			  
(defmacro iatst ((iterator tensor) &body body)
  "Short name for iterate-across-tensor-strides (see docstring)"
  `(iterate-across-tensor-strides (,iterator ,tensor) ,@body))						  
  