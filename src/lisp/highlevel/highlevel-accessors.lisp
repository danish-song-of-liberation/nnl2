(in-package :nnl2.hli)

;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/highlevel-accessors.lisp
;; File: highlevel-accessors.lisp

;; Contains a high-level interface for all the main functions in ffi-c-core.lisp

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defmacro fastcall (&body body)
  "A macro inspired by __fastcall (C) that 
  accelerates the performance of everything 
  inside it"
  
  `(locally (declare (optimize (speed 3)))
     ,@body))
	 
(defun make-shape-pntr (shape)
  "Creates a pointer to a C integers array from a lisp array/list
   shape: A sequence (vector or list) of integers (shape for tensor)
   Returns: (values foreign-pointer length) - A pointer to a C array and its length"
  
  (let* ((len (the (integer 0 *) (length shape)))
	     (shape-pntr (the cffi:foreign-pointer (cffi:foreign-alloc :int :count len))))
		
	(declare (type (integer 0 *) len))
	(declare (type cffi:foreign-pointer shape-pntr))
		
	(etypecase shape	
      (vector
	    (loop for i from 0 below len
              do (setf (cffi:mem-aref shape-pntr :int i) (aref shape i))))
			
	  (list
	    (loop for i from 0 below len
		      do (setf (cffi:mem-aref shape-pntr :int i) (nth i shape)))))
		  
    (values shape-pntr len)))
	
(declaim (ftype (function ((or vector list)) (values cffi:foreign-pointer (integer 0 *))) make-shape-pntr))	

(defmacro free-shape-pntr (pntr)
  "Frees the memory allocated through the make-shape-pntr function
   pntr: Pointer obtained through make-shape-pntr
   
   Example: 
   (let ((foo (make-shape-pntr #(1 2 3))))
     ...
	 
	 (free-shape-pntr foo))
   
   see: #'make-shape-pntr"
   
  `(cffi:foreign-free ,pntr))	
  
(defmacro with-automatic-memory-freeing (&body body)
  "Macro that calls the unwind-protect macro (created for better self-documentation)
  
   Example: 
   (nnl2.hli:with-automatic-memory-freeing
     ...)
	 
   See: unwind-protect"
  
  `(unwind-protect ,@body))
  
(defun reduce-list-shape (lst)
  "Returns the shape of the list
   Example: (nnl2.hli:reduce-list-shape '((0 0 0) (0 0 0))) -> '(2 3)"
   
  (when (consp lst)
    (let ((result '()))
      (loop for current = lst then (car current)
            while (consp current)
            do (push (length current) result))
      (nreverse result))))
	
(defun flatten (lst)
  "Accepts any list and returns it flat
   Example: (nnl2.hli:flatten '((1 2) (3 4))) -> '(1 2 3 4)
   Same speed as alexandria:flatten"

  (let ((result (the list '()))
        (stack (the list (list lst))))
	
	(declare (type list result stack))
		
    (loop while stack 
	      do (let ((current (the list (pop stack))))
			   (loop for elem in current do
				 (if (listp elem)
				   (push (the list elem) stack)
				   (push elem result)))))
				   
    (nreverse (the list result))))

(declaim (ftype (function (list) list) flatten reduce-list-shape))	

(defun reduce-list-size (lst)
  "Calculates the product of the lengths of all nested 
   lists using the first element of each level of nesting"
   
  (let ((size (the (integer 0 *) 1))
        (current lst))
		
	(declare (type (integer 0 *) size))
		
    (loop while (consp current)
          do (setf size (* size (length current))
                   current (car current)))
				   
    (the (integer 0 *) size)))
	
(declaim (inline reduce-list-size)
		 (ftype (function (list) (integer 0 *)) reduce-list-size))
	  
(defun list-to-flat-vector-and-shape (lst total-size)
  "Converts a nested list into a flat vector and calculates its shape
   
   lst: Nested list for conversion
   total-size: The total number of elements in the flattened list
   
   Return: (value flat-data shape) - 1. A flat vector of elements; 2. List of dimensions (shape) of the initial structure
   
   Example: (list-to-flat-vector-and-shape '((1 2) (3 4)) 4) -> (values #(1 2 3 4) (2 2))"
   
  (declare (type (fixnum 0 *) total-size)
		   (type list lst))
		   
  (let ((flat-data (the (simple-array * (*)) (make-array total-size)))
        (shape (the list '()))
        (index (the fixnum 0))
        (current lst))
		
	(declare (type fixnum index)
			 (type list shape)
			 (type (simple-array * (*)) flat-data))
    
	;; Iterate over nested lists, collecting their lengths
    (loop while (consp current)			    ; While the current item is a list
          do (push (length current) shape)  ; Adding the length of the current level to the shape
             (setf current (car current)))  ; Move to the next level of nesting
    
    (setf shape (nreverse shape))
    
    (labels ((fill-array (x)
			   "Recursively fills flat-data with elements from a nested structure
			    x: Current element (list or atom)"
				
               (if (consp x)
			     ;; If the element is a list, process each element recursively
                 (dolist (item x) (fill-array item))
				 ;; If the element is an atom
                 (progn
				   ;; Place the element in the array at the current index
                   (setf (aref flat-data index) x)
				   ;; Increment the index for the next element
                   (setq index (+ index 1))))))		 
			
      ;; Start recursive filling from the root element
      (fill-array lst)
	  
      (values flat-data shape))))	  

;; I know that (integer 0 *) is incorrect, but for some reason it works much faster than fixnum (0.210s+ vs 0.195s-)
(declaim (ftype (function (list (integer 0 *)) (values (simple-array * (*)) list)) list-to-flat-vector-and-shape)) 

(defmacro <- (tensor &rest forms)
  (if (null forms)
      tensor
      (let ((form (first forms))
            (rest (rest forms)))
			
        (if (listp form)
          `(<- ,(if (cdr form)
                  `(,(car form) ,tensor ,@(cdr form))
                  `(,form ,tensor))
            ,@rest)
			
          `(<- (,form ,tensor) ,@rest)))))
		  
(defmacro -> (tensor &rest forms)
  (if (null forms)
      tensor
      (let ((form (first forms))
            (rest (rest forms)))
			
        (if (listp form)
          `(-> ,(if (cdr form)
                  `(,(car form) ,@(cdr form) ,tensor)
                  `(,form ,tensor))
            ,@rest)
			
          `(-> (,form ,tensor) ,@rest)))))
		  
(defun stypeof (obj)
  "Simplified version of type-of (vanilla lisp)
  
   Used to prevent bugs. When passing (stypeof 2), it 
   will return not (INTEGER 0 4629328582) but INTEGER"
  
  (cond
    ((integerp obj) (the symbol 'integer))
	(t (type-of obj))))
	
(declaim (inline stypeof)
		 (ftype (function (atom) symbol) stypeof))

(defun object-type (obj)
  (nnl2.ffi:get-obj-type obj))

(in-package :nnl2.hli.ts)

(deftype nnl2-tensor () 
  "Type representing a foreign tensor pointer"
  
  #+sbcl      'sb-sys:system-area-pointer
  #+clisp     'fi:foreign-data
  #+ccl       'ccl:macptr
  #+ecl       'si:foreign-data
  #+abcl      'system:foreign-pointer
  #+lispworks 'fli:pointer
  #+allegro   'excl:foreign-pointer)

(defparameter *nnl2-tensor-types* '((:float64 . double-float) (:float32 . single-float) (:int32 . integer))
  "All types of nnl2 tensors and lisp types in an associative list")

(defun type/nnl2->lisp (tensor-type)
  "Converts the tensor system type to a lisp type
   tensor-type: type for conversion
   
   Example: (type/nnl2->lisp :float64) -> 'DOUBLE-FLOAT"

  (declare (type keyword tensor-type))
  
  (nnl2.hli:fastcall 
    (case (the keyword tensor-type) 
	  (:float64 (the symbol 'double-float)) 
	  (:float32 (the symbol 'single-float))
	  (:int32 (the symbol 'integer)))))
	
(declaim (ftype (function (keyword) symbol) type/nnl2->lisp)) ;; Inline not needed	
  
(defun type/lisp->nnl2 (lisp-type)
  "Converts a lisp type to a tensor type
   lisp-type: lisp type for conversion into nnl2 tensor type
   
   Example: (type/lisp->nnl2 'double-float) -> :FLOAT64"
   
  (declare (type symbol lisp-type))

  (nnl2.hli:fastcall
    (cond ((eql lisp-type 'double-float) :float64) 
		  ((eql lisp-type 'single-float) :float32) 
		  ((eql lisp-type 'integer)      :int32))))
		  
(declaim (ftype (function (symbol) keyword) type/lisp->nnl2)) ;; Inline not needed			  
	
(defun type/lisp->cffi (lisp-type)
  "Converts a lisp type to a cffi type
   lisp-type: lisp type for conversion into cffi type
   
   Example: (type/lisp->cffi 'double-float) -> :double"
   
  (declare (type symbol lisp-type))

  (nnl2.hli:fastcall
    (cond ((eql lisp-type 'double-float) :double) 
		  ((eql lisp-type 'single-float) :float) 
		  ((eql lisp-type 'integer)      :int))))
		  
(declaim (ftype (function (symbol) keyword) type/lisp->nnl2 type/lisp->cffi)) ;; Inline not needed		
	
(defun type/nnl2->cffi (cffi-type)
  "Converts the tensor system type to a cffi type
   cffi-type: type for conversion
   
   Example: (type/nnl2->cffi :float64) -> :double"

  (declare (type keyword cffi-type))
  
  (nnl2.hli:fastcall 
    (case (the keyword cffi-type) 
	  (:float64 (the keyword :double)) 
	  (:float32 (the keyword :float))
	  (:int32 (the keyword :int)))))	
	  
(declaim (ftype (function (keyword) keyword) type/lisp->cffi)) ;; Inline not needed		  
	
(defmacro free (tensor)
  "Releases the transmitted tensor
   tensor: Input tensor"

  `(nnl2.ffi:free-tensor ,tensor))
  
(declaim (inline free))  
	
(defmacro tlet ((&rest bindings) &body body)
  "nnl2 Own let form for tensors automatically 
   releasing created tensors at the end"
   
  (let ((vars (mapcar #'car bindings)))
    `(let ,bindings
	   (unwind-protect
	       (progn ,@body)
		   
		 (progn 
		   ,@(loop for var in vars
		           collect `(when (typep ,var 'nnl2-tensor) (free ,var))))))))

(defmacro tlet* ((&rest bindings) &body body)
  "The eigenform of let* for nnl2 tensors"
  
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
  "Makes an empty tensor of the specified shape
  
   indices: A list or vector with the dimensions of a tensor
   dtype (key): Type of tensor"

  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%empty shape rank dtype)))
	
(defmacro empty-with-shape-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  "Creates an empty tensor (filled with garbage) 
   using a pre-computed shape (gives a speed boost)
   
   shape-pntr: Pointer to a C array with shape
   rank: Length of shape
   dtype (key): Type of tensor"
  
  `(nnl2.hli:fastcall 
     (nnl2.ffi:%empty ,shape-pntr ,rank ,dtype)))
   
(declaim (inline empty-with-pntr))

(defun zeros (indices &key (dtype nnl2.system:*default-tensor-type*))
  "Creates a tensor filled with zeros
  
   indices: A list or vector with the dimensions of a tensor
   dtype (key): Type of tensor"
   
  (declare (type keyword dtype))		
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
      (declare (type integer rank))
      (nnl2.ffi:%zeros shape rank dtype))))	 

(defmacro zeros-with-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  "Creates an tensor filled with zeros
   using a pre-computed shape (gives a speed boost)
   
   shape-pntr: Pointer to a C array with shape
   rank: Length of shape
   dtype (key): Type of tensor"
   
  `(nnl2.ffi:%zeros ,shape-pntr ,rank ,dtype))

(defun ones (indices &key (dtype nnl2.system:*default-tensor-type*))
  "Creates a tensor filled with ones
  
   indices: A list or vector with the dimensions of a tensor
   dtype (key): Type of tensor"
   
  (declare (type keyword dtype))		
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
      (declare (type integer rank))
      (nnl2.ffi:%ones shape rank dtype))))

(defmacro ones-with-pntr (shape-pntr rank &key (dtype nnl2.system:*default-tensor-type*))
  "Creates a tensor filled with ones
   using a pre-computed shape (gives a speed boost)
   
   shape-pntr: Pointer to a C array with shape
   rank: Length of shape
   dtype (key): Type of tensor"
   
  `(nnl2.ffi:%ones ,shape-pntr ,rank ,dtype))
  
(declaim (ftype (function ((or vector list) &key (:dtype keyword)) nnl2-tensor) empty zeros ones))  

(defun full (indices &key (dtype nnl2.system:*default-tensor-type*) (filler 0.0d0))
  "Creates a tensor filled with a dtype key specified value
   
   indices: A list or vector with the dimensions of a tensor
   dtype (key): Type of tensor
   filler (key): Value to fill tensor
   
   Example:
   (nnl2.hli.ts:full #(5) :filler 2.0d0) -> #<NNL2:TENSOR ...: 2.0d0 2.0d0 2.0d0 2.0d0 2.0d0>"
   
  (declare (type keyword dtype))

  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
      (let* ((cffi-type (the keyword (type/nnl2->cffi dtype)))
		     (lisp-type (the symbol (type/nnl2->lisp dtype)))
	    	 (filler-pntr (cffi:foreign-alloc cffi-type)))
			 
		(declare (type keyword cffi-type))	 
		  
	    (setf (cffi:mem-ref filler-pntr cffi-type) (coerce filler lisp-type))
	  
        (let ((tensor (nnl2.ffi:%full shape rank dtype filler-pntr)))
	      (cffi:foreign-free filler-pntr)		  
		  tensor)))))

(defun from-flatten (flatten-data indices &key (dtype nnl2.system:*default-tensor-type*))
  "Creates a tensor from a flat list/vector with the specified shapes
   Example: (from-flatten '(1 2 3 4 5 6) '(2 3)) -> #<NNL2:TENSOR ... [2x3]: 1.0d0 2.0d0 3.0d0\n4.0d0 5.0d0 6.0d0>
   
   flatten-data: list/vector Data
   indices: Tensor shapes (list/vector)
   dtype (key): Type of tensor" 
   
  (nnl2.hli:fastcall 
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  ;; Creating a pointer to the tensor shape and obtaining the rank 
      (let* ((total-elems (length flatten-data))
		     (cffi-type (the keyword (type/nnl2->cffi dtype)))
		     (lisp-type (the symbol (type/nnl2->lisp dtype)))
		     (data-pntr (cffi:foreign-alloc cffi-type :count total-elems)))
		   
	    (declare (type symbol lisp-type) 
			     (type keyword cffi-type)
			     (type (integer 0 *) rank total-elems))
				 
	    (unwind-protect
	      (progn
            (etypecase flatten-data
	          (list 
			    ;; If the data is in the form of a list, iterate over the elements
	            (let ((lst flatten-data))
                 (nnl2.threading:pdotimes (i total-elems) ; Parallel iteration
                   (setf (cffi:mem-aref data-pntr cffi-type i) 
                         (coerce (car lst) lisp-type)
                         lst (cdr lst)))))
          
		      (vector
			    ;; If the data is in the form of a vector, it is more efficient to process it
                (case cffi-type
                  (:float
                    (nnl2.threading:pdotimes (i total-elems)
                      (setf (cffi:mem-aref data-pntr :float i)
                            (the single-float (coerce (aref flatten-data i) 'single-float)))))
							 
                  (:double
                    (nnl2.threading:pdotimes (i total-elems)
                      (setf (cffi:mem-aref data-pntr :double i)
                            (the double-float (coerce (aref flatten-data i) 'double-float)))))
							 
                  (t	
                    (nnl2.threading:pdotimes (i total-elems)
                      (setf (cffi:mem-aref data-pntr cffi-type i)
                            (coerce (aref flatten-data i) lisp-type)))))))))
	  
	    ;; Creating a tensor from C data and freeing temporary memory
        (let ((result (nnl2.ffi:%make-tensor-from-flatten data-pntr total-elems shape rank dtype)))
		  (cffi:foreign-free data-pntr) ; Freeing up C memory
		  result))))) ; Return of the created tensor

(defun make-tensor (data &key (dtype nnl2.system:*default-tensor-type*))
  "Makes a tensor from the specified data
   Example: (make-tensor #2A((1 2 3) (4 5 6))) or (make-tensor '((1 2 3) (4 5 6)))
   Tip: Try to use vectors instead of lists. This will give you a speed boost of ~2-3+ times"
   
  (etypecase data
    (array
	  (let* ((data-shape (array-dimensions data))
			 ;; Create a flat view of the array without copying data
			 (flat-data (make-array (array-total-size data) 
						  :element-type (array-element-type data) 
						  :displaced-to data)))
		 
		;; Convert flat array to tensor 
	    (from-flatten flat-data data-shape :dtype dtype)))
		
	(list
      (let ((total-size (the (integer 0 *) (nnl2.hli:reduce-list-size data))))
	    (declare (type (integer 0 *) total-size))
		
		;; Convert nested list to flat vector and extract shape information
        (multiple-value-bind (flat-vector shape)
          (nnl2.hli:list-to-flat-vector-and-shape data total-size)
		    ;; Convert the flat vector to tensor
            (from-flatten flat-vector shape :dtype dtype))))))
  
(defmacro full-with-pntr (shape-pntr rank &key filler (dtype nnl2.system:*default-tensor-type*))
   "Creates a tensor filled with specified value
    using a pre-computed shape (gives a speed boost)
   
    shape-pntr: Pointer to a C array with shape
    rank: Length of shape
	filler (key): Value to fill
    dtype (key): Type of tensor"
	
  `(nnl2.ffi:%full ,shape-pntr ,rank ,dtype ,filler))	 

(defun print-tensor (tensor &key full-print)
  "Prints a tensor with formatted output"
  
  (nnl2.ffi:print-tensor 
    tensor 
	full-print 
	nnl2.format:*nnl2-max-rows-format* 
	nnl2.format:*nnl2-max-cols-format* 
	nnl2.format:*nnl2-show-rows-after-skip* 
	nnl2.format:*nnl2-show-cols-after-skip*)
	
  tensor)
  
(cffi:defcfun ("nnl2_get_tensor_rank" rank) :int
  (tensor :pointer))    
  
(cffi:defcfun ("nnl2_get_tensor_dtype" dtype) nnl2.ffi:tensor-type
  (tensor :pointer))      

(cffi:defcfun ("nnl2_get_tensor_dtype" int-dtype) :int
  (tensor :pointer))         
  
(defun get-shape-as-list (tensor rank)
  "Gets the form of the specified tensor as a list
   tensor: Input tensor
   rank: Rank of input tensor"
   
  (loop with rank-t = (if rank rank (rank tensor))
		with shape-pointer = (nnl2.ffi:get-pointer-to-tensor-shape tensor)
		for i from 0 below rank-t
		collect (cffi:mem-aref shape-pointer :int i)))		

(defun get-shape-as-vector (tensor rank)
  "Gets the form of the specified tensor as a vector
   tensor: Input tensor
   rank: Rank of input tensor"
   
  (let* ((rank-t (if rank rank (rank tensor)))
		 (vec (make-array rank-t))
		 (shape-pointer (nnl2.ffi:get-pointer-to-tensor-shape tensor)))
		 
	(dotimes (i rank-t)
	  (setf (aref vec i) (cffi:mem-aref shape-pointer :int i)))
	  
	vec))
	
(declaim (ftype (function (nnl2-tensor (integer 0 *)) list) get-shape-as-list)
		 (ftype (function (nnl2-tensor (integer 0 *)) vector) get-shape-as-vector))
	
(defun shape (tensor &key (as :vector))
  "Function for getting the shape of a tensor
  
   tensor: Input tensor
   as (key): Return type
   
   Available returnable type: (:list :vector :pointer)
   
   Example 1: (shape foo :as :list) -> '(3 3)
   Example 2: (shape foo :as :vector) -> #(3 3)
   Example 3: (shape foo :as :pointer) -> Depends on the lisp implementation"
   
  (let ((rank (rank tensor)))
    (case as
      (:list    (get-shape-as-list tensor rank))
	  (:vector  (get-shape-as-vector tensor rank))
	  (:pointer (nnl2.ffi:get-pointer-to-tensor-shape tensor))
	  (otherwise (error "Unknown type: ~a~%" as)))))
	  
(declaim (inline shape))	  
	
(defun get-strides-as-list (tensor)
  "Gets the strides of the specified tensor as a list
   tensor: Input tensor
   rank: Rank of input tensor"
   
  (loop with len = (rank tensor)
		with strides-pointer = (nnl2.ffi:get-pointer-to-tensor-strides tensor)
		for i from 0 below len
		collect (cffi:mem-aref strides-pointer :int i)))	
	
(defun get-strides-as-vector (tensor)
  "Gets the strides of the specified tensor as a vector
   tensor: Input tensor
   rank: Rank of input tensor"
   
  (let* ((len (rank tensor))
         (vec (make-array len))
		 (strides-pointer (nnl2.ffi:get-pointer-to-tensor-strides tensor)))
		
	(dotimes (i len)
	  (setf (aref vec i) (cffi:mem-aref strides-pointer :int i)))
	  
	vec))
	
(defmacro strides (tensor &key (as :vector))
  "Function for getting the strides of a tensor
  
   tensor: Input tensor
   as (key): Return type
   
   Available returnable type: (:list :vector :pointer)
   
   Example 1: (strides foo :as :list) -> '(5 1)
   Example 2: (strides foo :as :vector) -> #(5 1)
   Example 3: (strides foo :as :pointer) -> Depends on the lisp implementation"
   
  (case as
    (:list    `(get-strides-as-list ,tensor))
    (:vector  `(get-strides-as-vector ,tensor))
	(:pointer `(nnl2.ffi:get-pointer-to-tensor-strides ,tensor))
	(otherwise (error "Unknown type: ~a~%" as))))

(defun gemm (a b &key (order :nnl2rowmajor) (transa :nnl2notrans) (transb :nnl2notrans) (alpha 1.0d0) (beta 0.0d0) m n k lda ldb)
  "General Matrix Multiplication (GEMM) operation
   C = alpha * op(A) * op(B) + beta * C
   
   Args:
   a, b: Input tensors
   order: Storage order (:nnl2rowmajor or :nnl2colmajor)
   transa, transb: Transpose flags (:nnl2notrans or :nnl2trans)
   alpha, beta: Scaling factors
   m, n, k: Matrix dimensions (automatically determined if not provided)
   lda, ldb: Leading dimensions (automatically determined if not provided)"
   
  (declare (optimize (speed 3) (safety 1)))
  
  (declare (type nnl2-tensor a b)
		   (type keyword order transa transb)
		   (type double-float alpha beta))

  (let* ((shape-a (shape a :as :vector))
		 (shape-b (shape b :as :vector))
		 
		 (m (or m (if (eq transa :nnl2notrans) 
                      (aref shape-a 0) 
                      (aref shape-a 1))))
					  
		 (n (or n (if (eq transb :nnl2notrans) 
                      (aref shape-b 1) 
                      (aref shape-b 0))))
					  
		 (k (or k (if (eq transa :nnl2notrans) 
                      (aref shape-a 1) 
                      (aref shape-a 0))))
					  
		 (lda (or lda (if (eq transa :nnl2notrans) k m)))
         (ldb (or ldb (if (eq transb :nnl2notrans) n k))))
		 
	(nnl2.ffi:%gemm order transa transb m n k alpha a lda b ldb beta)))
  
(defun gemm! (a b &key (out a) (order :nnl2rowmajor) (transa :nnl2notrans) (transb :nnl2notrans) (alpha 1.0d0) (beta 0.0d0) m n k lda ldb ldc)
  "General Matrix Multiplication (GEMM) operation with in-place modification
   C = alpha * op(A) * op(B) + beta * C
   
   Args:
   a, b: Input tensors
   out: Output tensor (defaults to a for in-place operation)
   order: Storage order (:nnl2rowmajor or :nnl2colmajor)
   transa, transb: Transpose flags (:nnl2notrans or :nnl2trans)
   alpha, beta: Scaling factors
   m, n, k: Matrix dimensions (automatically determined if not provided)
   lda, ldb, ldc: Leading dimensions (automatically determined if not provided)"
   
  (declare (optimize (speed 3) (safety 1)))
  
  (declare (type nnl2-tensor a b out)
           (type keyword order transa transb)
           (type double-float alpha beta))

  (let* ((shape-a (shape a :as :vector))
         (shape-b (shape b :as :vector))
         (shape-out (shape out :as :vector))
         
         (m (or m (if (eq transa :nnl2notrans) 
                     (aref shape-a 0) 
                     (aref shape-a 1))))
                     
         (n (or n (if (eq transb :nnl2notrans) 
                     (aref shape-b 1) 
                     (aref shape-b 0))))
                     
         (k (or k (if (eq transa :nnl2notrans) 
                     (aref shape-a 1) 
                     (aref shape-a 0))))
                     
         (lda (or lda (if (eq transa :nnl2notrans) k m)))
         (ldb (or ldb (if (eq transb :nnl2notrans) n k)))
         (ldc (or ldc (aref shape-out 1))))
         
    (nnl2.ffi:%gemm! order transa transb m n k alpha a lda b ldb beta out ldc)))      

(declaim (inline gemm gemm!))

(defun .exp! (tensor)
  "Applies the exponent to the tensor in place
   tensor: Input tensor"
   
  (nnl2.ffi:%.exp! tensor))

(defun .exp (tensor &key save-type)
  "Applies the exponent to the tensor
   tensor: Input tensor
   save-type (key): Try to preserve the type as much as possible (for example, for int32)"
   
  (nnl2.ffi:%.exp tensor save-type))  
  
(defun .log! (tensor)
  "Applies the natural logarithm to the tensor in place
   tensor: Input tensor"
  
  (nnl2.ffi:%.log! tensor))  
  
(defun .log (tensor &key save-type)
  "Applies the natural logarithm to the tensor
   tensor: Input tensor
   save-type (key): Try to preserve the type as much as possible (for example, for int32)"
   
  (nnl2.ffi:%.log tensor save-type))    
    
(cffi:defcfun ("get_size" size) :int
  (tensor :pointer))  

(cffi:defcfun ("get_size_in_bytes" size-in-bytes) :int
  (tensor :pointer))  
  
(defun get-scalar-value (pntr dtype)
  "Extracts a scalar value from a pointer, depending on the data type
   pntr: Value to extract
   dtype: Type of value"
   
  (case dtype
    (:float64 (cffi:mem-ref pntr :double))
    (:float32 (cffi:mem-ref pntr :float))
    (:int32 (cffi:mem-ref pntr :int))
    (otherwise (error "Unsupported type: ~A" dtype))))  
  
(defun view (tensor &rest shape)
  "Creates a tensor representation with a new shape 
   Returns a pointer or a scalar value
   tensor: Input tensor
   shape (&rest): Indices to extract subtensor/scalar"
   
  (nnl2.hli:fastcall
    (let* ((shape-rank (length shape))
           (tensor-rank (rank tensor))
           (tensor-dtype (dtype tensor))
           (shape-pntr (nnl2.hli:make-shape-pntr (subst -1 '* shape)))
           (void-ptr (nnl2.ffi:%view tensor shape-pntr shape-rank)))     
    
      (cond
        ((cffi:null-pointer-p void-ptr)
          (error "Couldn't create tensor representation"))
      
        ((= shape-rank tensor-rank)
          (get-scalar-value void-ptr tensor-dtype))
      
        (t void-ptr)))))
		
(defun tref (tensor &rest shape)
  "Extracts a copy of the tensor from the selected indices
  
   tensor: Input tensor
   shape: Input indices"
   
  (nnl2.hli:fastcall
    (let* ((shape-rank (length shape))
		   (tensor-rank (rank tensor))
	       (tensor-dtype (dtype tensor))
		   (shape (nnl2.hli:make-shape-pntr (subst -1 '* shape)))
		   (void-ptr (nnl2.ffi:%tref-getter tensor shape shape-rank)))	 
	
	  (cond 
	    ((cffi:null-pointer-p void-ptr)
		  (error "Couldn't create tensor representation"))
		  
		((= shape-rank tensor-rank)		
	      (get-scalar-value void-ptr tensor-dtype))
	  
	    (t void-ptr)))))		
	  
(defun (setf tref) (change-to tensor &rest shape)
  "Sets a new value for the tensor at the specified indices
   change-to: New value (scalar or tensor)
   tensor: Input tensor
   shape: Indices"
   
  (nnl2.hli:fastcall
    (let* ((shape-rank (length shape))	
		   (shape (nnl2.hli:make-shape-pntr (subst -1 '* shape)))
	       (tensor-dtype (dtype tensor))
		   (is-tensor-p (typep change-to 'nnl2-tensor)))
		 
	  (if is-tensor-p 
	    (nnl2.ffi:%tref-setter tensor shape shape-rank change-to t)
		(let* ((cffi-type (type/nnl2->cffi tensor-dtype))
			   (lisp-type (type/nnl2->lisp tensor-dtype))
			   (changer (cffi:foreign-alloc cffi-type)))
			   
		  (setf (cffi:mem-ref changer cffi-type) (coerce change-to lisp-type))
		  (nnl2.ffi:%tref-setter tensor shape shape-rank changer nil))))))
		  
(defmacro scale! (tensor multiplier)
  "Increases the tensor in-place by a multiplier
   tensor: Input tensor
   multiplier: Tensor is multiplied by"
   
  `(nnl2.ffi:%scale! ,tensor (float ,multiplier)))

(defmacro scale (tensor multiplier &key save-type)
  "Increases the tensor by a multiplier
   tensor: Input tensor
   multiplier: Tensor is multiplied by"
   
  `(nnl2.ffi:%scale ,tensor (float ,multiplier) ,save-type))  
  
(cffi:defcfun ("nnl2_empty_like" empty-like) :pointer
  (tensor :pointer))   
  
(cffi:defcfun ("nnl2_zeros_like" zeros-like) :pointer
  (tensor :pointer)) 
	
(cffi:defcfun ("nnl2_ones_like" ones-like) :pointer
  (tensor :pointer))   
		
(defun full-like (tensor &key (filler 0.0d0))
  "Creates a tensor filled with the specified numbers 
   of the same type and shape as the passed tensor
   
   tensor: Tensor from which to take the shape and type
   filler (&key): Value to fill new tensor"
   
  (let* ((dtype (dtype tensor))
		 (lisp-type (type/nnl2->lisp dtype))
		 (keyword-type (type/nnl2->cffi dtype))
		 (filler-pntr (cffi:foreign-alloc keyword-type))) 
		 
    (setf (cffi:mem-ref filler-pntr keyword-type) (coerce filler lisp-type))
	
    (let ((result (nnl2.ffi:%full-like tensor filler-pntr)))
      (cffi:foreign-free filler-pntr)
		result)))
		
(defun xavier-like (tensor &key in out (gain 1.0s0) (distribution :normal))
  "Creates a new tensor of the same shape and type as the input tensor,
   initialized using Xavier/Glorot initialization method
   
   tensor: The input tensor from which shape and data type are derived
   in (&key) (default - nil): Number of input units
   out (&key) (default - nil): Number of output units
   gain (&key) (default - 1.0s0): Scaling factor 
   distribution (&key) (default - :normal): Type of distribution (:normal/:uniform)"
   
  (assert (and in out gain distribution) nil "Incorrect keys was passed in xavier-like")
  (nnl2.ffi:%xavier-like tensor in out gain (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))))
  
(defun .abs! (tensor)
  "Applies the modulus of a number to a tensor in-place"
  (nnl2.ffi:%.abs! tensor))  
  
(defun .abs (tensor)
  "Applies the modulus of a number to a tensor"
  (nnl2.ffi:%.abs tensor))    

(defun .map! (funct &rest tensors &aux (first-tensor (car tensors)))
  "Applies the passed function to the first tensor element-wise
   
   funct: Function to apply
   tensors (&rest): Tensors for element-wise transmission to a function"
   
  (let ((aggreg-data (mapcar #'nnl2.ffi:get-tensor-data tensors))
        (numel (size first-tensor))
        (type-t (dtype first-tensor)))
        
    (ecase type-t    
      (:float32 (.map!-process-tensors funct aggreg-data numel #'nnl2.ffi:mem-aref-getter-float32 #'nnl2.ffi:mem-aref-setter-float32))
      (:float64 (.map!-process-tensors funct aggreg-data numel #'nnl2.ffi:mem-aref-getter-float64 #'nnl2.ffi:mem-aref-setter-float64))
      (:int32   (.map!-process-tensors funct aggreg-data numel #'nnl2.ffi:mem-aref-getter-int32   #'nnl2.ffi:mem-aref-setter-int32)))))

(defun .map!-process-tensors (funct aggreg-data numel getter setter)
  "Process tensors with the given getter/setter functions"
  
  (let ((ntensors (length aggreg-data))
        (data0 (car aggreg-data)))
    
    (ecase ntensors
      (1 (.map!-process-single-tensor funct data0 numel getter setter))
      (2 (.map!-process-double-tensor funct data0 (cadr aggreg-data) numel getter setter))
      (t (.map!-process-multiple-tensors funct aggreg-data numel getter setter)))))

(defun .map!-process-single-tensor (funct data0 numel getter setter)
  "Process single tensor case"
  
  (nnl2.threading:pdotimes (i numel)
    (funcall setter data0 i (funcall funct (funcall getter data0 i)))))

(defun .map!-process-double-tensor (funct data0 data1 numel getter setter)
  "Process two tensors case"
  
  (nnl2.threading:pdotimes (i numel)
    (funcall setter data0 i (funcall funct (funcall getter data0 i) (funcall getter data1 i)))))

(defun .map!-process-multiple-tensors (funct aggreg-data numel getter setter)
  "Process multiple tensors case"
  
  (let ((data0 (car aggreg-data)))
    (nnl2.threading:pdotimes (i numel)
      (funcall setter data0 i 
               (apply funct (loop for it in aggreg-data
                                collect (funcall getter it i)))))))						  
																			 
(defun .map (funct &rest tensors &aux (first-tensor (car tensors)))
  "Applies the passed function to tensors element-wise and returns new tensor
   
   funct: Function to apply
   tensors (&rest): Tensors for element-wise transmission to a function"
   
  (let* ((aggreg-data (mapcar #'nnl2.ffi:get-tensor-data tensors))
         (numel (size first-tensor))
         (type-t (dtype first-tensor))
         (new-tensor (empty-like first-tensor))
         (new-tensor-data (nnl2.ffi:get-tensor-data new-tensor)))
        
    (ecase type-t    
      (:float32 (.map-process-tensors funct aggreg-data new-tensor-data numel #'nnl2.ffi:mem-aref-getter-float32 #'nnl2.ffi:mem-aref-setter-float32))
      (:float64 (.map-process-tensors funct aggreg-data new-tensor-data numel #'nnl2.ffi:mem-aref-getter-float64 #'nnl2.ffi:mem-aref-setter-float64))
      (:int32   (.map-process-tensors funct aggreg-data new-tensor-data numel #'nnl2.ffi:mem-aref-getter-int32   #'nnl2.ffi:mem-aref-setter-int32)))
    
    new-tensor))

(defun .map-process-tensors (funct aggreg-data new-tensor-data numel getter setter)
  "Process tensors with the given getter/setter functions and write to new tensor"
  
  (let ((ntensors (length aggreg-data)))
    
    (ecase ntensors
      (1 (.map-process-single-tensor funct aggreg-data new-tensor-data numel getter setter))
      (2 (.map-process-double-tensor funct aggreg-data new-tensor-data numel getter setter))
      (t (.map-process-multiple-tensors funct aggreg-data new-tensor-data numel getter setter)))))

(defun .map-process-single-tensor (funct aggreg-data new-tensor-data numel getter setter)
  "Process single tensor case"
  
  (let ((data0 (car aggreg-data)))
    (nnl2.threading:pdotimes (i numel)
      (funcall setter new-tensor-data i (funcall funct (funcall getter data0 i))))))

(defun .map-process-double-tensor (funct aggreg-data new-tensor-data numel getter setter)
  "Process two tensors case"
  
  (let ((data0 (car aggreg-data))
        (data1 (cadr aggreg-data)))
    (nnl2.threading:pdotimes (i numel)
      (funcall setter new-tensor-data i (funcall funct (funcall getter data0 i) (funcall getter data1 i))))))

(defun .map-process-multiple-tensors (funct aggreg-data new-tensor-data numel getter setter)
  "Process multiple tensors case"
  
  (nnl2.threading:pdotimes (i numel)
    (funcall setter new-tensor-data i 
             (apply funct (loop for it in aggreg-data
                              collect (funcall getter it i))))))
	
(defun /map! (funct &rest tensors &aux (first-tensor (car tensors)))
  "Applies the function to each slice of the tensor in place (applies the changes in first tensor)
  
   funct: Function to apply
   tensors (&rest): Input tensors"
   
  (let ((first-shape (aref (shape first-tensor :as :vector) 0))
		(type-t (type/nnl2->cffi (dtype first-tensor))))
		
	(nnl2.threading:pdotimes (i first-shape)
	  (setf
        (tref first-tensor i) (apply funct (loop for it in tensors
												 collect (tref it i)))))))
														
(defun /map (funct &rest tensors &aux (first-tensor (car tensors)))
  "Applies the function to each slice of the tensor
  
   funct: Function to apply
   tensors (&rest): Input tensors"

  (let ((first-shape (aref (shape first-tensor :as :vector) 0))
		(type-t (type/nnl2->cffi (dtype first-tensor)))
		(new-tensor (empty-like first-tensor)))
		
	(nnl2.threading:pdotimes (i first-shape)
	  (setf
        (tref new-tensor i) (apply funct (loop for it in tensors
											       collect (tref it i)))))

	new-tensor))

(defun hstack (&rest tensors) "Combines the transmitted tensors horizontally" (reduce #'nnl2.ffi:%hstack tensors))	
(defun vstack (&rest tensors) "Combines the transmitted tensors verticallys" (reduce #'nnl2.ffi:%vstack tensors))	

(defun concat (axis &rest tensors) 
  "Concatenates tensors by the specified dimension
   axis: Dimension to concat
   tensors (&rest): Input tensors"
   
  (reduce #'(lambda (acc tensor) (nnl2.ffi:%concat acc tensor axis)) tensors))
  
(declaim (ftype (function (&rest nnl2-tensor) nnl2-tensor) hstack vstack) 
		 (ftype (function ((integer 0 *) &rest nnl2-tensor) nnl2-tensor) concat))  

(cffi:defcfun ("lisp_call_reluinplace" .relu!) :void
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_relu" .relu) :pointer
  (tensor :pointer)) 

(cffi:defcfun ("lisp_call_leakyreluinplace" %.leaky-relu!) :void
  (tensor :pointer)
  (alpha :float))  
  
(defun .leaky-relu! (tensor &key (alpha nnl2.system:*leaky-relu-default-shift*)) 
  "Elemently-wise applies an leaky relu to passed tensor in-place
   tensor: Input tensor
   alpha (&key): Leaky relu shift"
   
  (%.leaky-relu! tensor (coerce alpha 'single-float))) 
  
(cffi:defcfun ("lisp_call_leakyrelu" %.leaky-relu) :pointer
  (tensor :pointer)
  (alpha :float)
  (save-type :bool))  
  
(defun .leaky-relu (tensor &key (alpha nnl2.system:*leaky-relu-default-shift*) (save-type t))
  "Elemently-wise applies an leaky relu to passed tensor
   tensor: Input tensor
   alpha (&key): Leaky relu shift
   save-type (&key): Try to preserve the data type as much as possible (for example, by applying leaky relu to int32)"
   
  (%.leaky-relu tensor (coerce alpha 'single-float) save-type))

(cffi:defcfun ("lisp_call_sigmoidinplace" %.sigmoid!) :void
  (tensor :pointer)
  (approx :bool))
  
(defun .sigmoid! (tensor &key (approx t))
  (%.sigmoid! tensor approx))
  
(cffi:defcfun ("lisp_call_sigmoid" %.sigmoid) :pointer
  (tensor :pointer)
  (approx :bool))
  
(defun .sigmoid (tensor &key (approx t))
  (%.sigmoid tensor approx))  
  
(cffi:defcfun ("lisp_call_tanhinplace" %.tanh!) :void
  (tensor :pointer)
  (approx :bool))
  
(defun .tanh! (tensor &key (approx t))
  (%.tanh! tensor approx))  
  
(cffi:defcfun ("lisp_call_tanh" %.tanh) :pointer
  (tensor :pointer)
  (approx :bool))

(defun .tanh (tensor &key (approx t))
  (%.tanh tensor approx))  
  
(defun %internal-rand (indices dtype from to)
  "Creates a tensor of the specified shape from random numbers
   indices: Input shape
   dtype: Type of tensor
   from: Value to fill from
   to: Value to fill to"
   
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (let* ((cffi-type (type/nnl2->cffi dtype))
		   (lisp-type (type/nnl2->lisp dtype))
		   (coerced-from (coerce from lisp-type))
		   (coerced-to   (coerce to lisp-type))
	       (from-pntr (cffi:foreign-alloc cffi-type))
		   (to-pntr   (cffi:foreign-alloc cffi-type)))
		  
	  (setf (cffi:mem-ref from-pntr cffi-type) coerced-from
			(cffi:mem-ref to-pntr 	cffi-type) coerced-to)
			
	  (nnl2.ffi:%randn shape rank dtype from-pntr to-pntr))))
	  
(defun %internal-rand-inplace (tensor from to &key type-hint)
  "Fills a tensor with random numbers
   indices: Input tensor
   from: Value to fill from
   to: Value to fill to
   type-hint (&key): You can hint a type to optimal performance"
   
  (let* ((dtype (if type-hint type-hint (dtype tensor)))
		 (cffi-type (type/nnl2->cffi dtype))
		 (lisp-type (type/nnl2->lisp dtype))
		 (coerced-from (coerce from lisp-type))
		 (coerced-to   (coerce to lisp-type))
		 (from-pntr (cffi:foreign-alloc cffi-type))
		 (to-pntr   (cffi:foreign-alloc cffi-type)))
		 
	(setf (cffi:mem-ref from-pntr cffi-type) coerced-from
		  (cffi:mem-ref to-pntr   cffi-type) coerced-to)
		  
	(nnl2.ffi:%randn-inplace tensor from-pntr to-pntr)))
	  
(defun randn (indices &key (dtype nnl2.system:*default-tensor-type*) (from -1) (to 1))
  "Creates a tensor of the specified shape filled with random numbers from -1 to 1
   indices: Input shape
   dtype (&key) (default: nnl2.system:*default-tensor-type*): Type of tensor
   from (&key) (default: -1): Value to fill from
   to (&key) (default: 1): Value to fill to"
   
   (%internal-rand indices dtype from to))
   
(defun rand (indices &key (dtype nnl2.system:*default-tensor-type*) (from 0) (to 1))
  "Creates a tensor of the specified shape filled with random numbers from 0 to 1
   indices: Input shape
   dtype (&key) (default: nnl2.system:*default-tensor-type*): Type of tensor
   from (&key) (default: 0): Value to fill from
   to (&key) (default: 1): Value to fill to"
   
   (%internal-rand indices dtype from to))   
  
(defun randn! (tensor &key type-hint (from -1) (to 1))
  "Fills an existing tensor with random numbers from [-1, 1]
   tensor: Input tensor
   type-hint (&key): You can hint a type to optimal performance
   from (&key) (default: -1): Value to fill from
   to (&key) (default: 1): Value to fill to"
   
  (%internal-rand-inplace tensor from to :type-hint type-hint))
   
(defun rand! (tensor &key type-hint (from 0) (to 1))
  "Fills an existing tensor with random numbers from [0, 1]
   tensor: Input tensor
   type-hint (&key): You can hint a type to optimal performance
   from (&key) (default: -1): Value to fill from
   to (&key) (default: 1): Value to fill to"
   
  (%internal-rand-inplace tensor from to :type-hint type-hint))

(defun %internal-randn-like (tensor from to dtype)
  "Сreates a tensor filled with random numbers of the same shape as the passed tensor
   tensor: Input tensor
   dtype: Type of new tensor
   from: Value to fill from
   to: Value to fill to"
   
  (let* ((cffi-type (type/nnl2->cffi dtype))
         (lisp-type (type/nnl2->lisp dtype))
		 (coerced-from (coerce from lisp-type))
		 (coerced-to (coerce to lisp-type))
         (from-pntr (cffi:foreign-alloc cffi-type))
		 (to-pntr (cffi:foreign-alloc cffi-type)))
					   
	(setf (cffi:mem-ref from-pntr cffi-type) coerced-from
		  (cffi:mem-ref to-pntr cffi-type) coerced-to)
						 
	(nnl2.ffi:%randn-like tensor from-pntr to-pntr)))
	
(defun randn-like (tensor &key (from -1.0d0) (to 1.0d0) (dtype (dtype tensor)))
  "Сreates a tensor filled with random numbers from -1 to 1 of the same shape as the passed tensor
   tensor: Input tensor
   dtype (&key) (default: (nnl2.hli.ts:dtype tensor)): Type of new tensor
   from (&key) (default: -1.0d0): Value to fill from
   to (&key) (default: 1.0d0): Value to fill to"
   
  (%internal-randn-like tensor from to dtype))
  
(defun rand-like (tensor &key (from 0.0d0) (to 1.0d0) (dtype (dtype tensor)))
  "Сreates a tensor filled with random numbers from 0 to 1 of the same shape as the passed tensor
   tensor: Input tensor
   dtype (&key) (default: (nnl2.hli.ts:dtype tensor)): Type of new tensor
   from (&key) (default: 0.0d0): Value to fill from
   to (&key) (default: 1.0d0): Value to fill to"
   
  (%internal-randn-like tensor from to dtype))  
				 
(defun xavier (indices &key (dtype nnl2.system:*default-tensor-type*) (in 0) (out 0) (gain 1.0s0) (distribution :normal))
  "Makes a tensor with the xavier distribution in the specified shape
   indices: Input shape (vector/list)
   dtype (&key): Tensor Type
   in (&key): Number of inputs
   out (&key): Number of outputs
   gain (&key): Autologous
   distribution (&key): Available: (:normal :uniform)"

  (progn
    (assert (not (or (zerop in) (minusp in))) nil "Bad `in` was passed to xavier")
	(assert (not (or (zerop out) (minusp out))) nil "Bad `out` was passed to xavier")
  
    (multiple-value-bind (shape rank) (make-shape-pntr indices)
      (case distribution
	    (:normal   (nnl2.ffi:%xavier shape rank dtype in out gain 2.0s0))
	    (:uniform  (nnl2.ffi:%xavier shape rank dtype in out gain 6.0s0))
	    (otherwise (error "Unknown xavier distribution: ~a%" distribution))))))
		
(defun xavier! (tensor &key (in 0) (out 0) (gain 1.0s0) (distribution :normal))
  "Fills a tensor with the xavier distribution in the specified shape in place
   tensor: Input tensor (vector/list)
   in (&key): Number of inputs
   out (&key): Number of outputs
   gain (&key): Autologous
   distribution (&key): Available: (:normal :uniform)"
   
  (assert (not (or (zerop in) (minusp in))) nil "Bad `in` was passed to xavier!")
  (assert (not (or (zerop out) (minusp out))) nil "Bad `out` was passed to xavier!")
  
  (case distribution
    (:normal   (nnl2.ffi:%xavier-inplace tensor in out gain 2.0s0))
    (:uniform  (nnl2.ffi:%xavier-inplace tensor in out gain 6.0s0))
	(otherwise (error "Unknown xavier distribution: ~a%" distribution))))
  
(defun transpose (tensor &key force)
  (nnl2.ffi:%transpose tensor force))
  
(defun transpose! (tensor &key force)
  (nnl2.ffi:%transpose! tensor force))
  
(defun sum (tensor &key axis keepdim &aux (dtype (dtype tensor)))
  "WARNING: YET NOT SUPPORT MULTIPLE AXES
  
   Summarizes all tensor elements 
   along the selected axis
    
   tensor: Input tensor
   axis (&key): Axis to sum
   keepdim (&key): Autologous"
   
  (assert (not (and (not axis) keepdim)) nil 
    (format nil "[nnl2] In call `~a`: Cannot keep dimension in sum. Maybe you didn't specify the axis?" `(|sum| ,tensor |:keepdim| |t|)))
   
  (let* ((type-t (type/nnl2->cffi dtype))
		 (out (cffi:foreign-alloc type-t)))
					
	(if axis				
	  (nnl2.ffi:%sum-with-axis tensor axis keepdim)
	  (progn
        (nnl2.ffi:%sum-without-axis tensor out)
        (let ((result (cffi:mem-ref out type-t)))
          (cffi:foreign-free out) 
          result))))) 
	
(defun l2-norm (tensor &key (axes #(0)) &aux (dtype (dtype tensor)))
  "WARNING: YET DOES NOT SUPPORT AXES (W.I.P.)
   
   Applies l2 norm to passed tensor
  
   tensor: Input tensor
   axes (&key): Axes to apply the norm. DOES NOT WORK YET"
   
  (let* ((type-t (type/nnl2->cffi dtype))
		 (axes-pntr (nnl2.hli:make-shape-pntr axes))
		 (axes-len (length axes))
		 (out (cffi:foreign-alloc type-t)))
					
	(nnl2.ffi:%l2norm tensor axes-pntr axes-len out)
				
	(let ((result (cffi:mem-ref out type-t)))
	  (cffi:foreign-free out)
	  result)))
	
(defun norm (tensor &key (axes #(0)) (p :l2))
  "WARNING: YET DOES NOT SUPPORT AXES (W.I.P.)
   
   Applies passed norm to passed norm (available: (:l2))
   
   tensor: Input tensor
   axes (&key): Axes to apply the norm. DOES NOT FULLY WORK YET"
   
  (case p
    (:l2 (l2-norm tensor :axes axes))
	(otherwise (error "Incorrect :p key in norm~%"))))
	
(defun copy (tensor &key dtype)	
  "Copies the passed tensor
   tensor: Tensor to copy
   dtype (&key): Type of new tensor"
   
  (nnl2.ffi:%copy tensor (if dtype dtype (dtype tensor))))	
  
(defun .+/incf! (tensor increment)
  "Applies in-place addition to a tensor with a scalar
   tensor: Input tensor
   increment: Value to increment"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
	
	(nnl2.ffi:%.+/incf! tensor incf-pntr)))

(defun .+/incf (tensor increment)
  "Applies addition to a tensor with a scalar, returning a new tensor
   tensor: Input tensor
   increment: Value to increment"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce increment lisp-dtype))
	
	(nnl2.ffi:%.+/incf tensor incf-pntr)))
	
(defun .-/decf! (tensor decrement)
  "Applies in-place subtraction to a tensor with a scalar
   tensor: Input tensor
   decrement: Value to decrement"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce decrement lisp-dtype))
	
	(nnl2.ffi:%.-/decf! tensor incf-pntr)))	
	
(defun .-/decf (tensor decrement)
  "Applies subtraction to a tensor with a scalar, returning a new tensor
   tensor: Input tensor
   decrement: Value to decrement"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce decrement lisp-dtype))
	
	(nnl2.ffi:%.-/decf tensor incf-pntr)))	
	
(defun .*/mulf! (tensor multiplier)
  "Applies in-place multiplication to a tensor with a scalar
   tensor: Input tensor
   multiplier: Value to multiply"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
	
	(nnl2.ffi:%.*/mulf! tensor incf-pntr)))		
	
(defun .*/mulf (tensor multiplier)
  "Applies multiplication to a tensor with a scalar, returning a new tensor
   tensor: Input tensor
   multiplier: Value to multiply"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (incf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref incf-pntr cffi-dtype) (coerce multiplier lisp-dtype))
	
	(nnl2.ffi:%.*/mulf tensor incf-pntr)))	

(defun .//divf! (tensor divf)
  "Applies in-place division to a tensor with a scalar
   tensor: Input tensor
   divf: Value to divide by"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (divf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divf lisp-dtype))
	
	(nnl2.ffi:%.//divf! tensor divf-pntr)))		

(defun .//divf (tensor divf)
  "Applies division to a tensor with a scalar, returning a new tensor
   tensor: Input tensor
   divf: Value to divide by"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (divf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref divf-pntr cffi-dtype) (coerce divf lisp-dtype))
	
	(nnl2.ffi:%.//divf tensor divf-pntr)))	

(defun .^/powf! (tensor powf)
  "Applies in-place exponentiation to a tensor with a scalar
   tensor: Input tensor
   powf: Exponent value"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (powf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce powf lisp-dtype))
	
	(nnl2.ffi:%.^/powf! tensor powf-pntr)))

(defun .^/powf (tensor powf)
  "Applies exponentiation to a tensor with a scalar, returning a new tensor
   tensor: Input tensor
   powf: Exponent value"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (powf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref powf-pntr cffi-dtype) (coerce powf lisp-dtype))
	
	(nnl2.ffi:%.^/powf tensor powf-pntr)))	
	
(defun .max/maxf! (tensor maxf)
  "Applies in-place maximum operation to a tensor with a scalar
   tensor: Input tensor
   maxf: Maximum value threshold"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce maxf lisp-dtype))
	
	(nnl2.ffi:%.max/maxf! tensor maxf-pntr)))
	
(defun .max/maxf (tensor maxf)
  "Applies maximum operation to a tensor with a scalar, returning a new tensor
   tensor: Input tensor
   maxf: Maximum value threshold"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (maxf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref maxf-pntr cffi-dtype) (coerce maxf lisp-dtype))
	
	(nnl2.ffi:%.max/maxf tensor maxf-pntr)))	
	
(defun .min/minf! (tensor minf)
  "Applies in-place minimum operation to a tensor with a scalar
   tensor: Input tensor
   minf: Minimum value threshold"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (minf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce minf lisp-dtype))
	
	(nnl2.ffi:%.min/minf! tensor minf-pntr)))	
	
(defun .min/minf (tensor minf)
  "Applies minimum operation to a tensor with a scalar, returning a new tensor
   tensor: Input tensor
   minf: Minimum value threshold"
   
  (let* ((dtype (dtype tensor))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (minf-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref minf-pntr cffi-dtype) (coerce minf lisp-dtype))
	
	(nnl2.ffi:%.min/minf tensor minf-pntr)))

(defun axpy/axpf! (summand sumend alpha)
  "Applies in-place AXPY operation: summand = summand + alpha * sumend
   summand: Input tensor to be updated
   sumend: Tensor to add
   alpha: Scalar multiplier"
   
  (let* ((dtype (dtype summand))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (sumend-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref sumend-pntr cffi-dtype) (coerce sumend lisp-dtype))
	
	(nnl2.ffi:%axpy/axpf! summand sumend-pntr alpha)))	
	
(defun axpy/axpf (summand sumend alpha)
  "Applies AXPY operation, returning a new tensor: result = summand + alpha * sumend
   summand: Input tensor
   sumend: Tensor to add
   alpha: Scalar multiplier"
   
  (let* ((dtype (dtype summand))
		 (cffi-dtype (type/nnl2->cffi dtype))
		 (lisp-dtype (type/nnl2->lisp dtype))
		 (sumend-pntr (cffi:foreign-alloc cffi-dtype)))
		 
	(setf (cffi:mem-ref sumend-pntr cffi-dtype) (coerce sumend lisp-dtype))
	
	(nnl2.ffi:%axpy/axpf summand sumend-pntr alpha)))
	
(cffi:defcfun ("lisp_call_add_broadcasting_inplace" .+/broadcasting!) :void  
  (summand :pointer)
  (sumend :pointer))
  
(cffi:defcfun ("lisp_call_add_broadcasting" .+/broadcasting) :pointer
  (summand :pointer)
  (sumend :pointer))

(cffi:defcfun ("lisp_call_sub_broadcasting_inplace" .-/broadcasting!) :void  
  (minuend :pointer)
  (subtrahend :pointer))  
  
(cffi:defcfun ("lisp_call_sub_broadcasting" .-/broadcasting) :pointer
  (minuend :pointer)
  (subtrahend :pointer))
  
(cffi:defcfun ("lisp_call_mul_broadcasting_inplace" .*/broadcasting!) :void
  (multiplicand :pointer)
  (multiplier :pointer)) 
  
(cffi:defcfun ("lisp_call_mul_broadcasting" .*/broadcasting) :pointer   
  (multiplicand :pointer)
  (multiplier :pointer))

(cffi:defcfun ("lisp_call_div_broadcasting_inplace" .//broadcasting!) :void
  (dividend :pointer)
  (divisor :pointer))   
  
(cffi:defcfun ("lisp_call_div_broadcasting" .//broadcasting) :pointer
  (dividend :pointer)
  (divisor :pointer))    
  
(cffi:defcfun ("lisp_call_pow_broadcasting_inplace" .^/broadcasting!) :void
  (base :pointer)
  (exponent :pointer))  

(cffi:defcfun ("lisp_call_pow_broadcasting" .^/broadcasting) :pointer
  (base :pointer)
  (exponent :pointer))    
  
(cffi:defcfun ("lisp_call_max_broadcasting_inplace" .max/broadcasting!) :void
  (a :pointer)
  (b :pointer))     
  
(cffi:defcfun ("lisp_call_max_broadcasting" .max/broadcasting) :pointer
  (a :pointer)
  (b :pointer))

(cffi:defcfun ("lisp_call_min_broadcasting_inplace" .min/broadcasting!) :void
  (a :pointer)
  (b :pointer))     
  
(cffi:defcfun ("lisp_call_min_broadcasting" .min/broadcasting) :pointer
  (a :pointer)
  (b :pointer))      

(cffi:defcfun ("lisp_call_axpy_broadcasting_inplace" axpy/broadcasting!) :void
  (summand :pointer)
  (sumend :pointer)
  (alpha :float))   
  
(cffi:defcfun ("lisp_call_axpy_broadcasting" axpy/broadcasting) :pointer
  (summand :pointer)
  (sumend :pointer)
  (alpha :float))     
  
(defun both-scalars-p (a b)
  "Checks whether both arguments are scalars"
  (nnl2.hli:fastcall (and (typep a 'real) (typep b 'real))))

(defun any-scalar-p (a b)
  "Checks whether at least one argument is a scalar"
  (nnl2.hli:fastcall (or (typep a 'real) (typep b 'real))))

(defun shapes-equal-p (a b)
  "Checks whether the tensor shapes match"
  (nnl2.hli:fastcall (equal (shape a :as :list) (shape b :as :list))))

(defun higher-rank-tensor (a b)
  "Returns a pair (higher lower) depending on the rank"
  (nnl2.hli:fastcall (if (> (rank a) (rank b))
					   (values a b)
					   (values b a))))  
					   
(declaim (inline both-scalars-p any-scalar-p shapes-equal-p higher-rank-tensor))					   

(defmacro with-tensor-dispatch ((a b) scalar-case tensor-case same-shape-case broadcast-case)
  "Universal dispatcher for tensor operations"
  
  (let ((a-sym (gensym "A"))
        (b-sym (gensym "B")))
		
    `(let ((,a-sym ,a)
           (,b-sym ,b))
   
       (cond
         ((both-scalars-p ,a-sym ,b-sym) ,scalar-case)
         ((typep ,a-sym 'real) (error "You can't apply a tensor function to a scalar"))
         ((typep ,b-sym 'real) ,tensor-case)
         ((shapes-equal-p ,a-sym ,b-sym) ,same-shape-case)
         (t (multiple-value-bind (higher lower) (higher-rank-tensor ,a-sym ,b-sym) ,broadcast-case))))))

(defun axpy (a b &key (alpha 1.0) &aux (alpha (coerce alpha 'single-float)))
  "Calculates A * X + Y
   a: A
   b: Y
   alpha (&key): X"
  
  (declare (type real alpha))
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
	  (+ a (* b alpha))
	  (axpy/axpf a b alpha)
	  
	  (cond
	    ((= alpha 1.0) (nnl2.ffi:%+ a b))
		((= alpha -1.0) (nnl2.ffi:%- a b))
		((= alpha 0.0) a)
		(t
		  (nnl2.ffi:%axpy a b alpha)))
	  
	  (axpy/broadcasting higher lower alpha))))
			
(defun axpy! (a b &key (alpha 1.0) &aux (alpha (coerce alpha 'single-float)))
  "In-place version of A * X + Y
   a: A (modified in-place)
   b: Y
   alpha (&key): X"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (setq a (+ a (* b alpha)))
      (axpy/axpf! a b alpha)
	  
      (cond 
	    ((= alpha 1.0)  (nnl2.ffi:%+= a b) a)
		((= alpha -1.0) (nnl2.ffi:%-= a b) a)
		((= alpha 0.0) a)
		(t  
		  (nnl2.ffi:%axpy! a b alpha) a))
		  
      (axpy/broadcasting! higher lower alpha))))	

(defun .+/internal (a b)
  "Element-wise addition"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (+ a b)
      (.+/incf a b)
      (nnl2.ffi:%+ a b)
      (.+/broadcasting higher lower))))
	  
(defun .+ (&rest args)
  "Reduce tensors with element-wise addition"
  (reduce #'.+/internal args))  ;; mem leak xd

(defun +=/internal (a b)
  "In-place element-wise addition"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (incf a b)
      (.+/incf! a b)
      (nnl2.ffi:%+= a b)
      (.+/broadcasting! higher lower)))
	  
  a)
  
(defun += (&rest args)
  "Reduce tensors with in-place element-wise addition"
  (reduce #'+=/internal args))  

(defun .-/internal (a b)
  "Element-wise subtraction"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (- a b)
      (.-/decf a b)
      (nnl2.ffi:%- a b)
      (.-/broadcasting higher lower))))
	  
(defun .- (&rest args)
  "Reduce tensors with element-wise subtraction"
  (reduce #'.-/internal args))

(defun -=/internal (a b)
  "In-place element-wise subtraction"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (decf a b)
      (.-/decf! a b)
      (nnl2.ffi:%-= a b)
      (.-/broadcasting! higher lower)))
	  
  a)
  
(defun -= (&rest args)
  "Reduce tensors with in-place element-wise subtraction"
  (reduce #'-=/internal args))

(defun .*/internal (a b)
  "Element-wise multiplication"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (* a b)
      (.*/mulf a b)
      (nnl2.ffi:%* a b)
      (.*/broadcasting higher lower))))
	  
(defun .* (&rest args)
  "Reduce tensors with element-wise multiplication"
  (reduce #'.*/internal args))

(defun *=/internal (a b)
  "In-place element-wise multiplication"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (setq a (* a b))
      (.*/mulf! a b)
      (nnl2.ffi:%*= a b)
      (.*/broadcasting! higher lower)))
	  
  a)
  
(defun *= (&rest args)
  "Reduce tensors with in-place element-wise multiplication"
  (reduce #'*=/internal args))

(defun .//internal (a b)
  "Element-wise division"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (/ a b)
      (.//divf a b)
      (nnl2.ffi:%/ a b)
      (.//broadcasting higher lower))))
	  
(defun ./ (&rest args)
  "Reduce tensors with element-wise division"
  (reduce #'.//internal args))

(defun /!/internal (a b)
  "In-place element-wise division"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (setq a (/ a b))
      (.//divf! a b)
      (nnl2.ffi:%/= a b)
      (.//broadcasting! higher lower)))
	  
  a)
  
(defun /! (&rest args)
  "Reduce tensors with in-place element-wise division"
  (reduce #'/!/internal args))

(defun .^/internal (a b)
  "Element-wise exponentiation"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (expt a b)
      (.^/powf a b)
      (nnl2.ffi:%.^ a b)
      (.^/broadcasting higher lower))))
	  
(defun .^ (&rest args)
  "Reduce tensors with element-wise exponentiation"
  (reduce #'.^/internal args))

(defun ^=/internal (a b)
  "In-place element-wise exponentiation"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (setq a (expt a b))
      (.^/powf! a b)
      (nnl2.ffi:%^= a b)
      (.^/broadcasting! higher lower)))
	  
  a)
  
(defun ^= (&rest args)
  "Reduce tensors with in-place element-wise exponentiation"
  (reduce #'^=/internal args))

(defun .max/internal (a b)
  "Element-wise maximum"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (max a b)
      (.max/maxf a b)
      (nnl2.ffi:%.max a b)
      (.max/broadcasting higher lower))))
	  
(defun .max (&rest args)
  "Reduce tensors with element-wise maximum"
  (reduce #'.max/internal args))	  

(defun .max!/internal (a b)
  "In-place element-wise maximum"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (setq a (max a b))
      (.max/maxf! a b)
      (nnl2.ffi:%.max! a b)
      (.max/broadcasting! higher lower)))
	  
  a)
  
(defun .max! (&rest args)
  "Reduce tensors with in-place element-wise maximum"
  (reduce #'.max!/internal args))

(defun .min/internal (a b)
  "Element-wise minimum"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (min a b)
      (.min/minf a b)
      (nnl2.ffi:%.min a b)
      (.min/broadcasting higher lower))))
	  
(defun .min (&rest args)
  "Reduce tensors with element-wise minimum"
  (reduce #'.min/internal args))

(defun .min!/internal (a b)
  "In-place element-wise minimum"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (a b)
      (setq a (min a b))
      (.min/minf! a b)
      (nnl2.ffi:%.min! a b)
      (.min/broadcasting! higher lower)))
	  
  a)
  
(defun .min! (&rest args)
  "Reduce tensors with in-place element-wise minimum"
  (reduce #'.min!/internal args))
  
(declaim (ftype (function (&rest (or nnl2-tensor real)) nnl2-tensor) .+ .- .* ./ .^ .min .max += -= *= /! ^= .min! .max!)
		 (inline .+ .- .* ./ .^ .min .max += -= *= /! ^= .min! .max!))
			
(defun tensor-p (obj)
  "Predicate that determines whether an object is a tensor or not"
  (and (typep obj 'nnl2-tensor) (= (length (shape obj)) (rank obj))))						

(declaim (inline tensor-p))

(cffi:defcfun ("nnl2_cast" ncast) :pointer
  (tensor :pointer)
  (cast-to nnl2.ffi:tensor-type))		  
  
(defun reshape (tensor new-shape &key force)
  "Changes the tensor's shape to a new value and returns a copy of the tensor with the new shape
   tensor: Input tensor
   new-shape: Input new shape
   force (&key): Should the incompatibility of the form be ignored (5x5) -> (1x24)"
   
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr new-shape)
    (nnl2.ffi:%reshape tensor shape rank force)))
	
(defun reinterpret (tensor new-shape &key force)
  "Changes the tensor's shape to a new value and returns a view of the tensor with the new shape
   tensor: Input tensor
   new-shape: Input new shape
   force (&key): Should the incompatibility of the form be ignored (5x5) -> (1x24)"
   
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr new-shape)
    (nnl2.ffi:%reinterpret tensor shape rank force)))																			 
																			 
(defun process-to-indices (to tensor-shape)
  "Process `to` (first arg) indices, replacing -1 with actual dimension sizes"
  
  (loop for i from 0 below (length to)
        for t-val = (elt to i)
        collect (if (= t-val -1) (aref tensor-shape i) t-val)))

(defun slice (tensor &key from to)
  "Return a sliced copy of the tensor with the specified range
    
   Args:
       tensor: The input tensor to slice
	   
	   from: Starting indices for each dimension (vector/list)
			  If not provided, defaults to zeros for all dimensions
			  
       to: Ending indices for each dimension (vector/list)
	        If not provided, defaults to the tensor's shape (full tensor)
			Use -1 to automatically use the full dimension size
			
   Examples:
       (slice tensor :from #(0 0) :to #(3 3))    ;; Slice from (0,0) to (3,3)
	   (slice tensor :from #(1 1) :to #(-1 -1))  ;; Slice from (1,1) to end
	   (slice tensor :to #(2 2))                 ;; Slice from start to (2,2)
	   (slice tensor :from #(0 1))               ;; Slice from (0,1) to end

   Returns:
       A new tensor containing the sliced data as a copy"

  (let* ((tensor-shape (shape tensor :as :vector))
         (from (if from from (make-array (list (length tensor-shape)) :initial-element 0)))
         (to (if to to tensor-shape))
         (processed-to (process-to-indices to tensor-shape))
         (pntr-from (nnl2.hli:make-shape-pntr from))
         (pntr-to (nnl2.hli:make-shape-pntr processed-to)))
    
    (nnl2.ffi:%slice tensor pntr-from pntr-to)))																
							
(cffi:defcfun ("nnl2_nrows" nrows) :int
  (tensor :pointer))

(cffi:defcfun ("nnl2_ncols" ncols) :int
  (tensor :pointer)) 
  
(cffi:defcfun ("lisp_call_transposition" transposition) :pointer
  (tensor :pointer))
  
(cffi:defcfun ("lisp_call_transposition_inplace" transposition!) :void
  (tensor :pointer))  

(defun fill! (tensor val)
  "Fills a tensor with the specified value
   
   Args:
      tensor: Input Tensor
      val: Value to fill
	  
   Example:
      (fill! foo 1) - Fill tensor with ones"
	  
  (let* ((dtype (dtype tensor))
		 (cffi-type (type/nnl2->cffi dtype))
		 (lisp-type (type/nnl2->lisp dtype))
		 (alloc (cffi:foreign-alloc cffi-type)))
		 
	(setf (cffi:mem-ref alloc cffi-type) (coerce val lisp-type))
	
    (let ((status (nnl2.ffi:%fill! tensor alloc dtype)))
	  (cffi:foreign-free alloc)
	  (unless status (warn "Failed to fill passed tensor (fill!)")))))

(cffi:defcfun ("lisp_call_neg_inplace" .neg!) :void
  (tensor :pointer))

(cffi:defcfun ("lisp_call_neg" .neg) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_sqrt" .sqrt) :pointer
  (tensor :pointer))

(cffi:defcfun ("lisp_call_sqrt_inplace" .sqrt!) :void
  (tensor :pointer))  
  