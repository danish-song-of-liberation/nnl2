(in-package :nnl2.hli)

;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/highlevel-accessors.lisp
;; File: highlevel-accessors.lisp

;; Contains a high-level interface for all the main TS functions in ffi-c-core.lisp

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

(defparameter *nnl2-tensor-types* '((:float64 . double-float) (:float32 . single-float) (:int32 . integer) (:int64 . (signed-byte 64)))
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
	  (:int32 (the symbol 'integer))
	  (:int64 '(signed-byte 64)))))
	
(declaim (ftype (function (keyword) symbol) type/nnl2->lisp)) ;; Inline not needed	
  
(defun type/lisp->nnl2 (lisp-type)
  "Converts a lisp type to a tensor type
   lisp-type: lisp type for conversion into nnl2 tensor type
   
   Example: (type/lisp->nnl2 'double-float) -> :FLOAT64"
   
  (declare (type symbol lisp-type))

  (nnl2.hli:fastcall
    (cond ((eql lisp-type 'double-float)        :float64) 
          ((eql lisp-type 'single-float)        :float32) 
          ((eql lisp-type 'integer)             :int32)
          ((equal lisp-type '(signed-byte 64))  :int64))))
		  
(declaim (ftype (function (symbol) keyword) type/lisp->nnl2))

(defun type/lisp->cffi (lisp-type)
  "Converts a lisp type to a cffi type
   lisp-type: lisp type for conversion into cffi type
   
   Example: (type/lisp->cffi 'double-float) -> :double"
   
  (declare (type symbol lisp-type))

  (nnl2.hli:fastcall
    (cond ((eql lisp-type 'double-float)        :double) 
          ((eql lisp-type 'single-float)        :float) 
          ((eql lisp-type 'integer)             :int)
          ((equal lisp-type '(signed-byte 64))  :long))))
		  
(declaim (ftype (function (symbol) keyword) type/lisp->nnl2 type/lisp->cffi)) ;; Inline not needed		
	
(defun type/nnl2->cffi (cffi-type)
  "Converts the tensor system type to a cffi type
   cffi-type: type for conversion
   
   Example: (type/nnl2->cffi :float64) -> :double"

  (declare (type keyword cffi-type))
  
  (nnl2.hli:fastcall 
    (ecase (the keyword cffi-type) 
	  (:float64 (the keyword :double)) 
	  (:float32 (the keyword :float))
	  (:int32 (the keyword :int))
	  (:int64 (the keyword :int64)))))
	  
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

(defun save-tensor (tensor path)
  (let ((result (nnl2.ffi:%ts-serialize-tensor tensor (concatenate 'string path ".nnlt"))))
    (unless result (error "[nnl2] Failed to save tensor: serializer returned false (fail)"))))
  
(defun load-tensor (path)
  (nnl2.ffi:%ts-deserialize-tensor (concatenate 'string path ".nnlt")))  

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
		     (lisp-type (type/nnl2->lisp dtype))
	    	 (filler-pntr (cffi:foreign-alloc cffi-type)))
			 
		(declare (type keyword cffi-type))	 
		  
	    (setf (cffi:mem-ref filler-pntr cffi-type) (coerce filler lisp-type))
	  
        (let ((tensor (nnl2.ffi:%full shape rank dtype filler-pntr)))
	      (cffi:foreign-free filler-pntr)		  
		  tensor)))))

(defun arange (&key from to step dtype)
  "Create a 1-dimensional tensor with evenly spaced values within a given interval
  
   The function automatically determines whether to use integer or floating-point
   arithmetic based on the types of the input arguments

   Args:
       from (&key) (optional): Start value of the sequence
	   to (&key) (required): End value of the sequence
	   step (&key) (optional): Step size between consecutive values
	   dtype (&key) (optional): Data type of the resulting tensor
	   
   Returns:
       A 1D tensor containing the sequence [from, from+step, from+2*step, ...]
       where each value is less than to (for step > 0) or greater than to
       (for step < 0)

   Examples:
       (arange :from 0 :to 5) ;; -> [0 1 2 3 4] (int64)
	   (arange :from 0 :to 10 :step 2) ;; -> [0 2 4 6 8] (int64)
	   (arange :from 5 :to 0 :step -1 :dtype :int32) ;; -> [5 4 3 2 1] (int32)
	   (arange :from 0.0 :to 1.0 :step 0.2) ;; -> [0.0 0.2 0.4 0.6 0.8] (float64)
	   (arange :from 0 :to 5 :dtype :float32) ;; -> [0.0 1.0 2.0 3.0 4.0] (float32)
	   (arange :from 0 :to 2.5 :step 0.5) ;; -> [0.0 0.5 1.0 1.5 2.0] (float64)"
	   
  (unless to
    (error "arange requires the :to argument"))
  
  (let* ((floatp (or (floatp from) (floatp to) (floatp step)))
         (final-dtype (cond
                        (dtype dtype)  
                        (floatp :float64)  
                        (t :int64)))) 
    
	(unless from
      (setf from (if floatp 0.0 0)))
	
	(unless step
      (setf step (if floatp 
				   (if (< from to) 1.0 -1.0)
                   (if (< from to) 1 -1))))
	
    (if floatp
        ;; convert all arguments to float
        (nnl2.ffi:%float-arange (float from) (float to) (float step) final-dtype)
        ;; convert all arguments to integers
        (nnl2.ffi:%int-arange (truncate from) (truncate to) (truncate step) final-dtype))))

(defun linspace (&key start stop (num 50) (endpoint t) dtype)
  "Creates a tensor with a linear sequence of numbers
   
   Examples:
       (linspace :start 0 :stop 10 :num 5)    ; 0.0, 2.5, 5.0, 7.5, 10.0
       (linspace :start 0 :stop 1 :num 5 :endpoint nil) ; 0.0, 0.2, 0.4, 0.6, 0.8
       (linspace :start 5 :stop 0 :num 6 :dtype :int32) ; 5, 4, 3, 2, 1, 0
   
   Args:
       start (&key) (required): Starting value (inclusive)
       stop (&key) (required): Ending value (inclusive when endpoint=true)
       num (&key) (default: 50) (optional): Number of samples
       endpoint (&key) (default: t) (optional): If true, stop is included
       dtype (&key) (default: :float64 or :int64 based on inputs) (optional): Data type
   
   Returns:
     A tensor with the linear sequence"
  
  (assert start nil "[nnl2] linspace: Parameter :start is required")
  (assert stop nil "[nnl2] linspace: Parameter :stop is required")
  (assert (>= num 0) nil "[nnl2] linspace: Parameter :num must be >= 0")

  (let ((use-float (or (and start (or (floatp start) (floatp stop)))
                       (and stop (or (floatp start) (floatp stop))))))
    
    (unless dtype
      (setf dtype (cond
                    (use-float :float64)
                    (t :int64))))
	
    (if use-float
      (nnl2.ffi:%float-linspace (coerce start 'single-float) (coerce stop 'single-float) num endpoint dtype)
      (nnl2.ffi:%int-linspace (truncate start) (truncate stop) num endpoint dtype))))
						 
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

(defun make-tensor (data &key (dtype nnl2.system:*default-tensor-type*) (shape-hint nil))
  "Makes a tensor from the specified data
   Example: (make-tensor #2A((1 2 3) (4 5 6))) or (make-tensor '((1 2 3) (4 5 6)))
   Tip: Try to use vectors instead of lists. This will give you a speed boost of ~2-3+ times"
   
  (etypecase data
    (array
	  (let* ((data-shape (if shape-hint shape-hint (array-dimensions data)))
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
	
(defun strides (tensor &key (as :vector))
  "Function for getting the strides of a tensor
  
   tensor: Input tensor
   as (key): Return type
   
   Available returnable type: (:list :vector :pointer)
   
   Example 1: (strides foo :as :list) -> '(5 1)
   Example 2: (strides foo :as :vector) -> #(5 1)
   Example 3: (strides foo :as :pointer) -> Depends on the lisp implementation"
   
  (case as
    (:list     (get-strides-as-list tensor))
    (:vector   (get-strides-as-vector tensor))
	(:pointer  (nnl2.ffi:get-pointer-to-tensor-strides tensor))
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
  
(cffi:defcfun ("nnl2_gemmvp" gemmvp) :pointer
  (a :pointer)
  (b :pointer)
  (vector :pointer)) 

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
  
(defun .log10! (tensor)
  "Applies the base-10 logarithm to the tensor in place
   tensor: Input tensor"
  
  (nnl2.ffi:%.log10! tensor))  
  
(defun .log10 (tensor &key save-type)
  "Applies the base-10 logarithm to the tensor
   tensor: Input tensor
   save-type (key): Try to preserve the type as much as possible (for example, for int32)"
   
  (nnl2.ffi:%.log10 tensor save-type))      
    
(defun .log2! (tensor)
  "Applies the base-2 logarithm to the tensor in place
   tensor: Input tensor"
  
  (nnl2.ffi:%.log2! tensor))  
  
(defun .log2 (tensor &key save-type)
  "Applies the base-2 logarithm to the tensor
   tensor: Input tensor
   save-type (key): Try to preserve the type as much as possible (for example, for int32)"
   
  (nnl2.ffi:%.log2 tensor save-type))      	

(defun .log1p! (tensor)
  "Applies the natural logarithm of (1 + x) to the tensor in place
   tensor: Input tensor"
  
  (nnl2.ffi:%.log1p! tensor))  
  
(defun .log1p (tensor &key save-type)
  "Applies the natural logarithm of (1 + x) to the tensor
   tensor: Input tensor
   save-type (key): Try to preserve the type as much as possible (for example, for int32)"
   
  (nnl2.ffi:%.log1p tensor save-type))    
  
(cffi:defcfun ("get_size" numel) :int
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
   in (&key) (default: nil): Number of input units
   out (&key) (default: nil): Number of output units
   gain (&key) (default: 1.0s0): Scaling factor 
   distribution (&key) (default: :normal): Type of distribution (:normal/:uniform)"
   
  (assert (and in out gain distribution) nil "Incorrect keys was passed in xavier-like")
  (nnl2.ffi:%xavier-like tensor in out gain (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))))
  
(defun kaiming-like (tensor &key in out (gain (sqrt 2.0s0)) (distribution :normal) (mode :fan-in))
  "Creates a new tensor of the same shape and type as the input tensor,
   initialized using Kaiming (He) initialization method
   
   tensor: The input tensor from which shape and data type are derived
   in (&key) (default: nil): Number of input neurons
   out (&key) (default: nil): Number of output neurons
   gain (&key) (default: (sqrt 2.0s0)): Scaling factor (typically sqrt(2.0) for ReLU)
   distribution (&key) (default: :normal): Type of distribution (:normal/:uniform)
   mode (&key) (default: :fan-in): Mode of initialization (:fan-in/:fan-out/:fan-avg)"
   
  (assert (and in out gain distribution mode) nil "Incorrect keys was passed in kaiming-like")
  (nnl2.ffi:%kaiming-like tensor in out gain (ecase distribution (:normal 2.0s0) (:uniform 6.0s0)) (ecase mode (:fan-in 0) (:fan-out 1) (:fan-avg 2))))  
  
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
        (numel (numel first-tensor))
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
         (numel (numel first-tensor))
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
  
(defun %internal-uniform (indices dtype from to)
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
			
	  (nnl2.ffi:%uniform shape rank dtype from-pntr to-pntr))))
	  
(defun %internal-uniform-inplace (tensor from to &key type-hint)
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
		  
	(nnl2.ffi:%uniform-inplace tensor from-pntr to-pntr)))
	  
(defun uniform (indices &key (dtype nnl2.system:*default-tensor-type*) (from 0) (to 1))
  "Creates a tensor of the specified shape filled with random numbers from 0 to 1
   indices: Input shape
   dtype (&key) (default: nnl2.system:*default-tensor-type*): Type of tensor
   from (&key) (default: 0): Value to fill from
   to (&key) (default: 1): Value to fill to"
   
   (%internal-uniform indices dtype from to))
  
(defun uniform! (tensor &key type-hint (from 0) (to 1))
  "Fills an existing tensor with random numbers from [0, 1]
   tensor: Input tensor
   type-hint (&key): You can hint a type to optimal performance
   from (&key) (default: 0): Value to fill from
   to (&key) (default: 1): Value to fill to"
   
  (%internal-uniform-inplace tensor from to :type-hint type-hint))

(defun rand (indices &key (dtype nnl2.system:*default-tensor-type*))
  "Creates a tensor of the specified shape filled with random numbers from 0 to 1
   indices: Input shape
   dtype (&key) (default: nnl2.system:*default-tensor-type*): Type of tensor
   
   Returns: A new tensor with the specified shape filled with uniform random values [0, 1]
   
   Example:
     (rand #(2 3)) ; Creates a 2x3 tensor with random values"
	 
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%rand shape rank dtype)))
	
(cffi:defcfun ("lisp_call_rand_inplace" rand!) :pointer
  (tensor :pointer))  	

(defun randn (indices &key (dtype nnl2.system:*default-tensor-type*) (mean 0.0d0) (std 1.0d0))
  "Creates a tensor of the specified shape filled with random numbers from the standard normal distribution N(0, 1)
  
   indices: Input shape
   dtype (&key) (default: nnl2.system:*default-tensor-type*): Type of tensor
   mean (&key) (default: 0.0d0): Mean of the normal distribution
   std (&key) (default: 1.0d0): Standard deviation of the normal distribution
   
   Returns: A new tensor with the specified shape filled with uniform random values [0, 1]
   
   Example:
     (rand #(2 3)) ; Creates a 2x3 tensor with random values"
	 
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%randn shape rank dtype (coerce mean 'double-float) (coerce std 'double-float))))
	
(cffi:defcfun ("lisp_call_randn_inplace" randn!) :pointer
  (tensor :pointer))  	
  
(defun %internal-uniform-like (tensor from to dtype)
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
						 
	(nnl2.ffi:%uniform-like tensor from-pntr to-pntr)))
  
(defun uniform-like (tensor &key (from 0.0d0) (to 1.0d0) (dtype (dtype tensor)))
  "Сreates a tensor filled with random numbers from 0 to 1 of the same shape as the passed tensor
   tensor: Input tensor
   dtype (&key) (default: (nnl2.hli.ts:dtype tensor)): Type of new tensor
   from (&key) (default: 0.0d0): Value to fill from
   to (&key) (default: 1.0d0): Value to fill to"
   
  (%internal-uniform-like tensor from to dtype))  
				 
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
  
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
      (case distribution
	    (:normal   (nnl2.ffi:%xavier shape rank dtype in out gain 2.0s0))
	    (:uniform  (nnl2.ffi:%xavier shape rank dtype in out gain 6.0s0))
	    (otherwise (error "Unknown xavier distribution: ~a%" distribution))))))
	
(defun kaiming (indices &key (dtype nnl2.system:*default-tensor-type*) (in 0) (out 0) (gain 1.0s0) (distribution :normal) (mode :fan-in))
  "Makes a tensor with the kaiming (He) distribution in the specified shape
   indices: Input shape (vector/list)
   dtype (&key): Tensor Type
   in (&key): Number of input neurons
   out (&key): Number of output neurons
   gain (&key): Gain factor (float, default 1.0)
   distribution (&key) (default :normal): Distribution parameter (:normal/:uniform) 
   mode (&key): Initialization mode, can be :fan-in (default), :fan-out, or :fan-avg"

  (progn
    (assert (not (or (zerop in) (minusp in))) nil "Bad `in` was passed to kaiming")
    (assert (not (or (zerop out) (minusp out))) nil "Bad `out` was passed to kaiming")
    
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
      (let ((mode-value (ecase mode (:fan-in 0) (:fan-out 1) (:fan-avg 2)))
			(dist (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))))
			
        (nnl2.ffi:%kaiming shape rank dtype in out gain dist mode-value)))))
	
(defun kaiming! (tensor &key (in 0) (out 0) (gain (sqrt 2.0s0)) (distribution :normal) (mode :fan-in))
  "Fills a tensor with the kaiming (He) distribution in place
   tensor: Input tensor
   in (&key): Number of input neurons
   out (&key): Number of output neurons
   gain (&key): Gain factor (float, default sqrt(2.0) for ReLU)
   distribution (&key): Distribution parameter (:normal/:uniform, default :normal)
   mode (&key): Initialization mode, can be :fan-in (default), :fan-out, or :fan-avg"

  (assert (not (or (zerop in) (minusp in))) nil "Bad `in` was passed to kaiming!")
  (assert (not (or (zerop out) (minusp out))) nil "Bad `out` was passed to kaiming!")
  
  (let ((mode-value (ecase mode (:fan-in 0) (:fan-out 1) (:fan-avg 2)))
        (dist (ecase distribution (:normal 2.0s0) (:uniform 6.0s0))))
    
    (nnl2.ffi:%kaiming-inplace tensor in out gain dist mode-value)))
	
(defun xavier! (tensor &key (in 0) (out 0) (gain 1.0s0) (distribution :normal))
  "Fills a tensor with the xavier distribution in the specified shape in place
   tensor: Input tensor (vector/list)
   in (&key): Number of input neurons
   out (&key): Number of output neurons
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
   
  (declare (ignore axes))
   
  (let* ((type-t (type/nnl2->cffi dtype))
		 (out (cffi:foreign-alloc type-t)))
					
	(nnl2.ffi:%l2norm tensor out)
				
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
	
(defun .atan2/atan2f! (y x-scalar)
  "Applies in-place atan2 operation to y tensor with scalar x
   y: Input y-coordinate tensor (modified in place)
   x-scalar: Scalar x-coordinate value"
   
  (let* ((dtype (dtype y))
         (cffi-dtype (type/nnl2->cffi dtype))
         (lisp-dtype (type/nnl2->lisp dtype))
         (x-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref x-pntr cffi-dtype) (coerce x-scalar lisp-dtype))
    
    (nnl2.ffi:%atan2-correspondence-inplace y x-pntr)))

(defun .atan2/atan2f (y x-scalar)
  "Applies atan2 operation to y tensor with scalar x, returning a new tensor
   y: Input y-coordinate tensor
   x-scalar: Scalar x-coordinate value"
   
  (let* ((dtype (dtype y))
         (cffi-dtype (type/nnl2->cffi dtype))
         (lisp-dtype (type/nnl2->lisp dtype))
         (x-pntr (cffi:foreign-alloc cffi-dtype)))
         
    (setf (cffi:mem-ref x-pntr cffi-dtype) (coerce x-scalar lisp-dtype))
    
    (nnl2.ffi:%atan2-correspondence y x-pntr)))
	
(defmacro .square! (tensor)
  "Raises the tensor to the power of 2 in-place
   tensor: Input tensor to be modified in-place"
   
  `(.^/powf! ,tensor 2.0s0))  
  
(defmacro .cube! (tensor)
  "Raises the tensor to the power of 3 in-place
   tensor: Input tensor to be modified in-place"
   
  `(.^/powf! ,tensor 3.0s0))    
  
(defmacro .square (tensor)
  "Returns a new tensor with each element raised to the power of 2
   tensor: Input tensor"
   
  `(.^/powf ,tensor 2.0s0))  
  
(defmacro .cube (tensor)
  "Returns a new tensor with each element raised to the power of 3
   tensor: Input tensor"

  `(.^/powf ,tensor 3.0s0))  	
	
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
  
(defun .atan2 (y x)
  "Element-wise atan2(y/x)"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (y x)
      (atan y x) 
      (.atan2/atan2f y x)
      (nnl2.ffi:%atan2 y x) 
      (nnl2.ffi:%atan2-broadcasting y x)))) 

(defun .atan2! (y x)
  "Element-wise atan2! (y/x)"
  
  (nnl2.hli:fastcall   
    (with-tensor-dispatch (y x)
      (setq y (atan y x)) 
      (.atan2/atan2f! y x)
      (nnl2.ffi:%atan2-inplace y x) 
      (nnl2.ffi:%atan2-broadcasting-inplace y x)))) 	  
  
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
  
(cffi:defcfun ("nnl2_rand_like" rand-like) :pointer
  (tensor :pointer))  
  
(defun randn-like (tensor &key (mean 0.0d0) (std 1.0d0))
  "Creates a new tensor with the same shape as the input tensor, filled with random numbers from N(mean, std²)
   
   tensor: Input tensor to copy shape from
   mean (&key) (default: 0.0): Mean of the normal distribution
   std (&key) (default: 1.0): Standard deviation of the normal distribution
   
   Returns: A new tensor with the same shape as input, filled with random values from N(mean, std²)
   
   Examples:
     (randn-like my-tensor) ; Creates tensor like my-tensor with values from N(0, 1)
     (randn-like my-tensor :mean 5.0 :std 0.5) ; Creates tensor with values from N(5, 0.25)"
  
  (nnl2.ffi:%randn-like tensor (coerce mean 'double-float) (coerce std 'double-float)))

(cffi:defcfun ("lisp_call_sin_inplace" .sin!) :void
  (tensor :pointer))

(cffi:defcfun ("lisp_call_cos_inplace" .cos!) :void
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_sin" .sin) :pointer
  (tensor :pointer))

(cffi:defcfun ("lisp_call_cos" .cos) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_asin_inplace" .asin!) :void
  (tensor :pointer))

(cffi:defcfun ("lisp_call_acos_inplace" .acos!) :void
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_asin" .asin) :pointer
  (tensor :pointer))

(cffi:defcfun ("lisp_call_acos" .acos) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_tan_inplace" .tan!) :void
  (tensor :pointer))

(cffi:defcfun ("lisp_call_atan_inplace" .atan!) :void
  (tensor :pointer))

(cffi:defcfun ("lisp_call_tan" .tan) :pointer
  (tensor :pointer))

(cffi:defcfun ("lisp_call_atan" .atan) :pointer
  (tensor :pointer))  
  
(in-package :nnl2.hli.ts.loss)
  
(defun mse (prediction target)
  (let* ((result-pntr (cffi:foreign-alloc :double)))	 
	(nnl2.ffi:%mse prediction target result-pntr)
	
	(let ((result (cffi:mem-ref result-pntr :double)))
	  (cffi:foreign-free result-pntr)
	  
	  result)))
	
(defun mae (prediction target)
  (let* ((result-pntr (cffi:foreign-alloc :double)))	 
    (nnl2.ffi:%mae prediction target result-pntr)
    
    (let ((result (cffi:mem-ref result-pntr :double)))
      (cffi:foreign-free result-pntr)
      
      result)))
	  
(in-package :nnl2.hli.ts.linalg)

(defun gesvd (a order jobu jobvt m n lda ldu ldvt) 
  "Compute Singular Value Decomposition (SVD) using standard LAPACK algorithm.
   
   Performs the decomposition A = U * Σ * V^T, where:
   A is an m×n input matrix
   U is an m×m (or m×min(m,n)) orthogonal matrix of left singular vectors
   Σ is a diagonal matrix with singular values (size min(m,n))
   V^T is an n×n (or min(m,n)×n) orthogonal matrix of right singular vectors (transposed)
   
   Args:
       a: Input matrix tensor [m × n] (any floating-point dtype)
       order: Storage order, either :nnl2rowmajor or :nnl2colmajor
	   
       jobu: Character specifying computation of U matrix:
             #\\A: All m columns of U
             #\\S: First min(m,n) columns of U
             #\\O: First min(m,n) columns of U overwrite input A
             #\\N: U is not computed
			 
       jobvt: Character specifying computation of V^T matrix:
              #\\A: All n rows of V^T
              #\\S: First min(m,n) rows of V^T
              #\\O: First min(m,n) rows of V^T overwrite input A
              #\\N: V^T is not computed
			  
       m: Number of rows (automatically determined if NIL)
       n: Number of columns (automatically determined if NIL)
       lda: Leading dimension of A (automatically determined if NIL)
       ldu: Leading dimension of U (automatically determined if NIL)
       ldvt: Leading dimension of VT (automatically determined if NIL)
   
     Returns:
         Four values:
             1. s: Singular values tensor [min(m,n)] in descending order
			 2. u: Left singular vectors matrix (size depends on jobu)
			 3. vt: Right singular vectors matrix (size depends on jobvt)
			 4. info: Return code (0 = success, >0 = convergence failure, <0 = illegal argument)
   
   Notes:
     All output tensors have same dtype as input tensor
     When jobu='O' or jobvt='O', input matrix A is overwritten
     Leading dimensions are automatically computed based on order if not provided
     Uses standard LAPACK gesvd algorithm (slower but more stable for some cases)
   
   Example:
     (gesvd a :nnl2rowmajor #\\A #\\A) ; Full SVD, row-major
     (gesvd a :nnl2colmajor #\\S #\\S) ; Economy SVD, column-major
   
   See also: gesdd, svd"
   
  (declare (optimize (speed 3) (safety 1)))
	
  (declare (type nnl2.hli.ts::nnl2-tensor a)
           (type keyword order)
           (type character jobu jobvt))

  (let* ((shape-a (nnl2.hli.ts:shape a :as :vector))
         (m (or m (aref shape-a 0)))
         (n (or n (aref shape-a 1)))
         (min-mn (min m n))  
         (lda (or lda (if (eq order :nnl2rowmajor) n m)))     
         (s (nnl2.hli.ts:empty (list min-mn) :dtype (nnl2.hli.ts:dtype a)))   
		 
         (u (cond ((char-equal jobu #\N) (make-tensor nil))
                  ((char-equal jobu #\O) (cffi:null-pointer))
                  (t (let ((u-rows m) (u-cols (if (char-equal jobu #\A) m min-mn)))
                       (nnl2.hli.ts:empty `(,u-rows ,u-cols) :dtype (nnl2.hli.ts:dtype a))))))
         
         (vt (cond ((char-equal jobvt #\N) (make-tensor nil))
                   ((char-equal jobvt #\O) (cffi:null-pointer))
                   (t (let ((vt-rows (if (char-equal jobvt #\A) n min-mn)) (vt-cols n))
                        (nnl2.hli.ts:empty `(,vt-rows ,vt-cols) :dtype (nnl2.hli.ts:dtype a))))))
         
         (ldu (or ldu (cond ((char-equal jobu #\N) 1)
                            ((char-equal jobu #\O) 1)
                            ((eq order :nnl2rowmajor) (if (char-equal jobu #\A) m min-mn))
                            (t m))))
				
         (ldvt (or ldvt (cond ((char-equal jobvt #\N) 1)
                              ((char-equal jobvt #\O) 1)
                              ((eq order :nnl2rowmajor) n)
                              (t (if (char-equal jobvt #\A) n min-mn)))))
         
         (superb (nnl2.hli.ts:empty `(,(max 1 (* 5 min-mn))) :dtype (nnl2.hli.ts:dtype a))))
    
    (let ((info (nnl2.ffi:%gesvd order 
                                 (char-code jobu) 
                                 (char-code jobvt)
                                 m n
                                 a lda
                                 s
                                 u ldu
                                 vt ldvt
                                 superb)))
	  
      (values s 
	    (if (cffi:null-pointer-p u) nil u)
        (if (cffi:null-pointer-p vt) nil vt)
		info))))

(defun gesdd (a order jobz m n lda ldu ldvt) 
  "Compute Singular Value Decomposition (SVD) using Divide-and-Conquer algorithm
   
   Performs the decomposition A = U * Σ * V^T using LAPACK's divide-and-conquer
   algorithm (gesdd), which is typically faster than standard gesvd for large
   matrices but uses more workspace memory
   
   Args:
       a: Input matrix tensor [m × n] (any floating-point dtype)
       order: Storage order, either :nnl2rowmajor or :nnl2colmajor
	   
       jobz: Character specifying computation of singular vectors:
             #\\A: All m columns of U and all n rows of V^T
             #\\S: First min(m,n) columns of U and rows of V^T
			 
             #\\O: Overwrites A with singular vectors:
                    If m >= n: First n columns of U overwrite A
                    If m < n: First m rows of V^T overwrite A
					
             #\\N: Neither U nor V^T are computed
			 
       m: Number of rows (automatically determined if NIL)
       n: Number of columns (automatically determined if NIL)
       lda: Leading dimension of A (automatically determined if NIL)
       ldu: Leading dimension of U (automatically determined if NIL)
       ldvt: Leading dimension of VT (automatically determined if NIL)
   
     Returns:
         Four values:
             1. s: Singular values tensor [min(m,n)] in descending order
             2. u: Left singular vectors matrix (size depends on jobz)
             3. vt: Right singular vectors matrix (size depends on jobz)
             4. info: Return code (0 = success, >0 = convergence failure, <0 = illegal argument)
   
   Notes:
       All output tensors have same dtype as input tensor
       When jobz='O', input matrix A is overwritten with singular vectors
       Requires integer workspace tensor (iwork) of size 8*min(m,n)
       Typically faster than gesvd for matrices with min(m,n) > 25
       Uses LAPACK's divide-and-conquer algorithm (DBDSDC/SBDSDC)
   
   Example:
     (gesdd a :nnl2rowmajor #\\A) ; Full SVD, row-major
     (gesdd a :nnl2colmajor #\\S) ; Economy SVD, column-major
   
   See also: gesvd, svd"
   
  (declare (optimize (speed 3) (safety 1)))
  
  (declare (type nnl2.hli.ts::nnl2-tensor a)
           (type keyword order)
           (type character jobz))

  (let* ((shape-a (nnl2.hli.ts:shape a :as :vector))
         (m (or m (aref shape-a 0)))
         (n (or n (aref shape-a 1)))
         (min-mn (min m n))  
         (lda (or lda (if (eq order :nnl2rowmajor) n m)))     
         (s (nnl2.hli.ts:empty (list min-mn) :dtype (nnl2.hli.ts:dtype a)))   
         
         (u (cond ((char-equal jobz #\N) (make-tensor nil))
                  ((char-equal jobz #\O) (cffi:null-pointer))
                  (t (let ((u-rows m) (u-cols (if (char-equal jobz #\A) m min-mn)))
                       (nnl2.hli.ts:empty `(,u-rows ,u-cols) :dtype (nnl2.hli.ts:dtype a))))))
         
         (vt (cond ((char-equal jobz #\N) (make-tensor nil))
                   ((char-equal jobz #\O) (cffi:null-pointer))
                   (t (let ((vt-rows (if (char-equal jobz #\A) n min-mn)) (vt-cols n))
                        (nnl2.hli.ts:empty `(,vt-rows ,vt-cols) :dtype (nnl2.hli.ts:dtype a))))))
         
         (ldu (or ldu (cond ((char-equal jobz #\N) 1)
                            ((char-equal jobz #\O) 1)
                            ((eq order :nnl2rowmajor) (if (char-equal jobz #\A) m min-mn))
                            (t m))))
         
         (ldvt (or ldvt (cond ((char-equal jobz #\N) 1)
                              ((char-equal jobz #\O) 1)
                              ((eq order :nnl2rowmajor) (if (char-equal jobz #\A) n min-mn))
                              (t (if (char-equal jobz #\A) n min-mn)))))
         
         (iwork (nnl2.hli.ts:empty (list (* 8 min-mn)) :dtype :int32)))
    
    (let ((info (nnl2.ffi:%gesdd order 
                                 (char-code jobz)
                                 m n
                                 a lda
                                 s
                                 u ldu
                                 vt ldvt
                                 iwork)))
      
      (values s 
              (if (cffi:null-pointer-p u) nil u)
              (if (cffi:null-pointer-p vt) nil vt)
			  info))))

(defun svd (a &key (lapack :gesdd) (order :nnl2rowmajor) (jobu #\A) (jobvt #\A) (jobz #\A) m n lda ldu ldvt)
  "Singular Value Decomposition (SVD) operation
   A = U * Σ * V^T
   
   Args:
     a: Input matrix tensor [m × n]
	 
     lapack: Algorithm to use:
             :gesvd - Standard SVD algorithm
             :gesdd - Divide-and-Conquer SVD algorithm 
			 Default: :gesdd
			 
     order: Storage order (:nnl2rowmajor or :nnl2colmajor)
     
     Parameters for gesvd:
		 jobu: Options for computing U matrix (for gesvd):
			   #\A - all m columns of U
			   #\S - first min(m,n) columns of U  
			   #\O - first min(m,n) columns of U overwrite input A
			   #\N - no columns of U computed
			   
		 jobvt: Options for computing V^T matrix (for gesvd):
				#\A - all n rows of V^T
				#\S - first min(m,n) rows of V^T
				#\O - first min(m,n) rows of V^T overwrite input A
				#\N - no rows of V^T computed
            
     Parameters for gesdd:
		 jobz: Options for computing singular vectors (for gesdd):
			   #\A - all m columns of U and all n rows of V^T
			   #\S - first min(m,n) columns of U and rows of V^T
			   #\O - 
			   ;    If m >= n: first n columns of U overwrite A, V^T is computed
			   ;    If m < n: first m rows of V^T overwrite A, U is computed
			   #\N - neither U nor V^T are computed
			   
   m, n: Matrix dimensions (automatically determined if not provided)
   lda, ldu, ldvt: Leading dimensions (automatically determined if not provided)
   
   Returns:
       Multiple values: (s u vt info) where:
       s: singular values tensor [min(m,n)]
       u: left singular vectors tensor (size depends on jobu/jobz)
       vt: right singular vectors tensor (size depends on jobvt/jobz)
	   info: return code (0 = success)"
   
  (ecase lapack 
    (:gesvd (gesvd a order jobu jobvt m n lda ldu ldvt))
    (:gesdd (gesdd a order jobz m n lda ldu ldvt))))
	
(declaim (inline svd))  

(defun diag (tensor &key (k 0))
  "Extract or construct a diagonal matrix/vector
   
   Args:
       tensor: Input tensor (vector or matrix)
       k (&key) (default: 0): Diagonal index
       k = 0: main diagonal
       k > 0 : K-th diagonal above the main
       k < 0 : K-th diagonal below the main

   Returns:
       If input is vector: New square matrix with diagonal elements
       If input is matrix: New vector containing diagonal elements"
	   
  (if (= (nnl2.hli.ts:rank tensor) 1)
    (nnl2.ffi:%diag-vector-matrix tensor k)
	(nnl2.ffi:%diag-matrix-vector tensor k)))
		
(declaim (inline diag))
		
(defun luf (a &key (order :nnl2rowmajor) m n lda ipiv)
  "Compute LU factorization with partial pivoting
  
   Performs the decomposition A = P * L * U, where
   P is a permutation matrix
   L is lower triangular with unit diagonal elements
   U is upper triangular
  
   Args:
       a: Input matrix tensor [m × n] (any floating-point dtype)
       order: Storage order, either :nnl2rowmajor or :nnl2colmajor
       m: Number of rows (automatically determined if NIL)
       n: Number of columns (automatically determined if NIL)
       lda: Leading dimension of A (automatically determined if NIL)
       ipiv: Existing tensor for pivot indices or NIL to create new
  
   Returns:
       Three values:
           1. lu: Matrix containing LU factors in compact form
           2. ipiv: Pivot indices tensor [min(m,n), INT32]
           3. info: Return code (0 = success, >0 = singular matrix, <0 = illegal argument)
  
   Notes:
       Input matrix A is overwritten with LU factors in compact form:
         - Upper triangle (including diagonal) contains U
         - Lower triangle (excluding diagonal) contains multipliers for L
         - Diagonal elements of L are implied to be 1.0
       
       Pivot indices are 1-based (LAPACK convention)
       If info > 0, U(info,info) is exactly zero (matrix is singular)
  
   Example:
     (multiple-value-bind (lu ipiv info) 
         (getrf a :nnl2colmajor)
       (when (zerop info)
         ;; Use lu and ipiv for solving linear systems
         ))"
  
  (declare (optimize (speed 3) (safety 1)))
  
  (declare (type nnl2.hli.ts::nnl2-tensor a)
           (type keyword order))
  
  (let* ((shape-a (nnl2.hli.ts:shape a :as :vector))
         (m (or m (aref shape-a 0)))
         (n (or n (aref shape-a 1)))
         (min-mn (min m n))
         (lda (or lda (if (eq order :nnl2rowmajor) n m)))
         
         (ipiv-tensor (if (null ipiv)
                        (nnl2.hli.ts:empty (list min-mn) :dtype :int32)
                        (progn
                          (assert (equal (nnl2.hli.ts:shape ipiv :as :list) (list min-mn)) nil "IPIV tensor must have shape (~D)" min-mn)
                          ipiv)))
         
         (lu (nnl2.hli.ts:copy a)))
    
    (let ((info (nnl2.ffi:%getrf order 
                                 m n
                                 lu lda
                                 ipiv-tensor)))
      
      (values lu ipiv-tensor info))))		
		
(defun eye (rows cols &key (dtype nnl2.system:*default-tensor-type*))
  "Create identity matrix tensor
   
   Args:
       rows: Number of rows 
	   cols: Number of cols 
	   dtype (&key) (default: nnl2.system:*default-tensor-type*): Data type
	   
   Example:
       (nnl2.hli.ts:tlet ((a (nnl2.hli.ts.linalg:eye 3 3))) ...) ;; -> [[1 0 0] [0 1 0] [0 0 1]]"
	   
  (nnl2.ffi:%eye rows cols dtype))		

(cffi:defcfun ("nnl2_eye_like" eye-like) :pointer 
  (tensor :pointer))
  