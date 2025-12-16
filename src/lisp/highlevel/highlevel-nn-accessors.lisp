(in-package :nnl2.hli.nn)

;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/highlevel-nn-accessors.lisp
;; File: highlevel-nn-accessors.lisp

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(deftype nnl2-nn () 
  "Abstract type representing a neural network instance"
  #+sbcl      'sb-sys:system-area-pointer
  #+clisp     'fi:foreign-data
  #+ccl       'ccl:macptr
  #+ecl       'si:foreign-data
  #+abcl      'system:foreign-pointer
  #+lispworks 'fli:pointer
  #+allegro   'excl:foreign-pointer)

(progn
  (defconstant +nn-type-fnn+ 0
    "Type constant for a fully-connected feedforward network")
	
  (defconstant +nn-type-unknown+ 1
    "Type constant for an unknown/unsupported network type"))

(defparameter *nn-default-init-type* :kaiming/normal
  "Default initialization type used when creating new neural networks")

(cffi:defcfun ("nnl2_ann_free" free) :void
  (nn :pointer))

(defun forward (nn &rest args)
  "Computes the forward pass for the neural network 

   Args:
       nn: A neural network instance 
       args (&rest): Input tensors for the network

   Returns:
       Output of the network after applying forward propagation"
   
  (let* ((len (length args))
		 (args-pntr (cffi:foreign-alloc :pointer :count len)))
		 
	(dotimes (i len)
	  (setf (cffi:mem-aref args-pntr :pointer i) (nth i args)))
	  
	(let ((result (nnl2.ffi:%nn-forward nn args-pntr)))
	  (cffi:foreign-free args-pntr)
	  result)))
  
(defmacro nnlet ((&rest bindings) &body body)
  "Like cl:let, but automatically frees any nnl2-nn variables after the body executes"
  (let ((vars (mapcar #'car bindings)))
    `(let ,bindings
	   (unwind-protect
	       (progn ,@body)
		   
		 (progn 
		   ,@(loop for var in vars
		           collect `(when (typep ,var 'nnl2-nn) (free ,var))))))))

(defmacro nnlet* ((&rest bindings) &body body)
  "Like cl:let*, but automatically frees any nnl2-nn variables after the body executes"
  (if (null bindings)
   `(progn ,@body)
    (let* ((binding (first bindings))
           (var (if (consp binding) (car binding) binding))
           (value (if (consp binding) (cadr binding) nil)))
     `(let (,binding)
        (unwind-protect
          (nnlet* ,(rest bindings) ,@body)
          (when (typep ,var 'nnl2-nn) (free ,var)))))))
	
(defun extract-parameters (params num-params)
  "Extracts a list of parameter pointers from raw FFI data

   Args:
       params: Pointer to FFI array of parameters
       num-params: Number of parameters

   Returns:
       List of AD tensor instances corresponding to the parameters"
	   
  (let ((mem (cffi:foreign-alloc :pointer :count num-params))
		(lst-params nil))
		
    (dotimes (i num-params)
	  (push (cffi:mem-aref params :pointer i) lst-params))
	  
	(cffi:foreign-free mem)
	(nnl2.ffi:%nn-free-parameters params)
	
	lst-params))
	
(defun parameters (nn)
  "Returns a list of all parameters (AD tensors)
  
   Args:
       nn: Neural network instance"
	   
  (extract-parameters (nnl2.ffi:%nn-get-parameters nn) (nnl2.ffi:%nn-get-num-parameters nn)))

(defmacro fnn (in-features arrow out-features &key (bias t) (dtype nnl2.system:*default-tensor-type*) (init *nn-default-init-type*))
  "Creates a fully-connected feedforward neural network (FNN)
  
   Args:
       in-features: Number of input neurons
       arrow: DSL placeholder (ignored)
       out-features: Number of output neurons
       bias (&key) (default: t): Boolean, include bias or not (default T)
       dtype (&key) (default: nnl2.system:*default-tensor-type*): Tensor type (default *default-tensor-type*)
       init (&key) (default: nnl2.hli.nn:*nn-default-init-type*): Initialization method (keyword) or lambda function
	   
   Returns:
       A new FNN instance with parameters initialized

   :init can be:
       A keyword like :kaiming/uniform, :xavier/normal, :zeros, etc.
	   Or
       A lambda function of one argument (tensor), called on each parameter"	   
	   
  (declare (ignore arrow))
  `(let* ((nn (nnl2.ffi:%create-nn-fnn ,in-features ,out-features ,bias ,dtype ,(if (keywordp init) init :identity))))
     ,(when (not (keywordp init))
        `(dotimes (i (length (parameters nn)))
           (funcall ,init (nth i (parameters nn)))))
		   
     nn))
	 
(defmacro rnncell (in-features arrow hidden &key (bias t) (dtype nnl2.system:*default-tensor-type*) (init *nn-default-init-type*))
  (declare (ignore arrow))
  `(let* ((nn (nnl2.ffi:%create-nn-rnncell ,in-features ,hidden ,bias ,dtype ,(if (keywordp init) init :identity))))
				
     ,(when (not (keywordp init))
        `(dotimes (i (length (parameters nn)))
           (funcall ,init (nth i (parameters nn)))))
		   
     nn))	 
	 
(defun sequential (&rest layers)
  (let* ((len (length layers))
		 (layers-pntr (cffi:foreign-alloc :pointer :count len)))
		 
    (dotimes (i len)
	  (setf (cffi:mem-aref layers-pntr :pointer i) (nth i layers)))
	  
	(nnl2.ffi:%create-nn-sequential len layers-pntr)))
  
(defun .sigmoid (&key (approx t))
  (nnl2.ffi:%create-nn-sigmoid approx))
  
(defun .tanh (&key (approx t))
  (nnl2.ffi:%create-nn-tanh approx)) 

(cffi:defcfun ("nnl2_nn_relu_create" .relu) :pointer)

(defun .leaky-relu (&key (alpha nnl2.system:*leaky-relu-default-shift*))
  (nnl2.ffi:%create-nn-leaky-relu alpha)) 

(defun print-model (nn)
  (nnl2.ffi:%print-model nn t 0))
  
(in-package :nnl2.hli.nn.ga)

(defmacro encode (nn (vector nnlrepr) &body body)
  "Encodes a neural network model into a single flat vector
   Needed for GA 
   
   Args:
       nn: Input neural network 
	   
	   (vector nnlrepr):
	       vector: Name for result vector 
		   nnlrepr: Name for result encoder (may be decoded back to nn)
		   
	   body (&body): User code

   Example:
       (nnl2.hli.nn:nnlet ((corge (nnl2.hli.nn:fnn 2 -> 1)))
         (nnl2.hli.nn.ga:encode corge (nn-vector nn-encoder)
		   ;; nn-vector Is a flat vector with concatted corge parameters 
		   ;; nn-encoder Is a pointer to nnlrepr hard encoder structure
		   
		   (nnl2.hli.nn.ga:print-encoder nn-encoder) 
		   
		   ;; ... Your code
		   
		   ;; Decoding back to nn
		   (nnl2.hli.nn:nnlet ((garple (nnl2.hli.nn.ga:decode nn-encoder)))
		     (nnl2.hli.nn:print-model garple))))"
			 
  `(let* ((,nnlrepr (nnl2.ffi:%nnlrepr-encode ,nn))
		  (,vector (cffi:mem-aref ,nnlrepr :pointer 0)))
		 
	 ,@body
	 
	 (nnl2.ffi:%nnlrepr-free ,nnlrepr)))	
	
(defmacro free-encoder (encoder)
  "Frees nnlrepr encoder"
  `(nnl2.ffi:%nnlrepr-free ,encoder))	
	
(defun print-encoder (encoder)
  "Prints nnlrepr encoder"
  (nnl2.ffi:%nnlrepr-print-encoder (cffi:mem-aref encoder :pointer 1) t 0))
  
(defun decode (encoder)
  "Decodes nnlrepr encoder"
  (nnl2.ffi:%nnlrepr-decode encoder))    
  
(in-package :nnl2.hli.nn.ga.fitness)
  
(defun mse (x y)
  "Computes a modified inverse mean squared error (MSE) between two tensors
   
   Args:
       x: AD-tensor with input data 
       y: AD-tensor with target data 
	   
   Return:
	   Scalar with fitness-mse (/ 1 (+ 1 (sum (^ (- x y) 2))))
	   
	   (^ (- x y) 2) is ELEMENT-WISE
	   
	   (/ 1 (+ 1 (sum ...))) is NOT ELEMENT-WISE
	   
   Example:
       (nnl2.hli.ad:tlet ((x (nnl2.hli.ad:ones #(5 5)))
						  (y (nnl2.hli.ad:zeros #(5 5))))
						  
		 (print (nnl2.hli.nn.ga.fitness:mse x y))) ;; (/ 1 (+ 1 (mse x y))) = (/ 1 2) = 0.5
		 
   Needed for GA
   The higher the result, the smaller the error" 	   

  (/ 1.0 (+ 1.0 (nnl2.hli.ad.r.loss:mse x y :track-graph nil :force t))))    
  
(in-package :nnl2.hli.nn.ga.crossover)

(defun uniform (parent-x parent-y &key (rate 0.5s0))
  "Returns a mixture of the first and second tensor
   Needed for GA 
   
   Args:
      parent-x: First AD tensor 
	  parent-y: Second AD tensor 
	  rate (&key): Crossover rate (default: 0.5s0)"
	  
  (nnl2.ffi:%ga-crossover-uniform parent-x parent-y rate))
  