(in-package :nnl2.ffi)

(cffi:defcenum tensor-type
  (:int32 0)
  (:float32 1)
  (:float64 2))
  
(cffi:defcstruct tensor  
  (dtype tensor-type)
  (data :pointer)
  (shape :pointer)
  (rank :int))
  
(cffi:defcfun ("fast_make_tensor" internal-fast-make-tensor) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))  
  
(cffi:defcfun ("make_tensor" internal-make-tensor) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))
  
(cffi:defcfun ("at" internal-tensor-at) :pointer
  (tensor :pointer)
  (indices :pointer)
  (sum-indices :uint8))
  
(cffi:defcfun ("get_tensor_metadata" tensor-metadata!) :void
  (tensor :pointer)
  (dtype :pointer)
  (rank :pointer)
  (shape :pointer))  
  
(defun make-tensor-from-shape (shape &key dtype)
  (let ((tensor-rank (length shape)))
    (cffi:with-foreign-object (share-ptr :int tensor-rank)
	  (loop for dim in shape
			for i from 0
			do (setf (cffi:mem-aref share-ptr :int i) dim))
	  
	  (internal-make-tensor share-ptr (length shape) dtype))))
	  
(defun %make-tensor-from-shape (shape &key dtype)
  (let ((tensor-rank (length shape)))
    (cffi:with-foreign-object (share-ptr :int tensor-rank)
	  (loop for dim in shape
			for i from 0
			do (setf (cffi:mem-aref share-ptr :int i) dim))
	  
	  (internal-fast-make-tensor share-ptr (length shape) dtype))))	  
  
(defun tref (tensor &rest indices)
  (let ((tensor-rank (cffi:foreign-slot-value tensor '(:struct tensor) 'rank))
	    (indices-rank (length indices)))
		
    (cffi:with-foreign-object (indices-ptr :int32 tensor-rank)
      (loop for idx in indices
            for i from 0
            do (setf (cffi:mem-aref indices-ptr :int32 i) idx))

      (let ((ptr (internal-tensor-at tensor indices-ptr indices-rank)))
        (cond 
		  ((and (not (cffi:null-pointer-p (cffi:foreign-slot-value ptr '(:struct tensor) 'data))) (not (cffi:null-pointer-p ptr))) ptr)
          ((not (cffi:null-pointer-p ptr))
		     (case (cffi:foreign-slot-value tensor '(:struct tensor) 'dtype)
			   (:int32 (cffi:mem-ref ptr :int32))
			   (:float32 (cffi:mem-ref ptr :float))
			   (:float64 (cffi:mem-ref ptr :double)))))))))	
  
(defun (setf tref) (new-value tensor &rest indices)
  (let ((tensor-rank (length indices)))
    (cffi:with-foreign-object (indices-ptr :int32 tensor-rank)
      (loop for idx in indices
            for i from 0
            do (setf (cffi:mem-aref indices-ptr :int32 i) idx))

      (let ((ptr (nnl2.ffi::internal-tensor-at tensor indices-ptr tensor-rank)))
        (case (cffi:foreign-slot-value tensor 'tensor 'dtype)
          (:int32 (setf (cffi:mem-ref ptr :int32) new-value))
          (:float32 (setf (cffi:mem-ref ptr :float) new-value))
          (:float64 (setf (cffi:mem-ref ptr :double) new-value)))))))
  
(defun print-tensor (tensor &optional (stream *standard-output*))
  (declare (optimize (speed 0) (safety 3) (space 3) (debug 3))
		   (type stream stream)
		   (type cffi:foreign-pointer tensor))

    (cffi:with-foreign-objects ((dtype :int) (rank :int) (shape-ptr :pointer))
	  (tensor-metadata! tensor dtype rank shape-ptr)
	
	  (format stream "#<NNL2-TENSOR ~a [~{~ax~}~a]>" 
	    (case (cffi:mem-ref dtype :int)
		  (0 :int32)
		  (1 :float32)
		  (2 :float64))
		  
		(loop for i from 0 below (1- (cffi:mem-ref rank :int))
			  collect (cffi:mem-aref shape-ptr :int i))
			  
		(cffi:mem-aref shape-ptr :int (1- (cffi:mem-ref rank :int))))))
	  
