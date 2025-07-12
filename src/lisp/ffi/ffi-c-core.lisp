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
  
(cffi:defcfun ("fast_at" internal-fast-tensor-at) :pointer
  (tensor :pointer)
  (indices :pointer)
  (sum-indices :uint8))
  
(cffi:defcfun ("at" internal-tensor-at) :pointer
  (tensor :pointer)
  (indices :pointer)
  (sum-indices :uint8))  
  
(cffi:defcfun ("get_tensor_metadata" tensor-metadata!) :void
  (tensor :pointer)
  (dtype :pointer)
  (rank :pointer)
  (shape :pointer))  
  
(cffi:defcfun ("function_of_liberation" free) :void
  (tensor :pointer))  
  
(cffi:defcfun ("debug_tensor" debug-tensor) :void 
  (tensor tensor))  
  
(defun warn-about-indices (num)
  (warn "Are you absolutely SURE you want to create a tensor with `~d` (!) dimensions?~%~%   Y - Yes (Can lead to very bad consequences)~%   N - No (Tensor will not be created and function will return nil)~%   E - Exit (Program will be closed)~%~%   Default - Exit~%~%   Attention: This can be VERY, VERY dangerous.~%~%   If you don't want to see these messages, write `(setq nnl2.system:*silent-mode* t)`~%" num)
  
  (let ((answear (read-char)))
  
    (cond
	  ((or (char= answear #\Y) (char= answear #\y)) 0)
	  ((or (char= answear #\N) (char= answear #\n)) 1)
	  ((or (char= answear #\E) (char= answear #\e)) 2)
	  
	  (t 2))))
  
(defun make-tensor-from-shape (shape &key dtype)
  (let ((tensor-rank (length shape)))
    (when (and (not nnl2.system:*silent-mode*) (>= tensor-rank 10))
	  (let ((answear (warn-about-indices tensor-rank)))
	    (case answear
		  (1 (return-from make-tensor-from-shape nil))
		  (2 #+sbcl (sb-ext:exit)
			 #-sbcl (error "you don't use sbcl and your implementation doesn't have `sb-ext:exit` so you have to exit this way")))))
	
    (cffi:with-foreign-object (share-ptr :int tensor-rank)
	  (loop for dim in shape
			for i from 0
			do (setf (cffi:mem-aref share-ptr :int i) dim))
	  
	  (internal-make-tensor share-ptr (length shape) dtype))))
	  
(defun fast-make-tensor-from-shape (shape &key dtype)
  (let ((tensor-rank (length shape)))
    (cffi:with-foreign-object (share-ptr :int tensor-rank)
	  (loop for dim in shape
			for i from 0
			do (setf (cffi:mem-aref share-ptr :int i) dim))
	  
	  (internal-fast-make-tensor share-ptr (length shape) dtype))))	  
    
(defun fast-tref (tensor &rest indices)
  (let ((tensor-rank (cffi:foreign-slot-value tensor '(:struct tensor) 'rank))
	    (indices-rank (length indices)))
		
    (cffi:with-foreign-object (indices-ptr :int32 tensor-rank)
      (loop for idx in indices
            for i from 0
            do (setf (cffi:mem-aref indices-ptr :int32 i) idx))

      (let ((ptr (internal-fast-tensor-at tensor indices-ptr indices-rank)))
        (cond 
		  ((and (not (cffi:null-pointer-p (cffi:foreign-slot-value ptr '(:struct tensor) 'data))) (not (cffi:null-pointer-p ptr))) ptr)
          ((not (cffi:null-pointer-p ptr))
		     (case (cffi:foreign-slot-value tensor '(:struct tensor) 'dtype)
			   (:int32 (cffi:mem-ref ptr :int32))
			   (:float32 (cffi:mem-ref ptr :float))
			   (:float64 (cffi:mem-ref ptr :double)))))))))
  
(defun (setf fast-tref) (new-value tensor &rest indices)
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
		  
(defun print-1d-tensor (tensor rows stream type)
  (dotimes (i rows)
    (format stream "~%      ")
	
    (format stream "~f" (tref tensor i))))
		  
(defun print-2d-tensor (tensor rows cols stream type)
  (dotimes (i rows)
    (format stream "~%")
  
    (dotimes (j cols)
	  (format stream "      ~a" (tref tensor i j)))))
	  
;; I'm so happy for you that you decided to dig into the code's guts

(defun print-tensor (tensor &key (stream *standard-output*))
  "warning: code is not ready"

  (declare (optimize (speed 3) (safety 0) (space 0) (debug 0))
		   (type stream stream)
		   (type cffi:foreign-pointer tensor))

  (cffi:with-foreign-objects ((dtype :int) (rank :int) (shape-ptr :pointer))
	(tensor-metadata! tensor dtype rank shape-ptr)
	
	(let* ((dtype-val (cffi:mem-ref dtype :int))
		   (tensor-rank (cffi:mem-ref rank :int))
		   (shape-array (cffi:mem-ref shape-ptr :pointer))
		   
		   (type (case (cffi:mem-ref dtype :int)
				   (0 :int32)
				   (1 :float32)
				   (2 :float64)))
		 
		   (shape (loop for i from 0 below (1- tensor-rank)
						collect (cffi:mem-aref shape-array :int i)))
						
		   (last-shape (cffi:mem-aref shape-array :int (1- tensor-rank))))
	
	  (cond 
	    ((= tensor-rank 1)
		   (format stream "#<NNL2-TENSOR ~a [~a]:" type last-shape)
		   
		   (print-1d-tensor tensor last-shape stream type)
		   
		   (format stream ">~%"))
		
		((= tensor-rank 2)
		   (format stream "#<NNL2-TENSOR ~a [~{~ax~}~a]:" type shape last-shape)
		   
		   (print-2d-tensor tensor (car shape) last-shape stream type)
		   
		   (format stream ">~%"))
						   
		
		(t (format stream "#<NNL2-TENSOR ~a [~{~ax~}~a]>~%" type shape last-shape))))))
	  
