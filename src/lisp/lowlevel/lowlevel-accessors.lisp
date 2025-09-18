(in-package :nnl2.lli.ts)

(defun flat (tensor at)
  (let* ((dtype (case (nnl2.hli.ts:dtype tensor)
				  (:float64 :double)
				  (:float32 :float)
				  (:int32 :int)))
				  
		 (elem (nnl2.ffi:%lowlevel-tref tensor at)))
		 
	(if (cffi:null-pointer-p elem)
	  (error "Pointer can't be NULL")
	  (cffi:mem-ref elem dtype))))
	  
(defun (setf flat) (with tensor at)
  (let* ((cffi-type (case (nnl2.hli.ts:dtype tensor)
				      (:float64 :double)
				      (:float32 :float)
				      (:int32 :int)))
				   
		 (filler-pntr (cffi:foreign-alloc cffi-type)))

   (setf (cffi:mem-ref filler-pntr cffi-type) with)
   (nnl2.ffi:%lowlevel-tref-setter tensor at filler-pntr)))
  
(defun trefw (tensor &rest at)
  (multiple-value-bind (shape rank) (nnl2.hli.ts:make-shape-pntr at)
    (let ((elem (nnl2.ffi:%lowlevel-tref-with-coords tensor shape rank)))
	  (if (cffi:null-pointer-p elem)
	    (error "Pointer can't be NULL")
		(cffi:mem-ref elem (case (nnl2.hli.ts:dtype tensor) (:float64 :double) (:float32 :float) (:int32 :int)))))))
	    
(defun (setf trefw) (with tensor &rest at)
  (multiple-value-bind (shape rank) (nnl2.hli.ts:make-shape-pntr at)
    (let* ((cffi-type (case (nnl2.hli.ts:dtype tensor)
				        (:float64 :double)
				        (:float32 :float)
				        (:int32 :int)))
						
		   (filler-pntr (cffi:foreign-alloc cffi-type)))

     (setf (cffi:mem-ref filler-pntr cffi-type) with)
	 (nnl2.ffi:%lowlevel-tref-with-coords-setter tensor shape rank filler-pntr))))
		
(defun data (tensor)
  (nnl2.ffi:get-tensor-data tensor))
  
(defun mem-aref (tensor-data index &key (dtype :float64))
  (cffi:mem-aref tensor-data (case dtype (:float64 :double) (:float32 :float) (:int32 :int)) index))
  
(defmacro iterate-across-tensor-data ((iterator tensor) &body body)
  `(loop for i from 0 below (nnl2.hli.ts:size ,tensor)
	     do (let ((,iterator (flat ,tensor i)))
		      (progn ,@body))))
			  
(defmacro iterate-across-tensor-shape ((iterator tensor) &body body)
  `(loop for i from 0 below (nnl2.hli.ts:rank ,tensor)
		 do (let ((,iterator (nnl2.ffi:shape-at ,tensor i)))
			  (progn ,@body))))
			  			  
(defun alignment ()
  (nnl2.ffi:%get-mem-alignment))						  
			  