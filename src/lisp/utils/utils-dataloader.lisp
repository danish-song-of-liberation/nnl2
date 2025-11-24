(in-package :nnl2.utils)

(defstruct nnl2-internal-dataloader
  (samples nil)
  (labels nil)
  (batch-size nnl2.system:*default-batch-size*)
  (size 0)
  (max-epoch 0))
  
(defun dataloader (samples labels &key (batch-size nnl2.system:*default-batch-size*))
  (let ((nnl2-dataloader (make-nnl2-internal-dataloader :samples samples :labels labels :batch-size batch-size))
		(size (nnl2.hli.ts:nrows samples)))
		
    (setf (nnl2-internal-dataloader-max-epoch nnl2-dataloader) (floor (/ size batch-size)))
	(setf (nnl2-internal-dataloader-size nnl2-dataloader) size)
		  
    nnl2-dataloader))  
  
(defmacro batch (dataloader (batch-x batch-y) epoch &body body)
  "an elegant, hygienic macro that doesn't violate any conventions (todo refactor)"
  `(if (> (nnl2-internal-dataloader-max-epoch ,dataloader) ,epoch)
     (let* ((predcomp1 (nnl2-internal-dataloader-batch-size ,dataloader))
			(predcomp2 (* predcomp1 ,epoch)))
		
		(nnl2.hli.ts:tlet ((,batch-x (nnl2.hli.ts.utils:narrow (nnl2.hli.ad:data (nnl2-internal-dataloader-samples ,dataloader)) :dim 0 :start predcomp2 :len predcomp1))
						   (,batch-y (nnl2.hli.ts.utils:narrow (nnl2.hli.ad:data (nnl2-internal-dataloader-labels ,dataloader)) :dim 0 :start predcomp2 :len predcomp1)))
					 
		  ,@body))
	  
	 (nnl2.log:warning (format nil "Reached max epoch in dataloader"))))
	 
(defmacro process (dataloader (batch-x batch-y) &body body)
  `(dotimes (i (nnl2-internal-dataloader-max-epoch ,dataloader))
     (batch ,dataloader (,batch-x ,batch-y) i ,@body)))
 