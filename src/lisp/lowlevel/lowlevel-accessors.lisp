(in-package :nnl2.lli.ts)

(defun trefd (tensor at)
  (let ((dtype (case (nnl2.hli.ts:dtype tensor)
				 (:float64 :double)
				 (:float32 :float)
				 (:int32 :int))))
				 
    (cffi:mem-ref (nnl2.ffi:%lowlevel-tref tensor at) dtype)))
  
(defun trefw (tensor &rest at)
  (multiple-value-bind (shape rank) (nnl2.hli.ts:make-shape-pntr at)
    (cffi:mem-ref (nnl2.ffi:%lowlevel-tref-with-coords tensor shape rank) (case (nnl2.hli.ts:dtype tensor) (:float64 :double) (:float32 :float) (:int32 :int)))))
	    
  