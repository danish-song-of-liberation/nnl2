(in-package :nnl2.hli.ad)

(defun zeros (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  (nnl2.ffi:%ad-zeros shape rank dtype requires-grad name))))
	  
(defun ones (indices &key (dtype nnl2.system:*default-tensor-type*) requires-grad (name ""))
  (declare (type keyword dtype)
		   (type boolean requires-grad)
		   (type string name))
  
  (nnl2.hli:fastcall
    (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
	  (declare (type integer rank))
	  (nnl2.ffi:%ad-ones shape rank dtype requires-grad name))))
	  	  
	  
(cffi:defcfun ("nnl2_ad_get_data" data) :pointer
  (ad-tensor :pointer))  
  
(cffi:defcfun ("nnl2_ad_get_leaf" is-leaf) :bool
  (ad-tensor :pointer)) 

(cffi:defcfun ("nnl2_ad_get_requires_grad" requires-grad) :bool
  (ad-tensor :pointer))   
  