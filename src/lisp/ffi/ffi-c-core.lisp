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
  
(cffi:defcfun ("make_tensor" make-tensor) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))
  