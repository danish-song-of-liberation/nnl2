(in-package :nnl2.ffi)

(cffi:defcfun ("init_system" nnl-init-system) :void)

(cffi:defcenum tensor-type
  :int32
  :float32
  :float64)
  
(cffi:defcstruct tensor
  (tensor-type tensor-type)
  (data :pointer)
  (shape :pointer)
  (rank :int))
  
(cffi:defcfun ("lisp_call_empty" %empty) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))
  
(cffi:defcfun ("lisp_call_zeros" %zeros) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))  
  
(cffi:defcfun ("lisp_call_ones" %ones) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))  
    
(cffi:defcfun ("lisp_call_full" %full) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (filler :pointer))  
  
(cffi:defcfun ("free_tensor" free-tensor) :void
  (tensor :pointer))    

(nnl-init-system)
