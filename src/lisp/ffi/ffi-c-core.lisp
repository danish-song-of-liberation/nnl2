(in-package :nnl2.ffi)

(cffi:defcfun ("init_system" nnl-init-system) :void)

(cffi:defcenum tensor-type
  :int32
  :float32
  :float64)
  
(cffi:defcenum nnl2-order
  (:nnl2colmajor 101)
  (:nnl2rowmajor 102))
  
(cffi:defcenum nnl2-transpose
  (:nnl2notrans 111)
  (:nnl2trans 112))  
  
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

(cffi:defcfun ("at" %tref) :pointer
  (tensor :pointer)
  (shape :pointer)
  (rank :int))
  
(cffi:defcfun ("at_set" %tref-setter) :void
  (tensor :pointer)
  (shape :pointer)
  (rank :int)
  (change-to :pointer))  
  
(cffi:defcfun ("lisp_call_scaleinplace" %scale!) :void
  (tensor :pointer)
  (multiplier :float))  
   
(cffi:defcfun ("lisp_call_scale" %scale) :pointer
  (tensor :pointer)
  (multiplier :float))   
  
(cffi:defcfun ("lisp_call_sgemm" %sgemm) :pointer
  (order nnl2-order)
  (transa nnl2-transpose)
  (transb nnl2-transpose)
  (m :int)
  (n :int)
  (k :int)
  (alpha :float)
  (a :pointer)
  (lda :int)
  (b :pointer)
  (ldb :int)
  (beta :float))
  
(cffi:defcfun ("lisp_call_dgemm" %dgemm) :pointer
  (order nnl2-order)
  (transa nnl2-transpose)
  (transb nnl2-transpose)
  (m :int)
  (n :int)
  (k :int)
  (alpha :double)
  (a :pointer)
  (lda :int)
  (b :pointer)
  (ldb :int)
  (beta :double))  
  
(cffi:defcfun ("gemm" %gemm) :pointer
  (order nnl2-order)
  (transa nnl2-transpose)
  (transb nnl2-transpose)
  (m :int)
  (n :int)
  (k :int)
  (alpha :double)
  (a :pointer)
  (lda :int)
  (b :pointer)
  (ldb :int)
  (beta :double))   

(cffi:defcfun ("gemminplace" %gemm!) :pointer
  (order nnl2-order)
  (transa nnl2-transpose)
  (transb nnl2-transpose)
  (m :int)
  (n :int)
  (k :int)
  (alpha :double)
  (a :pointer)
  (lda :int)
  (b :pointer)
  (ldb :int)
  (beta :double)
  (c :pointer)
  (ldc :int))  
  
(cffi:defcfun ("lisp_call_addinplace" %+=) :void
  (summand :pointer)
  (sumend :pointer))
  
(cffi:defcfun ("lisp_call_subinplace" %-=) :void
  (summand :pointer)
  (sumend :pointer))  
  
(cffi:defcfun ("lisp_call_add" %+) :pointer
  (summand :pointer)
  (addend :pointer))  
  
(cffi:defcfun ("lisp_call_sub" %-) :pointer
  (summand :pointer)
  (addend :pointer))    
  
(cffi:defcfun ("lisp_call_mulinplace" %*=) :void
  (multiplicand :pointer)
  (multiplier :pointer))  
  
(cffi:defcfun ("lisp_call_divinplace" %/=) :void
  (dividend :pointer)
  (divisor :pointer))  
  
(cffi:defcfun ("lisp_call_mul" %*) :pointer
  (multiplicand :pointer)
  (multiplier :pointer))

(cffi:defcfun ("lisp_call_div" %/) :pointer
  (dividend :pointer)
  (divisor :pointer))
  
(cffi:defcfun ("lisp_call_powinplace" %^=) :void
  (base :pointer)
  (exponent :pointer))  
  
(cffi:defcfun ("lisp_call_expinplace" %.exp!) :void
  (tensor :pointer))
  
(cffi:defcfun ("lisp_call_pow" %.^) :pointer
  (base :pointer)
  (exponent :pointer))    
  
(cffi:defcfun ("lisp_call_exp" %.exp) :pointer
  (tensor :pointer))

(cffi:defcfun ("lisp_call_loginplace" %.log!) :void
  (tensor :pointer))  

(cffi:defcfun ("lisp_call_log" %.log) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("zeros_like" %zeros-like) :pointer
  (tensor :pointer)) 

(cffi:defcfun ("ones_like" %ones-like) :pointer
  (tensor :pointer))   
  
(cffi:defcfun ("full_like" %full-like) :pointer
  (tensor :pointer)
  (filler :pointer))
									  
(cffi:defcfun ("lisp_call_maxinplace" %.max!) :void
  (tensora :pointer)
  (tensorb :pointer))  
  
(cffi:defcfun ("lisp_call_mininplace" %.min!) :void
  (tensora :pointer)
  (tensorb :pointer))  

(cffi:defcfun ("lisp_call_max" %.max) :pointer
  (tensora :pointer)
  (tensorb :pointer))    
  
(cffi:defcfun ("lisp_call_min" %.min) :pointer
  (tensora :pointer)
  (tensorb :pointer))    
  
(cffi:defcfun ("free_tensor" free-tensor) :void
  (tensor :pointer))     
  
(cffi:defcfun ("print_tensor" print-tensor) :void
  (tensor :pointer))  
  
(cffi:defcfun ("get_tensor_rank" get-tensor-rank) :int
  (tensor :pointer))  
  
(cffi:defcfun ("get_tensor_dtype" get-tensor-dtype) tensor-type
  (tensor :pointer))    
  
(cffi:defcfun ("get_tensor_dtype" get-int-tensor-dtype) :int
  (tensor :pointer))     

(cffi:defcfun ("get_tensor_shape" get-pointer-to-tensor-shape) :pointer
  (tensor :pointer))        
  
(cffi:defcfun ("lisp_call_debug_blas_sgemminplace" %%internal-debug-sgemm!) :void
  (check-to :unsigned-long))  
  
(cffi:defcfun ("get_size" %get-size) :int
  (tensor :pointer))

(cffi:defcfun ("get_size_in_bytes" %get-size-in-bytes) :int
  (tensor :pointer))  
  
(cffi:defcfun ("get_mem_alignment" %%%internal-lisp-call-get-mem-alignment) :int)
  
(nnl-init-system)
