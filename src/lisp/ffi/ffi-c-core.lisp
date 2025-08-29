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
  
(cffi:defcfun ("ones" %ones) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))  
    
(cffi:defcfun ("lisp_call_full" %full) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (filler :pointer))  

(cffi:defcfun ("lisp_call_view" %view) :pointer
  (tensor :pointer)
  (shape :pointer)
  (rank :int))
  
(cffi:defcfun ("lisp_call_tref_setter" %tref-setter) :void
  (tensor :pointer)
  (shape :pointer)
  (rank :int)
  (change-to :pointer)
  (is-tensor :bool))

(cffi:defcfun ("lisp_call_tref_getter" %tref-getter) :pointer
  (tensor :pointer)
  (shape :pointer)
  (rank :int))  
  
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
  
(cffi:defcfun ("lisp_call_add_incf_inplace" %.+/incf!) :void
  (tensor :pointer)
  (increment :pointer))  
  
(cffi:defcfun ("lisp_call_sub_decf_inplace" %.-/decf!) :void
  (tensor :pointer)
  (increment :pointer))    
  
(cffi:defcfun ("lisp_call_add_broadcasting_inplace" %.+/broadcasting!) :void  
  (summand :pointer)
  (sumend :pointer))
  
(cffi:defcfun ("lisp_call_sub_broadcasting_inplace" %.-/broadcasting!) :void  
  (minuend :pointer)
  (subtrahend :pointer))  
  
(cffi:defcfun ("lisp_call_add" %+) :pointer
  (summand :pointer)
  (addend :pointer))  
  
(cffi:defcfun ("lisp_call_add_broadcasting" %.+/broadcasting) :pointer
  (summand :pointer)
  (sumend :pointer))
  
(cffi:defcfun ("lisp_call_add_incf" %.+/incf) :pointer
  (tensor :pointer)
  (increment :pointer))  
  
(cffi:defcfun ("lisp_call_sub" %-) :pointer
  (summand :pointer)
  (addend :pointer))    
  
(cffi:defcfun ("lisp_call_sub_broadcasting" %.-/broadcasting) :pointer
  (minuend :pointer)
  (subtrahend :pointer))
 
(cffi:defcfun ("lisp_call_sub_decf" %.-/decf) :pointer
  (tensor :pointer)
  (increment :pointer))  
  
(cffi:defcfun ("lisp_call_mulinplace" %*=) :void
  (multiplicand :pointer)
  (multiplier :pointer))  
  
(cffi:defcfun ("lisp_call_mul_mulf_inplace" %.*/mulf!) :void
  (tensor :pointer)
  (multiplier :pointer))  
  
(cffi:defcfun ("lisp_call_mul_broadcasting_inplace" %.*/broadcasting!) :void
  (multiplicand :pointer)
  (multiplier :pointer)) 
  
(cffi:defcfun ("lisp_call_divinplace" %/=) :void
  (dividend :pointer)
  (divisor :pointer))  
    
(cffi:defcfun ("lisp_call_div_divf_inplace" %.//divf!) :void
  (tensor :pointer)
  (dif :pointer))

(cffi:defcfun ("lisp_call_div_broadcasting_inplace" %.//broadcasting!) :void
  (dividend :pointer)
  (divisor :pointer))   
  
(cffi:defcfun ("lisp_call_mul" %*) :pointer
  (multiplicand :pointer)
  (multiplier :pointer))
  
(cffi:defcfun ("lisp_call_mul_mulf" %.*/mulf) :pointer
  (tensor :pointer)
  (multiplier :pointer)) 

(cffi:defcfun ("lisp_call_mul_broadcasting" %.*/broadcasting) :pointer   
  (multiplicand :pointer)
  (multiplier :pointer))

(cffi:defcfun ("lisp_call_div" %/) :pointer
  (dividend :pointer)
  (divisor :pointer))
  
(cffi:defcfun ("lisp_call_div_divf" %.//divf) :pointer
  (tensor :pointer)
  (dif :pointer))  
  
(cffi:defcfun ("lisp_call_div_broadcasting" %.//broadcasting) :pointer
  (dividend :pointer)
  (divisor :pointer))    
  
(cffi:defcfun ("lisp_call_powinplace" %^=) :void
  (base :pointer)
  (exponent :pointer))  
  
(cffi:defcfun ("lisp_call_pow_powf_inplace" %.^/powf!) :void
  (tensor :pointer)
  (powf :pointer))  
  
(cffi:defcfun ("lisp_call_pow_broadcasting_inplace" %.^/broadcasting!) :void
  (base :pointer)
  (exponent :pointer))  
  
(cffi:defcfun ("lisp_call_expinplace" %.exp!) :void
  (tensor :pointer))
  
(cffi:defcfun ("lisp_call_pow" %.^) :pointer
  (base :pointer)
  (exponent :pointer))    
  
(cffi:defcfun ("lisp_call_pow_powf" %.^/powf) :pointer
  (tensor :pointer)
  (powf :pointer)) 

(cffi:defcfun ("lisp_call_pow_broadcasting" %.^/broadcasting) :pointer
  (base :pointer)
  (exponent :pointer))    
  
(cffi:defcfun ("lisp_call_exp" %.exp) :pointer
  (tensor :pointer))

(cffi:defcfun ("lisp_call_loginplace" %.log!) :void
  (tensor :pointer))  

(cffi:defcfun ("lisp_call_log" %.log) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_axpy_inplace" %axpy!) :void
  (summand :pointer)
  (sumend :pointer)
  (alpha :float))  
  
(cffi:defcfun ("lisp_call_axpy" %axpy) :pointer
  (summand :pointer)
  (sumend :pointer)
  (alpha :float))    
  
(cffi:defcfun ("lisp_call_axpf_inplace" %axpy/axpf!) :void
  (summand :pointer)
  (sumend :pointer)
  (alpha :float))    
  
(cffi:defcfun ("lisp_call_axpf" %axpy/axpf) :pointer
  (summand :pointer)
  (sumend :pointer)
  (alpha :float))      
  
(cffi:defcfun ("lisp_call_axpy_broadcasting_inplace" %axpy/broadcasting!) :void
  (summand :pointer)
  (sumend :pointer)
  (alpha :float))  
  
(cffi:defcfun ("lisp_call_axpy_broadcasting" %axpy/broadcasting) :pointer
  (summand :pointer)
  (sumend :pointer)
  (alpha :float))  
  
(cffi:defcfun ("empty_like" %empty-like) :pointer
  (tensor :pointer))   
  
(cffi:defcfun ("zeros_like" %zeros-like) :pointer
  (tensor :pointer)) 

(cffi:defcfun ("ones_like" %ones-like) :pointer
  (tensor :pointer))   
  
(cffi:defcfun ("full_like" %full-like) :pointer
  (tensor :pointer)
  (filler :pointer))
  
(cffi:defcfun ("randn_like" %randn-like) :pointer
  (tensor :pointer)
  (from :pointer)
  (to :pointer))

(cffi:defcfun ("lisp_call_xavier" %xavier) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (in :int)
  (out :int)
  (gain :float)
  (distribution :float))  
  
(cffi:defcfun ("lisp_call_hstack" %hstack) :pointer
  (tensora :pointer)
  (tensorb :pointer))
  
(cffi:defcfun ("lisp_call_vstack" %vstack) :pointer
  (tensora :pointer)
  (tensorb :pointer))  
  
(cffi:defcfun ("lisp_call_concat" %concat) :pointer
  (tensora :pointer)
  (tensorb :pointer)
  (axis :int))

(cffi:defcfun ("lisp_call_randn" %randn) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (from :pointer)
  (to :pointer))  

(cffi:defcfun ("lisp_call_maxinplace" %.max!) :void
  (tensora :pointer)
  (tensorb :pointer))    
  
(cffi:defcfun ("lisp_call_max_maxf_inplace" %.max/maxf!) :void
  (tensor :pointer)
  (maxf :pointer))    
  
(cffi:defcfun ("lisp_call_max_broadcasting_inplace" %.max/broadcasting!) :void
  (a :pointer)
  (b :pointer))    
  
(cffi:defcfun ("lisp_call_mininplace" %.min!) :void
  (tensora :pointer)
  (tensorb :pointer))  
  
(cffi:defcfun ("lisp_call_min_minf_inplace" %.min/minf!) :void
  (tensor :pointer)
  (minf :pointer))  
  
(cffi:defcfun ("lisp_call_min_broadcasting_inplace" %.min/broadcasting!) :void
  (a :pointer)
  (b :pointer))      

(cffi:defcfun ("lisp_call_max" %.max) :pointer
  (tensora :pointer)
  (tensorb :pointer))   

(cffi:defcfun ("lisp_call_max_maxf" %.max/maxf) :pointer
  (tensor :pointer)
  (maxf :pointer))  
  
(cffi:defcfun ("lisp_call_max_broadcasting" %.max/broadcasting) :pointer
  (a :pointer)
  (b :pointer))

(cffi:defcfun ("lisp_call_min_broadcasting" %.min/broadcasting) :pointer
  (a :pointer)
  (b :pointer))      
  
(cffi:defcfun ("lisp_call_min" %.min) :pointer
  (tensora :pointer)
  (tensorb :pointer)) 

(cffi:defcfun ("lisp_call_min_minf" %.min/minf) :pointer
  (tensor :pointer)
  (minf :pointer))    
  
(cffi:defcfun ("lisp_call_absinplace" %.abs!) :void
  (tensora :pointer))
  
(cffi:defcfun ("lisp_call_abs" %.abs) :pointer
  (tensora :pointer))
  
(cffi:defcfun ("lisp_call_reluinplace" %.relu!) :void
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_relu" %.relu) :pointer
  (tensor :pointer)) 

(cffi:defcfun ("lisp_call_leakyreluinplace" %.leaky-relu!) :void
  (tensor :pointer)
  (alpha :float))  
  
(cffi:defcfun ("lisp_call_leakyrelu" %.leaky-relu) :pointer
  (tensor :pointer)
  (alpha :float))    
  
(cffi:defcfun ("lisp_call_sigmoidinplace" %.sigmoid!) :void
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_sigmoid" %.sigmoid) :pointer
  (tensor :pointer))    
  
(cffi:defcfun ("lisp_call_tanhinplace" %.tanh!) :void
  (tensor :pointer))
  
(cffi:defcfun ("lisp_call_tanh" %.tanh) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_transposeinplace" %transpose!) :void
  (tensor :pointer))

(cffi:defcfun ("lisp_call_transpose" %transpose) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_sum" %sum) :void
  (tensor :pointer)
  (axes :pointer)
  (num-axes :int)
  (out :pointer))  
  
(cffi:defcfun ("lisp_call_l2norm" %l2norm) :void
  (tensor :pointer)
  (axes :pointer)
  (num-axes :int)
  (out :pointer))   

(cffi:defcfun ("lisp_call_copy" %copy) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("nnl2_free_tensor" free-tensor) :void
  (tensor :pointer))     
  
(cffi:defcfun ("nnl2_print_tensor" print-tensor) :void
  (tensor :pointer)
  (full-print :bool)
  (max-rows :int)
  (max-cols :int)
  (show-rows :int)
  (show-cols :int))
  
(cffi:defcfun ("get_tensor_rank" get-tensor-rank) :int
  (tensor :pointer))  
  
(cffi:defcfun ("get_tensor_dtype" get-tensor-dtype) tensor-type
  (tensor :pointer))    
  
(cffi:defcfun ("get_tensor_dtype" get-int-tensor-dtype) :int
  (tensor :pointer))     

(cffi:defcfun ("get_tensor_shape" get-pointer-to-tensor-shape) :pointer
  (tensor :pointer))        
  
(cffi:defcfun ("get_tensor_data" get-tensor-data) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_debug_blas_sgemminplace" %%internal-debug-sgemm!) :void
  (check-to :unsigned-long))  
  
(cffi:defcfun ("get_size" %get-size) :int
  (tensor :pointer))

(cffi:defcfun ("get_size_in_bytes" %get-size-in-bytes) :int
  (tensor :pointer))  

(cffi:defcfun ("get_mem_alignment" %%%internal-lisp-call-get-mem-alignment) :int)

(cffi:defcfun ("make_tensor_from_flatten" %make-tensor-from-flatten) :pointer
  (data :pointer)
  (num-elems-data :int)
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))

(cffi:defcfun ("internal_get_float_data_tensor" %%%internal-get-float-data-tensor) :pointer
  (tensor :pointer))
 
(cffi:defcfun ("nnl2_set_view_backend" %set-view-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_tref_setter_backend" %set-tref-setter-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("nnl2_set_tref_getter_backend" %set-tref-getter-backend) :void
  (backend-name :string))     
 
(cffi:defcfun ("set_inplace_fill_backend" %set-inplace-fill-backend) :void
  (backend-name :string))  

(cffi:defcfun ("nnl2_set_empty_backend" %set-empty-backend) :void
  (backend-name :string))  

(cffi:defcfun ("nnl2_set_zeros_backend" %set-zeros-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_sgemminplace_backend" %set-sgemminplace-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_dgemminplace_backend" %set-dgemminplace-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_addinplace_backend" %set-addinplace-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_subinplace_backend" %set-subinplace-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_add_backend" %set-add-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_sub_backend" %set-sub-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_mul_inplace_backend" %set-mulinplace-backend) :void
  (backend-name :string))  
    
(cffi:defcfun ("set_div_inplace_backend" %set-divinplace-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_mul_backend" %set-mul-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_div_backend" %set-div-backend) :void
  (backend-name :string))    	
  
(cffi:defcfun ("set_powinplace_backend" %set-powinplace-backend) :void
  (backend-name :string))  
    
(cffi:defcfun ("set_expinplace_backend" %set-expinplace-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_pow_backend" %set-pow-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_exp_backend" %set-exp-backend) :void
  (backend-name :string))  
  
(cffi:defcfun ("set_loginplace_backend" %set-loginplace-backend) :void
  (backend-name :string))  
    
(cffi:defcfun ("set_log_backend" %set-log-backend) :void
  (backend-name :string))  
 
(cffi:defcfun ("set_scaleinplace_backend" %set-scaleinplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_scale_backend" %set-scale-backend) :void
  (backend-name :string))  
     
(cffi:defcfun ("set_maxinplace_backend" %set-maxinplace-backend) :void
  (backend-name :string))  
  	 
(cffi:defcfun ("set_mininplace_backend" %set-mininplace-backend) :void
  (backend-name :string))  
  	 
(cffi:defcfun ("set_max_backend" %set-max-backend) :void
  (backend-name :string))  
  	 
(cffi:defcfun ("set_min_backend" %set-min-backend) :void
  (backend-name :string))  	 
  
(cffi:defcfun ("set_abs_backend" %set-abs-backend) :void
  (backend-name :string))  
  
(cffi:defcfun ("set_abs_inplace_backend" %set-abs-inplace-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_hstack_backend" %set-hstack-backend) :void
  (backend-name :string))  
    
(cffi:defcfun ("set_vstack_backend" %set-vstack-backend) :void
  (backend-name :string))  
  
(cffi:defcfun ("set_reluinplace_backend" %set-reluinplace-backend) :void
  (backend-name :string))  
    
(cffi:defcfun ("set_relu_backend" %set-relu-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_leakyreluinplace_backend" %set-leakyreluinplace-backend) :void
  (backend-name :string))  
  
(cffi:defcfun ("set_leakyrelu_backend" %set-leakyrelu-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_sigmoidinplace_backend" %set-sigmoidinplace-backend) :void
  (backend-name :string))  
      
(cffi:defcfun ("set_sigmoid_backend" %set-sigmoid-backend) :void
  (backend-name :string))  
  
(cffi:defcfun ("set_tanhinplace_backend" %set-tanhinplace-backend) :void
  (backend-name :string))  
    
(cffi:defcfun ("set_tanh_backend" %set-tanh-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_concat_backend" %set-concat-backend) :void
  (backend-name :string))  
    
(cffi:defcfun ("set_randn_backend" %set-randn-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_xavier_backend" %set-xavier-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_transposeinplace_backend" %set-transposeinplace-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_transpose_backend" %set-transpose-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_sum_backend" %set-sum-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_l2norm_backend" %set-l2norm-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_copy_backend" %set-copy-backend) :void
  (backend-name :string))  
 
(cffi:defcfun ("set_add_incf_inplace_backend" %set-add-incf-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_add_incf_backend" %set-add-incf-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_sub_decf_inplace_backend" %set-sub-decf-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_sub_decf_backend" %set-sub-decf-backend) :void
  (backend-name :string))  
          
(cffi:defcfun ("set_mul_mulf_inplace_backend" %set-mul-mulf-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_mul_mulf_backend" %set-mul-mulf-backend) :void
  (backend-name :string))  
     
(cffi:defcfun ("set_div_divf_inplace_backend" %set-div-divf-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_div_divf_backend" %set-div-divf-backend) :void
  (backend-name :string))  
     	 
(cffi:defcfun ("set_pow_powf_inplace_backend" %set-pow-powf-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_pow_powf_backend" %set-pow-powf-backend) :void
  (backend-name :string))  
    
(cffi:defcfun ("set_max_maxf_inplace_backend" %set-max-maxf-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_max_maxf_backend" %set-max-maxf-backend) :void
  (backend-name :string))  
     	
(cffi:defcfun ("set_min_minf_inplace_backend" %set-min-minf-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_min_minf_backend" %set-min-minf-backend) :void
  (backend-name :string))  
     		
(cffi:defcfun ("set_add_broadcasting_inplace_backend" %set-add-broadcasting-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_add_broadcasting_backend" %set-add-broadcasting-backend) :void
  (backend-name :string))  
 
(cffi:defcfun ("set_sub_broadcasting_inplace_backend" %set-sub-broadcasting-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_sub_broadcasting_backend" %set-sub-broadcasting-backend) :void
  (backend-name :string))   

(cffi:defcfun ("set_mul_broadcasting_inplace_backend" %set-mul-broadcasting-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_mul_broadcasting_backend" %set-mul-broadcasting-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_div_broadcasting_inplace_backend" %set-div-broadcasting-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_div_broadcasting_backend" %set-div-broadcasting-backend) :void
  (backend-name :string))  
  
(cffi:defcfun ("set_pow_broadcasting_inplace_backend" %set-pow-broadcasting-inplace-backend) :void
  (backend-name :string))  
   
(cffi:defcfun ("set_pow_broadcasting_backend" %set-pow-broadcasting-backend) :void
  (backend-name :string))    

(cffi:defcfun ("set_max_broadcasting_backend" %set-max-broadcasting-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_max_broadcasting_inplace_backend" %set-max-broadcasting-inplace-backend) :void
  (backend-name :string))    
   
(cffi:defcfun ("set_min_broadcasting_backend" %set-min-broadcasting-backend) :void
  (backend-name :string))    

(cffi:defcfun ("set_min_broadcasting_inplace_backend" %set-min-broadcasting-inplace-backend) :void
  (backend-name :string))      
  
(cffi:defcfun ("set_fill_tensor_with_data_backend" %set-fill-tensor-with-data-backend) :void
  (backend-name :string))  
  
(cffi:defcfun ("set_axpy_inplace_backend" %set-axpy-inplace-backend) :void
  (backend-name :string))    
  
(cffi:defcfun ("set_axpy_backend" %set-axpy-backend) :void
  (backend-name :string))      
  
(cffi:defcfun ("set_axpf_inplace_backend" %set-axpf-inplace-backend) :void
  (backend-name :string))      
  
(cffi:defcfun ("set_axpf_backend" %set-axpf-backend) :void
  (backend-name :string))      
  
(cffi:defcfun ("set_axpy_broadcasting_inplace_backend" %set-axpy-broadcasting-inplace-backend) :void
  (backend-name :string))      
  
(cffi:defcfun ("set_axpy_broadcasting_backend" %set-axpy-broadcasting-backend) :void
  (backend-name :string)) 
  
(cffi:defcfun ("nnl2_get_view_backend" %get-view-backend) :string)    
(cffi:defcfun ("nnl2_get_tref_getter_backend" %get-tref-getter-backend) :string)    
(cffi:defcfun ("nnl2_get_empty_backend" %get-empty-backend) :string)  
(cffi:defcfun ("nnl2_get_zeros_backend" %get-zeros-backend) :string)
(cffi:defcfun ("get_inplace_fill_backend" %get-inplace-fill-backend) :string)
(cffi:defcfun ("get_gemm_backend" %get-gemm-backend) :string)
(cffi:defcfun ("get_addinplace_backend" %get-addinplace-backend) :string)
(cffi:defcfun ("get_subinplace_backend" %get-subinplace-backend) :string)
(cffi:defcfun ("get_add_backend" %get-add-backend) :string)    
(cffi:defcfun ("get_sub_backend" %get-sub-backend) :string)   
(cffi:defcfun ("get_mulinplace_backend" %get-mulinplace-backend) :string)    
(cffi:defcfun ("get_divinplace_backend" %get-divinplace-backend) :string)      
(cffi:defcfun ("get_mul_backend" %get-mul-backend) :string)   
(cffi:defcfun ("get_div_backend" %get-div-backend) :string)    
(cffi:defcfun ("get_powinplace_backend" %get-powinplace-backend) :string)   
(cffi:defcfun ("get_pow_backend" %get-pow-backend) :string)    
(cffi:defcfun ("get_expinplace_backend" %get-expinplace-backend) :string)
(cffi:defcfun ("get_exp_backend" %get-exp-backend) :string)
(cffi:defcfun ("get_loginplace_backend" %get-loginplace-backend) :string)
(cffi:defcfun ("get_log_backend" %get-log-backend) :string)
(cffi:defcfun ("get_scaleinplace_backend" %get-scaleinplace-backend) :string) 
(cffi:defcfun ("get_scale_backend" %get-scale-backend) :string)  
(cffi:defcfun ("get_maxinplace_backend" %get-maxinplace-backend) :string)  
(cffi:defcfun ("get_mininplace_backend" %get-mininplace-backend) :string)  
(cffi:defcfun ("get_max_backend" %get-max-backend) :string)   
(cffi:defcfun ("get_min_backend" %get-min-backend) :string)  
(cffi:defcfun ("get_absinplace_backend" %get-absinplace-backend) :string) 
(cffi:defcfun ("get_abs_backend" %get-abs-backend) :string)   
(cffi:defcfun ("get_hstack_backend" %get-hstack-backend) :string)   
(cffi:defcfun ("get_vstack_backend" %get-vstack-backend) :string)   
(cffi:defcfun ("get_reluinplace_backend" %get-reluinplace-backend) :string)  
(cffi:defcfun ("get_relu_backend" %get-relu-backend) :string) 
(cffi:defcfun ("get_leakyreluinplace_backend" %get-leakyreluinplace-backend) :string)  
(cffi:defcfun ("get_leakyrelu_backend" %get-leakyrelu-backend) :string) 
(cffi:defcfun ("get_sigmoidinplace_backend" %get-sigmoidinplace-backend) :string)  
(cffi:defcfun ("get_sigmoid_backend" %get-sigmoid-backend) :string) 
(cffi:defcfun ("get_tanhinplace_backend" %get-tanhinplace-backend) :string)  
(cffi:defcfun ("get_tanh_backend" %get-tanh-backend) :string) 
(cffi:defcfun ("get_concat_backend" %get-concat-backend) :string) 
(cffi:defcfun ("get_randn_backend" %get-randn-backend) :string) 
(cffi:defcfun ("get_xavier_backend" %get-xavier-backend) :string) 
(cffi:defcfun ("get_transposeinplace_backend" %get-transposeinplace-backend) :string) 
(cffi:defcfun ("get_transpose_backend" %get-transpose-backend) :string) 
(cffi:defcfun ("get_sum_backend" %get-sum-backend) :string) 
(cffi:defcfun ("get_l2norm_backend" %get-l2norm-backend) :string) 
(cffi:defcfun ("get_copy_backend" %get-copy-backend) :string) 
(cffi:defcfun ("get_axpy_inplace_backend" %get-axpy-inplace-backend) :string) 
(cffi:defcfun ("get_axpy_backend" %get-axpy-backend) :string) 

(cffi:defcfun ("get_nnl2_view_num_backends" %get-view-num-backends) :int)
(cffi:defcfun ("get_nnl2_view_backends" %get-view-backends) :pointer)
(cffi:defcfun ("get_nnl2_tref_getter_num_backends" %get-tref-getter-num-backends) :int)
(cffi:defcfun ("get_nnl2_tref_getter_backends" %get-tref-getter-backends) :pointer)
(cffi:defcfun ("get_nnl2_empty_num_backends" %get-empty-num-backends) :int)
(cffi:defcfun ("get_nnl2_empty_backends" %get-empty-backends) :pointer)
(cffi:defcfun ("get_nnl2_zeros_num_backends" %get-zeros-num-backends) :int)
(cffi:defcfun ("get_nnl2_zeros_backends" %get-zeros-backends) :pointer)
(cffi:defcfun ("get_inplace_fill_num_backends" %get-inplace-fill-num-backends) :int)
(cffi:defcfun ("get_inplace_fill_backends" %get-inplace-fill-backends) :pointer)
(cffi:defcfun ("get_dgemminplace_num_backends" %get-gemm-num-backends) :int)
(cffi:defcfun ("get_dgemminplace_backends" %get-gemm-backends) :pointer)
(cffi:defcfun ("get_addinplace_num_backends" %get-addinplace-num-backends) :int)
(cffi:defcfun ("get_addinplace_backends" %get-addinplace-backends) :pointer)
(cffi:defcfun ("get_subinplace_num_backends" %get-subinplace-num-backends) :int)
(cffi:defcfun ("get_subinplace_backends" %get-subinplace-backends) :pointer)
(cffi:defcfun ("get_add_num_backends" %get-add-num-backends) :int)
(cffi:defcfun ("get_add_backends" %get-add-backends) :pointer)
(cffi:defcfun ("get_sub_num_backends" %get-sub-num-backends) :int)
(cffi:defcfun ("get_sub_backends" %get-sub-backends) :pointer)
(cffi:defcfun ("get_mulinplace_num_backends" %get-mulinplace-num-backends) :int)
(cffi:defcfun ("get_mulinplace_backends" %get-mulinplace-backends) :pointer)
(cffi:defcfun ("get_divinplace_num_backends" %get-divinplace-num-backends) :int)
(cffi:defcfun ("get_divinplace_backends" %get-divinplace-backends) :pointer)
(cffi:defcfun ("get_mul_num_backends" %get-mul-num-backends) :int)
(cffi:defcfun ("get_mul_backends" %get-mul-backends) :pointer)
(cffi:defcfun ("get_div_num_backends" %get-div-num-backends) :int)
(cffi:defcfun ("get_div_backends" %get-div-backends) :pointer)
(cffi:defcfun ("get_powinplace_num_backends" %get-powinplace-num-backends) :int)
(cffi:defcfun ("get_powinplace_backends" %get-powinplace-backends) :pointer)
(cffi:defcfun ("get_pow_num_backends" %get-pow-num-backends) :int)
(cffi:defcfun ("get_pow_backends" %get-pow-backends) :pointer)
(cffi:defcfun ("get_expinplace_num_backends" %get-expinplace-num-backends) :int)
(cffi:defcfun ("get_expinplace_backends" %get-expinplace-backends) :pointer)
(cffi:defcfun ("get_exp_num_backends" %get-exp-num-backends) :int)
(cffi:defcfun ("get_exp_backends" %get-exp-backends) :pointer)
(cffi:defcfun ("get_loginplace_num_backends" %get-loginplace-num-backends) :int)
(cffi:defcfun ("get_loginplace_backends" %get-loginplace-backends) :pointer)
(cffi:defcfun ("get_log_num_backends" %get-log-num-backends) :int)
(cffi:defcfun ("get_log_backends" %get-log-backends) :pointer)
(cffi:defcfun ("get_scaleinplace_num_backends" %get-scaleinplace-num-backends) :int)
(cffi:defcfun ("get_scaleinplace_backends" %get-scaleinplace-backends) :pointer)
(cffi:defcfun ("get_scale_num_backends" %get-scale-num-backends) :int)
(cffi:defcfun ("get_scale_backends" %get-scale-backends) :pointer)
(cffi:defcfun ("get_maxinplace_num_backends" %get-maxinplace-num-backends) :int)
(cffi:defcfun ("get_maxinplace_backends" %get-maxinplace-backends) :pointer)
(cffi:defcfun ("get_mininplace_num_backends" %get-mininplace-num-backends) :int)
(cffi:defcfun ("get_mininplace_backends" %get-mininplace-backends) :pointer)
(cffi:defcfun ("get_max_num_backends" %get-max-num-backends) :int)
(cffi:defcfun ("get_max_backends" %get-max-backends) :pointer)
(cffi:defcfun ("get_min_num_backends" %get-min-num-backends) :int)
(cffi:defcfun ("get_min_backends" %get-min-backends) :pointer)
(cffi:defcfun ("get_absinplace_num_backends" %get-absinplace-num-backends) :int)
(cffi:defcfun ("get_absinplace_backends" %get-absinplace-backends) :pointer)
(cffi:defcfun ("get_abs_num_backends" %get-abs-num-backends) :int)
(cffi:defcfun ("get_abs_backends" %get-abs-backends) :pointer)
(cffi:defcfun ("get_hstack_num_backends" %get-hstack-num-backends) :int)
(cffi:defcfun ("get_hstack_backends" %get-hstack-backends) :pointer)
(cffi:defcfun ("get_vstack_num_backends" %get-vstack-num-backends) :int)
(cffi:defcfun ("get_vstack_backends" %get-vstack-backends) :pointer)
(cffi:defcfun ("get_reluinplace_num_backends" %get-reluinplace-num-backends) :int)
(cffi:defcfun ("get_reluinplace_backends" %get-reluinplace-backends) :pointer)
(cffi:defcfun ("get_relu_num_backends" %get-relu-num-backends) :int)
(cffi:defcfun ("get_relu_backends" %get-relu-backends) :pointer)
(cffi:defcfun ("get_leakyreluinplace_num_backends" %get-leakyreluinplace-num-backends) :int)
(cffi:defcfun ("get_leakyreluinplace_backends" %get-leakyreluinplace-backends) :pointer)
(cffi:defcfun ("get_leakyrelu_num_backends" %get-leakyrelu-num-backends) :int)
(cffi:defcfun ("get_leakyrelu_backends" %get-leakyrelu-backends) :pointer)
(cffi:defcfun ("get_sigmoidinplace_num_backends" %get-sigmoidinplace-num-backends) :int)
(cffi:defcfun ("get_sigmoidinplace_backends" %get-sigmoidinplace-backends) :pointer)
(cffi:defcfun ("get_sigmoid_num_backends" %get-sigmoid-num-backends) :int)
(cffi:defcfun ("get_sigmoid_backends" %get-sigmoid-backends) :pointer)
(cffi:defcfun ("get_tanhinplace_num_backends" %get-tanhinplace-num-backends) :int)
(cffi:defcfun ("get_tanhinplace_backends" %get-tanhinplace-backends) :pointer)
(cffi:defcfun ("get_tanh_num_backends" %get-tanh-num-backends) :int)
(cffi:defcfun ("get_tanh_backends" %get-tanh-backends) :pointer)
(cffi:defcfun ("get_concat_num_backends" %get-concat-num-backends) :int)
(cffi:defcfun ("get_concat_backends" %get-concat-backends) :pointer)
(cffi:defcfun ("get_randn_num_backends" %get-randn-num-backends) :int)
(cffi:defcfun ("get_randn_backends" %get-randn-backends) :pointer)
(cffi:defcfun ("get_xavier_num_backends" %get-xavier-num-backends) :int)
(cffi:defcfun ("get_xavier_backends" %get-xavier-backends) :pointer)
(cffi:defcfun ("get_transposeinplace_num_backends" %get-transposeinplace-num-backends) :int)
(cffi:defcfun ("get_transposeinplace_backends" %get-transposeinplace-backends) :pointer)
(cffi:defcfun ("get_transpose_num_backends" %get-transpose-num-backends) :int)
(cffi:defcfun ("get_transpose_backends" %get-transpose-backends) :pointer)
(cffi:defcfun ("get_sum_num_backends" %get-sum-num-backends) :int)
(cffi:defcfun ("get_sum_backends" %get-sum-backends) :pointer)
(cffi:defcfun ("get_l2norm_num_backends" %get-l2norm-num-backends) :int)
(cffi:defcfun ("get_l2norm_backends" %get-l2norm-backends) :pointer)
(cffi:defcfun ("get_copy_num_backends" %get-copy-num-backends) :int)
(cffi:defcfun ("get_copy_backends" %get-copy-backends) :pointer)
(cffi:defcfun ("get_axpy_inplace_num_backends" %get-axpy-inplace-num-backends) :int)
(cffi:defcfun ("get_axpy_inplace_backends" %get-axpy-inplace-backends) :pointer)
(cffi:defcfun ("get_axpy_num_backends" %get-axpy-num-backends) :int)
(cffi:defcfun ("get_axpy_backends" %get-axpy-backends) :pointer)

(nnl-init-system)
