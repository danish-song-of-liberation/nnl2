(in-package :nnl2.ffi)

;; NNL2

;; Filepath: nnl2/src/lisp/ffi/ffi-c-core.lisp
;; File: ffi-c-core.lisp

;; The file contains the import of almost all libnnl.dll 
;; functions into the lisp environment

;; Note:
;;	 I only left the docstring in functions that will be 
;;	 called regularly. I won't/will leave little documentation 
;;	 in copy-paste functions that do things that are already 
;;	 clear, leaving normal documentation only in functions that 
;;	 may be called regularly in a high-level interface

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(cffi:defcfun ("nnl2_init_system" nnl-init-system) :void)

;; -- Main structures --

(cffi:defcenum tensor-type
  "nnl2 Type system import into cffi"
  :int32    ;; aka 'integer (lisp), int32_t (c), int (c), :int (cffi)
  :float32  ;; aka 'single-float (lisp), float (c), :float (cffi)
  :float64) ;; aka 'double-float (lisp), double (c), :double (cffi)"
  
(cffi:defcenum nnl2-order
  "Enum for storage order (made for BLAS)"
  (:nnl2colmajor 101)  ;; Column-major order
  (:nnl2rowmajor 102)) ;; Row-major order
   
(cffi:defcenum nnl2-transpose
  "nnl2 BLAS-Like enum for operations that 
   may require transposition (like GEMM)"
  (:nnl2notrans 111) 
  (:nnl2trans 112))  
  
(cffi:defcenum nnl2-obj-type
  "Determines the tensor type (ts/ad)"
  :nnl2-type-ts 
  :nnl2-type-ad
  :nnl2-type-unknown)
   
(cffi:defcstruct tensor
  "Tensor structure representing a multi-dimensional array"
  (tensor-type tensor-type)  ;; Data type of tensor elements
  (data :pointer)			 ;; Pointer to the raw tensor data
  (shape :pointer)			 ;; Array of dimension sizes 
  (strides :pointer)  		 ;; Array of byte strides for each dimension
  (rank :int)				 ;; Number of dimensions (ndim)	
  (is-view :bool)			 ;; Flag indicating if this is a view (not owning data)
  (magic-number :char)       ;; This is necessary to avoid memory corruption when releasing the tensor
  (ts-obj nnl2-obj-type))	 ;; To separate TS tensors from AD tensors
  
;; -- Main operations --  
  
(cffi:defcfun ("lisp_call_empty" %empty) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))
  
(cffi:defcfun ("nnl2_zeros" %zeros) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))   
  
(cffi:defcfun ("nnl2_ones" %ones) :pointer
  (shape (:pointer :int32))
  (rank :int)
  (dtype tensor-type))  
    
(cffi:defcfun ("nnl2_full" %full) :pointer
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
  (multiplier :float)
  (save-type :bool))
  
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
  
(cffi:defcfun ("lisp_call_add" %+) :pointer
  (summand :pointer)
  (addend :pointer))  
  
(cffi:defcfun ("lisp_call_add_incf" %.+/incf) :pointer
  (tensor :pointer)
  (increment :pointer))  
  
(cffi:defcfun ("lisp_call_sub" %-) :pointer
  (summand :pointer)
  (addend :pointer))    
 
(cffi:defcfun ("lisp_call_sub_decf" %.-/decf) :pointer
  (tensor :pointer)
  (increment :pointer))  
  
(cffi:defcfun ("lisp_call_mulinplace" %*=) :void
  (multiplicand :pointer)
  (multiplier :pointer))  
  
(cffi:defcfun ("lisp_call_mul_mulf_inplace" %.*/mulf!) :void
  (tensor :pointer)
  (multiplier :pointer))  
  
(cffi:defcfun ("lisp_call_divinplace" %/=) :void
  (dividend :pointer)
  (divisor :pointer))  
    
(cffi:defcfun ("lisp_call_div_divf_inplace" %.//divf!) :void
  (tensor :pointer)
  (dif :pointer))
  
(cffi:defcfun ("lisp_call_mul" %*) :pointer
  (multiplicand :pointer)
  (multiplier :pointer))
  
(cffi:defcfun ("lisp_call_mul_mulf" %.*/mulf) :pointer
  (tensor :pointer)
  (multiplier :pointer)) 

(cffi:defcfun ("lisp_call_div" %/) :pointer
  (dividend :pointer)
  (divisor :pointer))
  
(cffi:defcfun ("lisp_call_div_divf" %.//divf) :pointer
  (tensor :pointer)
  (dif :pointer))  
  
(cffi:defcfun ("lisp_call_powinplace" %^=) :void
  (base :pointer)
  (exponent :pointer))  
  
(cffi:defcfun ("lisp_call_pow_powf_inplace" %.^/powf!) :void
  (tensor :pointer)
  (powf :pointer))  
  
(cffi:defcfun ("lisp_call_expinplace" %.exp!) :void
  (tensor :pointer))
  
(cffi:defcfun ("lisp_call_pow" %.^) :pointer
  (base :pointer)
  (exponent :pointer))    
  
(cffi:defcfun ("lisp_call_pow_powf" %.^/powf) :pointer
  (tensor :pointer)
  (powf :pointer)) 
  
(cffi:defcfun ("lisp_call_exp" %.exp) :pointer
  (tensor :pointer)
  (save-type :bool))

(cffi:defcfun ("lisp_call_loginplace" %.log!) :void
  (tensor :pointer))  

(cffi:defcfun ("lisp_call_log" %.log) :pointer
  (tensor :pointer)
  (save-type :bool))
  
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
  
(cffi:defcfun ("nnl2_full_like" %full-like) :pointer
  (tensor :pointer)
  (filler :pointer))
  
(cffi:defcfun ("nnl2_randn_like" %randn-like) :pointer
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
  
(cffi:defcfun ("lisp_call_xavier_inplace" %xavier-inplace) :void
  (tensor :pointer)
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

(cffi:defcfun ("lisp_call_randn_inplace" %randn-inplace) :pointer
  (tensor :pointer)
  (from :pointer)
  (to :pointer))  
  
(cffi:defcfun ("lisp_call_maxinplace" %.max!) :void
  (tensora :pointer)
  (tensorb :pointer))    
  
(cffi:defcfun ("lisp_call_max_maxf_inplace" %.max/maxf!) :void
  (tensor :pointer)
  (maxf :pointer))    
  
(cffi:defcfun ("lisp_call_mininplace" %.min!) :void
  (tensora :pointer)
  (tensorb :pointer))  
  
(cffi:defcfun ("lisp_call_min_minf_inplace" %.min/minf!) :void
  (tensor :pointer)
  (minf :pointer))   

(cffi:defcfun ("lisp_call_max" %.max) :pointer
  (tensora :pointer)
  (tensorb :pointer))   

(cffi:defcfun ("lisp_call_max_maxf" %.max/maxf) :pointer
  (tensor :pointer)
  (maxf :pointer))  
  
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
  
(cffi:defcfun ("lisp_call_sum_without_axis" %sum-without-axis) :void
  (tensor :pointer)
  (out :pointer))  
  
(cffi:defcfun ("lisp_call_sum_with_axis" %sum-with-axis) tensor
  (tensor :pointer)
  (axis :int)
  (keepdim :bool))    
  
(cffi:defcfun ("lisp_call_l2norm" %l2norm) :void
  (tensor :pointer)
  (axes :pointer)
  (num-axes :int)
  (out :pointer))   

(cffi:defcfun ("lisp_call_copy" %copy) :pointer
  (tensor :pointer)
  (copy-type tensor-type))  
  
(cffi:defcfun ("nnl2_free_tensor" free-tensor) :void
  (tensor :pointer))     
  
(cffi:defcfun ("nnl2_print_tensor" print-tensor) :void
  (tensor :pointer)
  (full-print :bool)
  (max-rows :int)
  (max-cols :int)
  (show-rows :int)
  (show-cols :int))

(cffi:defcfun ("nnl2_get_tensor_shape" get-pointer-to-tensor-shape) :pointer
  (tensor :pointer))        
  
(cffi:defcfun ("nnl2_get_tensor_data" get-tensor-data) :pointer
  (tensor :pointer))  
  
(cffi:defcfun ("nnl2_get_tensor_strides" get-pointer-to-tensor-strides) :pointer
  (tensor :pointer))    
  
(cffi:defcfun ("nnl2_shape_at" shape-at) :int
  (tensor :pointer)
  (index :int))
  
(cffi:defcfun ("get_nnl2_object_type" get-obj-type) :int ;; faster that nnl2-obj-type
  (tensor :pointer))  
  
(cffi:defcfun ("lisp_call_debug_blas_sgemminplace" %%internal-debug-sgemm!) :void
  (check-to :unsigned-long))  

(cffi:defcfun ("get_mem_alignment" %get-mem-alignment) :int)
	
(cffi:defcfun ("make_tensor_from_flatten" %make-tensor-from-flatten) :pointer
  (data :pointer)
  (num-elems-data :int)
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))
  
(cffi:defcfun ("lisp_call_reshape" %reshape) :pointer
  (tensor :pointer)
  (new-shape :pointer)
  (new-shape-len :int)
  (force :bool))
  
(cffi:defcfun ("lisp_call_reinterpret" %reinterpret) :pointer
  (tensor :pointer)
  (new-shape :pointer)
  (new-shape-len :int)
  (force :bool))  
  
(cffi:defcfun ("nnl2_get_raw_tensor_elem_at" %lowlevel-tref) :pointer
  (tensor :pointer)
  (at :size))
  
(cffi:defcfun ("nnl2_set_raw_tensor_elem_at" %lowlevel-tref-setter) :void
  (tensor :pointer)
  (at :size)
  (with :pointer))
  
(cffi:defcfun ("nnl2_get_raw_tensor_elem" %lowlevel-tref-with-coords) :pointer
  (tensor :pointer)
  (coords :pointer)
  (coords-len :int))
  
(cffi:defcfun ("nnl2_set_raw_tensor_elem" %lowlevel-tref-with-coords-setter) :void
  (tensor :pointer)
  (coords :pointer)
  (coords-len :int)
  (with :pointer))  
  
(cffi:defcfun ("lisp_call_slice" %slice) :pointer
  (tensor :pointer)
  (slice-from :pointer)
  (slice-to :pointer))  
  
(cffi:defcfun ("lisp_call_cut" %cut) :pointer
  (tensor :pointer)
  (slice-from :pointer)
  (slice-to :pointer)) 

(cffi:defcfun ("lisp_call_transpose" %transpose) :pointer
  (tensor :pointer)
  (force :bool))  
  
(cffi:defcfun ("lisp_call_transposeinplace" %transpose!) :void
  (tensor :pointer)
  (force :bool))  
  
(cffi:defcfun ("lisp_call_inplace_fill" %fill!) :bool
  (tensor :pointer)
  (value :pointer)
  (dtype tensor-type))  
  
;; -- AD --

(cffi:defcstruct ad-tensor
  "Tensor structure with support for automatic differentiation"
  (ad-obj nnl2-obj-type)
  (data tensor)
  (grad tensor)
  (requires-grad :bool)
  (is-leaf :bool)
  (backward-fn :pointer)
  (roots :pointer)
  (num-roots :long)
  (visited-gen :unsigned-long)
  (name :string)
  (magic-number :char)
  (grad-initialized :bool)
  (extra-multiplier :float)
  (extra-bool :bool)
  (extra-correspondence :pointer))  
  
(cffi:defcenum ad-mode 
  ad-reverse-mode
  ad-p1-mode
  ad-p2-mode
  ad-p3-mode)  
  
(cffi:defcfun ("nnl2_ad_get_roots" %ad-roots) :pointer
  (ad-tensor :pointer))
  
(cffi:defcfun ("nnl2_ad_roots_setter" %ad-roots-setter) :void
  (ad-tensor :pointer)
  (new-roots :pointer)
  (new-num-roots :int))  
  
(cffi:defcfun ("nnl2_ad_backpropagation" %backpropagation) :void
  (ad-tensor :pointer)
  (retain-graph :bool))
  
(cffi:defcfun ("nnl2_ad_backpropagation_through_time" %bptt) :void
  (ad-tensor :pointer)
  (retain-graph :bool))      

(cffi:defcfun ("nnl2_ad_empty" %ad-empty) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string))

(cffi:defcfun ("nnl2_ad_zeros" %ad-zeros) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string))
  
(cffi:defcfun ("nnl2_ad_ones" %ad-ones) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string))  
  
(cffi:defcfun ("nnl2_ad_full" %ad-full) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string)
  (filler :pointer))
 
(cffi:defcfun ("nnl2_ad_randn" %ad-randn) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string)
  (from :pointer)
  (to :pointer))
  
(cffi:defcfun ("nnl2_ad_xavier" %ad-xavier) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string)
  (in :int)
  (out :int)
  (gain :float)
  (distribution :float))
  
(cffi:defcfun ("nnl2_ad_add" %ad-.+) :pointer
  (summand :pointer)
  (addend :pointer)
  (mode ad-mode)
  (track-grad :bool)) 
  
(cffi:defcfun ("nnl2_ad_add_inplace" %ad-+=) :void
  (summand :pointer)
  (addend :pointer)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_sub" %ad-.-) :pointer
  (minuend :pointer)
  (subtrahend :pointer)
  (mode ad-mode)
  (track-grad :bool)) 
  
(cffi:defcfun ("nnl2_ad_sub_inplace" %ad--=) :void
  (minuend :pointer)
  (subtrahend :pointer)
  (track-graph :bool))  
  
(cffi:defcfun ("nnl2_ad_mul" %ad-.*) :pointer
  (multiplicand :pointer)
  (multiplier :pointer)
  (mode ad-mode)
  (track-grad :bool))   
  
(cffi:defcfun ("nnl2_ad_mul_inplace" %ad-*=) :void
  (multiplicand :pointer)
  (multiplier :pointer)
  (track-graph :bool))  
  
(cffi:defcfun ("nnl2_ad_div" %ad-./) :pointer
  (dividend :pointer)
  (divisor :pointer)
  (mode ad-mode)
  (track-grad :bool))   
  
(cffi:defcfun ("nnl2_ad_div_inplace" %ad-/!) :pointer
  (dividend :pointer)
  (divisor :pointer)
  (track-graph :bool))   
  
(cffi:defcfun ("nnl2_ad_pow" %ad-.^) :pointer
  (base :pointer)
  (exponent :pointer)
  (mode ad-mode)
  (track-grad :bool))
  
(cffi:defcfun ("nnl2_ad_pow_inplace" %ad-^=) :void
  (base :pointer)
  (exponent :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_log" %ad-.log) :pointer
  (ad-tensor :pointer)
  (save-type :bool)
  (mode ad-mode)
  (track-grad :bool))    
  
(cffi:defcfun ("nnl2_ad_inplace_log" %ad-.log!) :void
  (ad-tensor :pointer)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_scale" %ad-scale) :pointer
  (ad-tensor :pointer)
  (multiplier :float)
  (save-type :bool)
  (mode ad-mode)
  (track-grad :bool)) 

(cffi:defcfun ("nnl2_ad_inplace_scale" %ad-scale!) :void
  (ad-tensor :pointer)
  (multiplier :float)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_min" %ad-.min) :pointer
  (a :pointer)
  (b :pointer)
  (mode ad-mode)
  (track-grad :bool))
  
(cffi:defcfun ("nnl2_ad_min_inplace" %ad-.min!) :void
  (a :pointer)
  (b :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_max" %ad-.max) :pointer
  (a :pointer)
  (b :pointer)
  (mode ad-mode)
  (track-grad :bool))    

(cffi:defcfun ("nnl2_ad_max_inplace" %ad-.max!) :void
  (a :pointer)
  (b :pointer)
  (track-graph :bool))    
  
(cffi:defcfun ("nnl2_ad_exp" %ad-.exp) :pointer
  (ad_tensor :pointer)
  (save-type :bool)
  (mode ad-mode)
  (track-grad :bool)) 

(cffi:defcfun ("nnl2_ad_inplace_exp" %ad-.exp!) :void
  (ad_tensor :pointer)
  (track-graph :bool))   
  
(cffi:defcfun ("nnl2_ad_axpy" %ad-axpy) :pointer
  (summand :pointer)
  (addend :pointer)
  (multiplier :float)
  (mode ad-mode)
  (track-grad :bool))    

(cffi:defcfun ("nnl2_ad_inplace_axpy" %ad-axpy!) :void
  (summand :pointer)
  (addend :pointer)
  (multiplier :float)
  (track-graph :bool)) 
  
(cffi:defcfun ("nnl2_ad_abs" %ad-.abs) :pointer
  (ad-pointer :pointer)
  (mode ad-mode)
  (track-grad :bool))    
  
(cffi:defcfun ("nnl2_ad_inplace_abs" %ad-.abs!) :void
  (ad-pointer :pointer)
  (track-graph :bool))   

(cffi:defcfun ("nnl2_ad_add_broadcasting" %ad-.+/broadcasting) :pointer
  (summand :pointer)
  (addend :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_gemm" %ad-gemm) :pointer
  (multiplicand :pointer)
  (multiplier :pointer)
  (mode ad-mode)
  (track-grad :bool))   

(cffi:defcfun ("nnl2_ad_neg" %.neg) :pointer
  (ad-tensor :pointer)
  (mode ad-mode)
  (track-grad :bool))	  
  
(cffi:defcfun ("nnl2_ad_inplace_leakyrelu" %ad-.leaky-relu!) :void
  (ad-tensor :pointer)
  (alpha :float)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_leakyrelu" %ad-.leaky-relu) :pointer
  (ad-tensor :pointer)
  (alpha :float)
  (save-type :bool)
  (mode ad-mode)
  (track-grad :bool)) 

(cffi:defcfun ("nnl2_ad_inplace_relu" %ad-.relu!) :void
  (ad-tensor :pointer)
  (track-graph :bool))  

(cffi:defcfun ("nnl2_ad_relu" %ad-.relu) :pointer
  (ad-tensor :pointer)
  (mode ad-mode)
  (track-grad :bool))  

(cffi:defcfun ("nnl2_ad_inplace_sigmoid" %ad-.sigmoid!) :void
  (ad-tensor :pointer)
  (approx :bool)
  (track-graph :bool))
 
(cffi:defcfun ("nnl2_ad_sigmoid" %ad-.sigmoid) :pointer
  (ad-tensor :pointer)
  (approx :bool)
  (mode ad-mode)
  (track-grad :bool))     
 
(cffi:defcfun ("nnl2_ad_inplace_tanh" %ad-.tanh!) :void
  (ad-tensor :pointer)
  (approx :bool)
  (track-graph :bool)) 

(cffi:defcfun ("nnl2_ad_add_broadcasting" %ad-add-broadcasting) :pointer
  (addend :pointer)
  (sumend :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_axpy_broadcasting" %ad-axpy-broadcasting) :pointer
  (axpyend :pointer)
  (sumend :pointer)
  (multiplier :float)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_div_broadcasting" %ad-div-broadcasting) :pointer
  (dividend :pointer)
  (divisor :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_max_broadcasting" %ad-max-broadcasting) :pointer
  (tensor-a :pointer)
  (tensor-b :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_min_broadcasting" %ad-min-broadcasting) :pointer
  (tensor-a :pointer)
  (tensor-b :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_mul_broadcasting" %ad-mul-broadcasting) :pointer
  (multiplier :pointer)
  (multiplicand :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_pow_broadcasting" %ad-pow-broadcasting) :pointer
  (base :pointer)
  (exponent :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_sub_broadcasting" %ad-sub-broadcasting) :pointer
  (minuend :pointer)
  (subtrahend :pointer)
  (mode ad-mode)
  (track-grad :bool))
  
(cffi:defcfun ("nnl2_ad_add_broadcasting_inplace" %ad-add-broadcasting-inplace) :void
  (summand :pointer)
  (addend :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_axpy_broadcasting_inplace" %ad-axpy-broadcasting-inplace) :void
  (sumend :pointer)
  (axpyend :pointer)
  (multiplier :float)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_div_broadcasting_inplace" %ad-div-broadcasting-inplace) :void
  (dividend :pointer)
  (divisor :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_max_broadcasting_inplace" %ad-max-broadcasting-inplace) :void
  (tensor-a :pointer)
  (tensor-b :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_min_broadcasting_inplace" %ad-min-broadcasting-inplace) :void
  (tensor-a :pointer)
  (tensor-b :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_mul_broadcasting_inplace" %ad-mul-broadcasting-inplace) :void
  (multiplicand :pointer)
  (multiplier :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_pow_broadcasting_inplace" %ad-pow-broadcasting-inplace) :void
  (base :pointer)
  (exponent :pointer)
  (track-graph :bool))
			  
(cffi:defcfun ("nnl2_ad_sub_broadcasting_inplace" %ad-sub-broadcasting-inplace) :void
  (minuend :pointer)
  (subtrahend :pointer)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_tanh" %ad-.tanh) :pointer
  (ad-tensor :pointer)
  (approx :bool)
  (mode ad-mode)
  (track-grad :bool))  
  
(cffi:defcfun ("nnl2_ad_add_correspondence" %ad-add-correspondence) :pointer
  (tensor :pointer)
  (inc :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_axpf" %ad-axpf) :pointer
  (summand :pointer)
  (sumend :pointer)
  (alpha :float)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_div_correspondence" %ad-div-correspondence) :pointer
  (tensor :pointer)
  (divisor :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_max_correspondence" %ad-max-correspondence) :pointer
  (tensor :pointer)
  (threshold :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_min_correspondence" %ad-min-correspondence) :pointer
  (tensor :pointer)
  (threshold :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_mul_correspondence" %ad-mul-correspondence) :pointer
  (tensor :pointer)
  (multiplier :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_pow_correspondence" %ad-pow-correspondence) :pointer
  (tensor :pointer)
  (exponent :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_sub_correspondence" %ad-sub-correspondence) :pointer
  (tensor :pointer)
  (dec :pointer)
  (mode ad-mode)
  (track-grad :bool))

(cffi:defcfun ("nnl2_ad_add_incf_inplace" %ad-add-incf-inplace) :void
  (tensor :pointer)
  (inc :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_axpf_inplace" %ad-axpf-inplace) :void
  (summand :pointer)
  (sumend :pointer)
  (alpha :float)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_div_divf_inplace" %ad-div-divf-inplace) :void
  (tensor :pointer)
  (divisor :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_max_maxf_inplace" %ad-max-maxf-inplace) :void
  (tensor :pointer)
  (threshold :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_min_minf_inplace" %ad-min-minf-inplace) :void
  (tensor :pointer)
  (threshold :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_mul_mulf_inplace" %ad-mul-mulf-inplace) :void
  (tensor :pointer)
  (multiplier :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_pow_powf_inplace" %ad-pow-powf-inplace) :void
  (tensor :pointer)
  (exponent :pointer)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_sub_decf_inplace" %ad-sub-decf-inplace) :void
  (tensor :pointer)
  (dec :pointer)
  (track-graph :bool))  
  
(cffi:defcfun ("nnl2_ad_step" %ad-step) :pointer
  (ad-tensor :pointer)
  (learning-rate :float))
  
(cffi:defcfun ("nnl2_ad_step_inplace" %ad-step!) :void
  (ad-tensor :pointer)
  (learning-rate :float))  
  
(cffi:defcfun ("nnl2_ad_inplace_transposition" %ad-transposition-inplace) :void
  (ad-tensor :pointer)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_inplace_transpose" %ad-transpose-inplace) :void
  (ad-tensor :pointer)
  (track-graph :bool)
  (force :bool))
  
(cffi:defcfun ("nnl2_ad_transpose" %ad-transpose) :pointer
  (ad-tensor :pointer)
  (mode ad-mode)
  (track-graph :bool)
  (force :bool))
  
(cffi:defcfun ("nnl2_ad_transposition" %ad-transposition) :pointer
  (ad-tensor :pointer)
  (mode ad-mode)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_reshape" %ad-reshape) :pointer
  (tensor :pointer)
  (new-shape :pointer)
  (new-shape-len :int)
  (force :bool)
  (ad-mode ad-mode)
  (track-graph :bool))
   
(cffi:defcfun ("nnl2_ad_reinterpret" %ad-reinterpret) :pointer
  (tensor :pointer)
  (new-shape :pointer)
  (new-shape-len :int)
  (force :bool)
  (ad-mode ad-mode)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_get_shape" %ad-shape) :pointer
  (ad-tensor :pointer))  
  
(cffi:defcfun ("nnl2_ad_get_rank" %ad-rank) :int
  (ad-tensor :pointer))    
  
(cffi:defcfun ("nnl2_ad_get_dtype_as_data" %ad-dtype-as-data) tensor-type
  (ad-pointer :pointer))
  
(cffi:defcfun ("nnl2_ad_get_dtype_as_data" %ad-dtype-as-data-int) :int 
  (ad-pointer :pointer))  

(cffi:defcfun ("nnl2_ad_get_dtype_as_grad" %ad-dtype-as-grad) tensor-type
  (ad-pointer :pointer))  

(cffi:defcfun ("nnl2_ad_get_dtype_as_grad" %ad-dtype-as-grad-int) :int 
  (ad-pointer :pointer))    
  
;; -- Backends --  
 
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
  
(cffi:defcfun ("set_i32gemminplace_backend" %set-i32gemminplace-backend) :void
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

(cffi:defcfun ("set_randn_inplace_backend" %set-randn-inplace-backend) :void
  (backend-name :string))   
  	
(cffi:defcfun ("set_xavier_backend" %set-xavier-backend) :void
  (backend-name :string))  
 
(cffi:defcfun ("set_xavier_inplace_backend" %set-xavier-inplace-backend) :void
  (backend-name :string)) 
	
(cffi:defcfun ("set_transposeinplace_backend" %set-transposeinplace-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_transpose_backend" %set-transpose-backend) :void
  (backend-name :string))  
  	
(cffi:defcfun ("set_sum_without_axis_backend" %set-sum-without-axis-backend) :void
  (backend-name :string))  
  
(cffi:defcfun ("set_sum_with_axis_backend" %set-sum-with-axis-backend) :void
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
  
(cffi:defcfun ("set_reshape_backend" %set-reshape-backend) :void
  (backend-name :string))
  
(cffi:defcfun ("set_reinterpret_backend" %set-reinterpret-backend) :void
  (backend-name :string))
  
(cffi:defcfun ("set_slice_backend" %set-slice-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_cut_backend" %set-cut-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_transposition_backend" %set-transposition-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_transposition_inplace_backend" %set-transposition-inplace-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_neginplace_backend" %set-neginplace-backend) :void
  (backend-name :string))

(cffi:defcfun ("set_neg_backend" %set-neg-backend) :void
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
(cffi:defcfun ("get_randn_inplace_backend" %get-randn-inplace-backend) :string) 
(cffi:defcfun ("get_xavier_backend" %get-xavier-backend) :string) 
(cffi:defcfun ("get_xavier_inplace_backend" %get-xavier-inplace-backend) :string) 
(cffi:defcfun ("get_transposeinplace_backend" %get-transposeinplace-backend) :string) 
(cffi:defcfun ("get_transpose_backend" %get-transpose-backend) :string) 
(cffi:defcfun ("get_sum_without_axis_backend" %get-sum-without-axis-backend) :string) 
(cffi:defcfun ("get_l2norm_backend" %get-l2norm-backend) :string) 
(cffi:defcfun ("get_copy_backend" %get-copy-backend) :string) 
(cffi:defcfun ("get_axpy_inplace_backend" %get-axpy-inplace-backend) :string) 
(cffi:defcfun ("get_axpy_backend" %get-axpy-backend) :string) 
(cffi:defcfun ("get_reshape_backend" %get-reshape-backend) :string) 
(cffi:defcfun ("get_reinterpret_backend" %get-reinterpret-backend) :string) 
(cffi:defcfun ("get_slice_backend" %get-slice-backend) :string)
(cffi:defcfun ("get_cut_backend" %get-cut-backend) :string)
(cffi:defcfun ("get_transposition_backend" %get-transposition-backend) :string)
(cffi:defcfun ("get_transposition_inplace_backend" %get-transposition-inplace-backend) :string)
(cffi:defcfun ("get_neginplace_backend" %get-neginplace-backend) :string)
(cffi:defcfun ("get_neg_backend" %get-neg-backend) :string)

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
(cffi:defcfun ("get_randn_inplace_num_backends" %get-randn-inplace-num-backends) :int)
(cffi:defcfun ("get_randn_inplace_backends" %get-randn-inplace-backends) :pointer)
(cffi:defcfun ("get_xavier_num_backends" %get-xavier-num-backends) :int)
(cffi:defcfun ("get_xavier_backends" %get-xavier-backends) :pointer)
(cffi:defcfun ("get_xavier_inplace_num_backends" %get-xavier-inplace-num-backends) :int)
(cffi:defcfun ("get_xavier_inplace_backends" %get-xavier-inplace-backends) :pointer)
(cffi:defcfun ("get_transposeinplace_num_backends" %get-transposeinplace-num-backends) :int)
(cffi:defcfun ("get_transposeinplace_backends" %get-transposeinplace-backends) :pointer)
(cffi:defcfun ("get_transpose_num_backends" %get-transpose-num-backends) :int)
(cffi:defcfun ("get_transpose_backends" %get-transpose-backends) :pointer)
(cffi:defcfun ("get_sum_without_axis_num_backends" %get-sum-without-axis-num-backends) :int)
(cffi:defcfun ("get_sum_without_axis_backends" %get-sum-without-axis-backends) :pointer)
(cffi:defcfun ("get_l2norm_num_backends" %get-l2norm-num-backends) :int)
(cffi:defcfun ("get_l2norm_backends" %get-l2norm-backends) :pointer)
(cffi:defcfun ("get_copy_num_backends" %get-copy-num-backends) :int)
(cffi:defcfun ("get_copy_backends" %get-copy-backends) :pointer)
(cffi:defcfun ("get_axpy_inplace_num_backends" %get-axpy-inplace-num-backends) :int)
(cffi:defcfun ("get_axpy_inplace_backends" %get-axpy-inplace-backends) :pointer)
(cffi:defcfun ("get_axpy_num_backends" %get-axpy-num-backends) :int)
(cffi:defcfun ("get_axpy_backends" %get-axpy-backends) :pointer)
(cffi:defcfun ("get_reshape_num_backends" %get-reshape-num-backends) :int)
(cffi:defcfun ("get_reshape_backends" %get-reshape-backends) :pointer)
(cffi:defcfun ("get_reinterpret_num_backends" %get-reinterpret-num-backends) :int)
(cffi:defcfun ("get_reinterpret_backends" %get-reinterpret-backends) :pointer)
(cffi:defcfun ("get_slice_num_backends" %get-slice-num-backends) :int)
(cffi:defcfun ("get_slice_backends" %get-slice-backends) :pointer)
(cffi:defcfun ("get_cut_num_backends" %get-cut-num-backends) :int)
(cffi:defcfun ("get_cut_backends" %get-cut-backends) :pointer)
(cffi:defcfun ("get_transposition_num_backends" %get-transposition-num-backends) :int)
(cffi:defcfun ("get_transposition_inplace_num_backends" %get-transposition-inplace-num-backends) :int)
(cffi:defcfun ("get_transposition_backends" %get-transposition-backends) :pointer)
(cffi:defcfun ("get_transposition_inplace_backends" %get-transposition-inplace-backends) :pointer)
(cffi:defcfun ("get_neginplace_backends" %get-neginplace-backends) :pointer)
(cffi:defcfun ("get_neginplace_num_backends" %get-neginplace-num-backends) :int)
(cffi:defcfun ("get_neg_backends" %get-neg-backends) :pointer)
(cffi:defcfun ("get_neg_num_backends" %get-neg-num-backends) :int)

;; -- mem-aref setters/getters --

(cffi:defcfun ("nnl2_fast_float64_set" mem-aref-setter-float64) :void
  (data :pointer)
  (index :int)
  (value :double))

(cffi:defcfun ("nnl2_fast_float64_get" mem-aref-getter-float64) :double
  (data :pointer)
  (index :int))

(cffi:defcfun ("nnl2_fast_float32_set" mem-aref-setter-float32) :void
  (data :pointer)
  (index :int)
  (value :float))

(cffi:defcfun ("nnl2_fast_float32_get" mem-aref-getter-float32) :float
  (data :pointer)
  (index :int))

(cffi:defcfun ("nnl2_fast_int32_set" mem-aref-setter-int32) :void
  (data :pointer)
  (index :int)
  (value :int))

(cffi:defcfun ("nnl2_fast_int32_get" mem-aref-getter-int32) :int
  (data :pointer)
  (index :int))
  
(nnl-init-system) ;; Initializing backends. Calling the nnl2_init_system function from src/c/nnl2_core.c
