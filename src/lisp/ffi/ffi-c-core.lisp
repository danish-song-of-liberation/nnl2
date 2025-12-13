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
  
(cffi:defcfun ("nnl2_uniform_like" %uniform-like) :pointer
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
  
(cffi:defcfun ("lisp_call_kaiming" %kaiming) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (fan-in :int)
  (fan-out :int)
  (gain :float)
  (distribution :float)
  (mode :int))  
  
(cffi:defcfun ("lisp_call_kaiming_inplace" %kaiming-inplace) :void
  (tensor :pointer)
  (fan-in :int)
  (fan-out :int)
  (gain :float)
  (distribution :float)
  (mode :int))
  
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

(cffi:defcfun ("lisp_call_uniform" %uniform) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (from :pointer)
  (to :pointer))  

(cffi:defcfun ("lisp_call_uniform_inplace" %uniform-inplace) :pointer
  (tensor :pointer)
  (from :pointer)
  (to :pointer))  
  
(cffi:defcfun ("lisp_call_rand" %rand) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type))  
  
(cffi:defcfun ("lisp_call_randn" %randn) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (mean :double)
  (std :double))
  
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
  
(cffi:defcfun ("nnl2_strides_at" strides-at) :int
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
  
(cffi:defcfun ("nnl2_xavier_like" %xavier-like) :pointer	
  (tensor :pointer)
  (in :int)
  (out :int)
  (gain :float)
  (dist :float))
  
(cffi:defcfun ("nnl2_kaiming_like" %kaiming-like) :pointer
  (tensor :pointer)
  (fan-in :int)
  (fan-out :int)
  (gain :float)
  (distribution :float)
  (mode :int))  
  
(cffi:defcfun ("lisp_call_axpy_inplace_regional" %ts-regional-axpy-inplace) :void
  (summand :pointer)
  (sumend :pointer)
  (alpha :float)
  (from :pointer)
  (to :pointer))  
  
(cffi:defcfun ("nnl2_narrow" %narrow) :pointer
  (tensor :pointer)
  (dim :unsigned-char)
  (start :unsigned-long)
  (len :unsigned-long))  
  
(cffi:defcfun ("lisp_call_mse" %mse) :void
  (prediction :pointer)
  (target :pointer)
  (record :pointer))
  
(cffi:defcfun ("nnl2_ts_set_magic_number" %ts-set-magic-number) :void
  (tensor :pointer)
  (new-magic :char))
  
(cffi:defcfun ("nnl2_randn_like" %randn-like) :pointer
  (tensor :pointer)
  (mean :double)
  (std :double))
  
;; -- AD --

(cffi:defcstruct ad-tensor
  "Tensor structure with support for automatic differentiation"
  (ad-obj nnl2-obj-type)  		    	;; To separate AD tensors from TS tensors
  (data tensor)							;; Data of the AD tensor
  (grad tensor)							;; Gradient of the AD tensor
  (requires-grad :bool)					;; A flag that determines whether to count the gradient or not
  (is-leaf :bool)						;; Is the AD tensor the main one or not
  (backward-fn :pointer)				;; AD-tensor function for backpropagation
  (roots :pointer)						;; The roots of a tensor
  (num-roots :long)						;; Number of roots
  (visited-gen :unsigned-long)			;; Used for topological sort (generation-based marking)
  (name :string)						;; Name for debugging
  (magic-number :char)					;; This is necessary to avoid memory corruption when releasing the tensor
  (grad-initialized :bool)				;; If false, the gradient is either NULL or has uninitialized memory
  (extra-multiplier :float)				;; For edgy cases such as axpy, scale
  (extra-integer :unsigned-char)		;; For edgy cases such as tref, view	
  (extra-bool :bool)					;; For edgy cases requiring additional boolean
  (extra-correspondence :pointer)	  	;; For correspondence ops
  (extra-field :pointer)				;; Put anything in here
  (extra-free :pointer))				;; Free up the garbage in extra-field
  
(cffi:defcenum ad-mode 
  ad-reverse-mode
  ad-p1-mode
  ad-p2-mode
  ad-p3-mode)  
  
(cffi:defcenum nnl2-object-type
  (:nnl2-type-ts 0)        ;; nnl2_tensor
  (:nnl2-type-ad 1)        ;; nnl2_ad_tensor  
  (:nnl2-type-unknown 2))  
  
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
 
(cffi:defcfun ("nnl2_ad_uniform" %ad-uniform) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string)
  (from :pointer)
  (to :pointer))

(cffi:defcfun ("nnl2_ad_rand" %ad-rand) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string))

(cffi:defcfun ("nnl2_ad_randn" %ad-randn) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string)
  (mean :double)
  (std :double))
  
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
  
(cffi:defcfun ("nnl2_ad_kaiming" %ad-kaiming) :pointer
  (shape :pointer)
  (rank :int)
  (dtype tensor-type)
  (requires-grad :bool)
  (name :string)
  (fan-in :int)
  (fan-out :int)
  (gain :float)
  (distribution :float)
  (mode :int))  
  
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
  
(cffi:defcfun ("nnl2_ad_sin" %ad-.sin) :pointer
  (ad-pointer :pointer)
  (mode ad-mode)
  (track-grad :bool))     

(cffi:defcfun ("nnl2_ad_cos" %ad-.cos) :pointer
  (ad-pointer :pointer)
  (mode ad-mode)
  (track-grad :bool))   

(cffi:defcfun ("nnl2_ad_asin" %ad-.asin) :pointer
  (ad-pointer :pointer)
  (mode ad-mode)
  (track-grad :bool))     

(cffi:defcfun ("nnl2_ad_acos" %ad-.acos) :pointer
  (ad-pointer :pointer)
  (mode ad-mode)
  (track-grad :bool))   
  
(cffi:defcfun ("nnl2_ad_atan" %ad-.atan) :pointer
  (ad-pointer :pointer)
  (mode ad-mode)
  (track-grad :bool))     

(cffi:defcfun ("nnl2_ad_tan" %ad-.tan) :pointer
  (ad-pointer :pointer)
  (mode ad-mode)
  (track-grad :bool)) 
  
(cffi:defcfun ("nnl2_ad_sin_inplace" %ad-.sin!) :pointer
  (ad-pointer :pointer)
  (track-grad :bool))    

(cffi:defcfun ("nnl2_ad_cos_inplace" %ad-.cos!) :pointer
  (ad-pointer :pointer)
  (track-grad :bool))   

(cffi:defcfun ("nnl2_ad_asin_inplace" %ad-.asin!) :pointer
  (ad-pointer :pointer)
  (track-grad :bool))    

(cffi:defcfun ("nnl2_ad_acos_inplace" %ad-.acos!) :pointer
  (ad-pointer :pointer)
  (track-grad :bool))   
  
(cffi:defcfun ("nnl2_ad_tan_inplace" %ad-.tan!) :pointer
  (ad-pointer :pointer)
  (track-grad :bool))    

(cffi:defcfun ("nnl2_ad_atan_inplace" %ad-.atan!) :pointer
  (ad-pointer :pointer)
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

(cffi:defcfun ("nnl2_ad_slice" %ad-slice) :pointer
  (ad-tensor :pointer)
  (slice-from :pointer)
  (slice-to :pointer)
  (mode ad-mode)
  (track-graph :bool))  
  
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
  
(cffi:defcfun ("nnl2_ad_l2norm" %ad-l2norm) :pointer
  (input :pointer)
  (force :bool)
  (mode ad-mode)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_sum_without_axis" %ad-sum-without-axis) :pointer
  (input :pointer)
  (force :bool)
  (mode ad-mode)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_sum_with_axis" %ad-sum-with-axis) :pointer
  (input :pointer)
  (axis :int)
  (keepdim :bool)
  (mode ad-mode)
  (track-graph :bool))

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
  
(cffi:defcfun ("nnl2_ad_sqrt" %ad-sqrt) :pointer
  (tensor :pointer)
  (ad-mode ad-mode)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_sqrt_inplace" %ad-sqrt-inplace) :void
  (tensor :pointer)
  (retain-graph :bool))

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
  
(cffi:defcfun ("nnl2_ad_copy" %ad-copy) :pointer
  (ad-tensor :pointer)  
  (dtype tensor-type))
  
(cffi:defcfun ("nnl2_ad_full_like" %ad-full-like) :pointer
  (ad-tensor :pointer)
  (filler :pointer))
 
(cffi:defcfun ("nnl2_ad_xavier_like" %ad-xavier-like) :pointer
  (ad-tensor :pointer)
  (in :int)
  (out :int)
  (gain :float)
  (dist :float))
  
(cffi:defcfun ("nnl2_ad_kaiming_like" %ad-kaiming-like) :pointer
  (ad-tensor :pointer)
  (fan-in :int)
  (fan-out :int)
  (gain :float)
  (distribution :float)
  (mode :int))
  
(cffi:defcfun ("nnl2_ad_uniform_like" %ad-uniform-like) :pointer
  (ad-tensor :pointer)
  (from :pointer)
  (to :pointer))  
  
(cffi:defcfun ("nnl2_ad_internal_lisp_data_pntr_share_setter" %data-pntr-share-setter) :void
  (ad-tensor :pointer)
  (tensor :pointer))
  
(cffi:defcfun ("nnl2_ad_vstack" %ad-vstack) :pointer
  (tensora :pointer)
  (tensorb :pointer)  
  (mode ad-mode)
  (track-graph :bool))

(cffi:defcfun ("nnl2_ad_hstack" %ad-hstack) :pointer
  (tensora :pointer)
  (tensorb :pointer)  
  (mode ad-mode)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_concat" %ad-concat) :pointer
  (tensora :pointer)
  (tensorb :pointer) 
  (axis :int)  
  (mode ad-mode)
  (track-graph :bool))  
  
(cffi:defcfun ("nnl2_ad_view" %ad-view) :pointer
  (ad-tensor :pointer)
  (indices :pointer)
  (num-indices :char)
  (mode ad-mode)
  (track-graph :bool)
  (force :bool))

(cffi:defcfun ("nnl2_ad_tref_getter" %ad-tref) :pointer
  (ad-tensor :pointer)
  (indices :pointer)
  (num-indices :char)
  (mode ad-mode)
  (track-graph :bool)
  (force :bool))

(cffi:defcfun ("nnl2_ad_trefw" %ad-trefw) :pointer
  (ad-tensor :pointer)
  (indices :pointer)
  (num-indices :char)
  (mode ad-mode)
  (track-graph :bool)
  (force :bool))

(cffi:defcfun ("nnl2_ad_flat" %ad-flat) :pointer
  (ad-tensor :pointer)
  (at :unsigned-long)
  (mode ad-mode)
  (track-graph :bool)
  (force :bool))
  
(cffi:defcfun ("nnl2_ad_gemmvp" %ad-gemmvp) :pointer
  (multiplicand :pointer)
  (multiplier :pointer)
  (vector :pointer)
  (mode ad-mode)
  (track-graph :bool))
  
(cffi:defcfun ("nnl2_ad_extra_multiplier_setter" %nnl2-ad-extra-multiplier-setter) :void
  (ad-tensor :pointer)
  (new-multiplier :float))

(cffi:defcfun ("nnl2_ad_extra_bool_setter" %nnl2-ad-extra-bool-setter) :void
  (ad-tensor :pointer)
  (new-bool :bool))  
  
(cffi:defcfun ("nnl2_ad_extra_integer_setter" %nnl2-ad-extra-integer-setter) :void
  (ad-tensor :pointer)
  (new-integer :unsigned-char))
  
(cffi:defcfun ("nnl2_ad_tensor_backward_fn_setter" %nnl2-ad-backward-fn-setter) :void
  (ad-tensor :pointer)
  (new-backward-fn :pointer))  
  
(cffi:defcfun ("nnl2_ad_tensor_grad_initialized_setter" %nnl2-ad-grad-initialized-setter) :void
  (ad-tensor :pointer)
  (new-bool :bool))  
  
(cffi:defcfun ("nnl2_ad_tensor_magic_number_setter" %nnl2-ad-magic-number-setter) :void
  (ad-tensor :pointer)
  (new-magic :unsigned-char))  
  
(cffi:defcfun ("nnl2_ad_tensor_name_setter" %nnl2-ad-name-setter) :void
  (tensor :pointer)
  (new-name :string))
  
(cffi:defcfun ("nnl2_ad_tensor_visited_gen_setter" %nnl2-ad-visited-gen-setter) :void
  (tensor :pointer)
  (new-visited-gen :unsigned-long))
  
(cffi:defcfun ("nnl2_ad_tensor_ts_type_setter" %nnl2-ad-object-type-setter) :void
  (tensor :pointer)
  (new-type nnl2-object-type))
  
(cffi:defcfun ("nnl2_ad_tensor_extra_correspondence_setter" %nnl2-ad-extra-correspondence-setter) :void
  (tensor :pointer)
  (new-correspondence :pointer))  
  
(cffi:defcfun ("nnl2_ad_tensor_extra_field_setter" %nnl2-ad-extra-field-setter) :void
  (tensor :pointer)
  (new-extra-field :pointer))

(cffi:defcfun ("nnl2_ad_tensor_extra_free_setter" %nnl2-ad-extra-free-setter) :void
  (tensor :pointer)
  (new-extra-free :pointer))
  
(cffi:defcfun ("nnl2_ad_narrow" %ad-narrow) :pointer
  (ad-tensor :pointer)
  (ndim :unsigned-char)
  (start :int)
  (len :int)
  (mode ad-mode)
  (track-graph :bool))  
    
(cffi:defcfun ("nnl2_ad_mse" %ad-mse) :pointer
  (prediction :pointer)
  (target :pointer)
  (force :bool)
  (mode ad-mode)
  (track-graph :bool))  
  
(cffi:defcfun ("nnl2_ad_randn_like" %ad-randn-like) :pointer  
  (ad-tensor :pointer)
  (mean :double)
  (std :double))  
	
;; -- Optimizers --  
  
(cffi:defcstruct nnl2-optim
  (tensors :pointer)  	;; Array of pointers to tensors to be optimized
  (num-tensors :size))  ;; Number of tensors in the array
  
(cffi:defcfun ("nnl2_optim_gd_create" %optim-make-gd) :pointer
  (tensors :pointer)
  (num-tensors :long)
  (learning-rate :float))  

(cffi:defcfun ("nnl2_optim_gd_optim_type_setter" %nnl2-optim-gd-optim-type-setter) :void
  (optim :pointer)
  (new-optim-type :int))

(cffi:defcfun ("nnl2_optim_gd_data_setter" %nnl2-optim-gd-data-setter) :void
  (optim :pointer)
  (new-data nnl2-optim))

(cffi:defcfun ("nnl2_optim_gd_lr_setter" %nnl2-optim-gd-lr-setter) :void
  (optim :pointer)
  (new-lr :float))

(cffi:defcfun ("nnl2_optim_tensors_setter" %nnl2-optim-tensors-setter) :void
  (optim :pointer)
  (new-tensors :pointer))

(cffi:defcfun ("nnl2_optim_num_tensors_setter" %nnl2-optim-num-tensors-setter) :void
  (optim :pointer)
  (new-num-tensors :size))  
  
;; -- Neural Networks --

(cffi:defcenum nnl2-nn-init-type
  (:identity  0)   ;; Do not perform any initialization. The weights remain untouched. Used when a user-supplied custom initializer function
  (:zeros     1)   ;; Fill the weight tensor with zeros
  (:rand      2)   ;; Fill tensor with values sampled from a uniform distribution in [0, 1]
  (:randn     3)   ;; Fill tensor with values sampled from a standard normal distribution (mean=0, std=1)
  (:xavier/normal    4)   ;; Xavier (Glorot) initialization using a normal distribution
  (:xavier/uniform   5)   ;; Xavier (Glorot) initialization using a uniform distribution
  (:kaiming/normal   6)   ;; Kaiming (He) initialization using a normal distribution
  (:kaiming/uniform  7)   ;; Kaiming (He) initialization using a uniform distribution
  (:unknown  9))	;; Undefined or unsupported initialization type

(cffi:defcenum nnl2-nn-type 
  (:fnn 0)  	    ;; Fully Connected Neural Network 
  (:rnncell 1)      ;; Vanilla Recurrent Neural Network Cell
  (:sequential 2)   ;; Sequential neural network (layers in sequence)
  (:sigmoid 3)		;; Sigmoid layer
  (:tanh 4)			;; Tanh layer
  (:relu 5)			;; ReLU layer
  (:leaky-relu 6)   ;; Leaky-ReLU layer
  (:unknown 7))     ;; Unknown or unsupported network type 

(cffi:defcenum nnl2-nn-handle-as
  (:copy 0)    ;; Make a copy of the passed tensors
  (:view 1))   ;; Take tensor pointers and work with them directly

(cffi:defcfun ("nnl2_nn_unirnn_cell_get_hidden_size" %unirnncell-hidden) :int 
  (cell :pointer))

(cffi:defcfun ("nnl2_nn_sigmoid_create" %create-nn-sigmoid) :pointer
  (approx :bool))

(cffi:defcfun ("nnl2_nn_tanh_create" %create-nn-tanh) :pointer
  (approx :bool))
  
(cffi:defcfun ("nnl2_nn_leaky_relu_create" %create-nn-leaky-relu) :pointer
  (alpha :float))  

(cffi:defcfun ("nnl2_nn_fnn_create" %create-nn-fnn) :pointer
  (in-features :int)
  (out-features :int)
  (use-bias :bool)
  (dtype tensor-type)
  (init-type nnl2-nn-init-type))
  
(cffi:defcfun ("nnl2_nn_rnn_cell_create" %create-nn-rnncell) :pointer
  (in-features :int)
  (hidden-size :int)
  (use-bias :bool)
  (dtype tensor-type)
  (init-type nnl2-nn-init-type))  
  
(cffi:defcfun ("nnl2_ann_forward" %nn-forward) :pointer
  (model :pointer)
  (args :pointer))  
  
(cffi:defcfun ("nnl2_nn_sequential_create" %create-nn-sequential) :pointer
  (num-layers :size)
  (layers :pointer))
  
(cffi:defcfun ("nnl2_nn_fnn_forward" %nn-fnn-forward) :pointer
  (nn :pointer)
  (x :pointer))
  
(cffi:defcfun ("nnl2_nn_get_type" %nn-get-type) nnl2-nn-type
  (nn :pointer))
  
(cffi:defcfun ("nnl2_ann_num_parameters" %nn-get-num-parameters) :long
  (nn :pointer)) 
  
(cffi:defcfun ("nnl2_ann_parameters" %nn-get-parameters) :pointer
  (nn :pointer))   
  
(cffi:defcfun ("nnl2_ann_free_parameters" %nn-free-parameters) :void
  (parameters :pointer))
  
(cffi:defcfun ("nnl2_ann_print" %print-model) :pointer
  (model :pointer)
  (terpri :bool)
  (depth :int))  
  
(cffi:defcfun ("nnl2_nn_fnn_manual_create" %nn-manual-fnn) :pointer 
  (in-features :int)
  (out-features :int)
  (use-bias :bool)
  (dtype tensor-type)
  (w :pointer)
  (b :pointer)
  (handle-as nnl2-nn-handle-as))
  
(cffi:defcfun ("nnl2_nn_rnn_cell_manual_create" %nn-manual-rnn-cell) :pointer
  (input-size :int)
  (hidden-size :int)
  (bias :bool)
  (dtype tensor-type)
  (wxh :pointer)
  (whh :pointer)
  (bxh :pointer)
  (bhh :pointer)
  (handle-as nnl2-nn-handle-as))  
  
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
    
(cffi:defcfun ("set_uniform_backend" %set-uniform-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_uniform_inplace_backend" %set-uniform-inplace-backend) :void
  (backend-name :string))   
 
(cffi:defcfun ("set_rand_backend" %set-rand-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_rand_inplace_backend" %set-rand-inplace-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_randn_backend" %set-randn-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_randn_inplace_backend" %set-randn-inplace-backend) :void
  (backend-name :string)) 
  
(cffi:defcfun ("set_xavier_backend" %set-xavier-backend) :void
  (backend-name :string))  
 
(cffi:defcfun ("set_xavier_inplace_backend" %set-xavier-inplace-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_kaiming_backend" %set-kaiming-backend) :void
  (backend-name :string))  
 
(cffi:defcfun ("set_kaiming_inplace_backend" %set-kaiming-inplace-backend) :void
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

(cffi:defcfun ("set_transposition_backend" %set-transposition-backend) :void
  (backend-name :string)) 

(cffi:defcfun ("set_transposition_inplace_backend" %set-transposition-inplace-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_neginplace_backend" %set-neginplace-backend) :void
  (backend-name :string))

(cffi:defcfun ("set_neg_backend" %set-neg-backend) :void
  (backend-name :string))    
  
(cffi:defcfun ("set_sqrtinplace_backend" %set-sqrtinplace-backend) :void
  (backend-name :string))

(cffi:defcfun ("set_sqrt_backend" %set-sqrt-backend) :void
  (backend-name :string)) 
  
(cffi:defcfun ("set_cos_backend" %set-cos-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_cosinplace_backend" %set-cosinplace-backend) :void
  (backend-name :string))     

(cffi:defcfun ("set_sin_backend" %set-sin-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_sininplace_backend" %set-sininplace-backend) :void
  (backend-name :string))     

(cffi:defcfun ("set_acos_backend" %set-acos-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_acosinplace_backend" %set-acosinplace-backend) :void
  (backend-name :string))     

(cffi:defcfun ("set_asin_backend" %set-asin-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_asininplace_backend" %set-asininplace-backend) :void
  (backend-name :string))
  
(cffi:defcfun ("set_tan_backend" %set-tan-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_taninplace_backend" %set-taninplace-backend) :void
  (backend-name :string))     

(cffi:defcfun ("set_atan_backend" %set-atan-backend) :void
  (backend-name :string))   
  
(cffi:defcfun ("set_ataninplace_backend" %set-ataninplace-backend) :void
  (backend-name :string))  

(cffi:defcfun ("set_mse_backend" %set-mse-backend) :void
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
(cffi:defcfun ("get_uniform_backend" %get-uniform-backend) :string) 
(cffi:defcfun ("get_uniform_inplace_backend" %get-uniform-inplace-backend) :string) 
(cffi:defcfun ("get_rand_backend" %get-rand-backend) :string) 
(cffi:defcfun ("get_rand_inplace_backend" %get-rand-inplace-backend) :string) 
(cffi:defcfun ("get_randn_backend" %get-randn-backend) :string) 
(cffi:defcfun ("get_randn_inplace_backend" %get-randn-inplace-backend) :string) 
(cffi:defcfun ("get_xavier_backend" %get-xavier-backend) :string) 
(cffi:defcfun ("get_xavier_inplace_backend" %get-xavier-inplace-backend) :string) 
(cffi:defcfun ("get_kaiming_backend" %get-kaiming-backend) :string) 
(cffi:defcfun ("get_kaiming_inplace_backend" %get-kaiming-inplace-backend) :string) 
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
(cffi:defcfun ("get_transposition_backend" %get-transposition-backend) :string)
(cffi:defcfun ("get_transposition_inplace_backend" %get-transposition-inplace-backend) :string)
(cffi:defcfun ("get_neginplace_backend" %get-neginplace-backend) :string)
(cffi:defcfun ("get_neg_backend" %get-neg-backend) :string)
(cffi:defcfun ("get_sqrtinplace_backend" %get-sqrtinplace-backend) :string)
(cffi:defcfun ("get_sqrt_backend" %get-sqrt-backend) :string)
(cffi:defcfun ("get_cosinplace_backend" %get-cosinplace-backend) :string)
(cffi:defcfun ("get_cos_backend" %get-cos-backend) :string)
(cffi:defcfun ("get_sininplace_backend" %get-sininplace-backend) :string)
(cffi:defcfun ("get_sin_backend" %get-sin-backend) :string)
(cffi:defcfun ("get_acosinplace_backend" %get-acosinplace-backend) :string)
(cffi:defcfun ("get_acos_backend" %get-acos-backend) :string)
(cffi:defcfun ("get_asininplace_backend" %get-asininplace-backend) :string)
(cffi:defcfun ("get_asin_backend" %get-asin-backend) :string)
(cffi:defcfun ("get_taninplace_backend" %get-taninplace-backend) :string)
(cffi:defcfun ("get_tan_backend" %get-tan-backend) :string)
(cffi:defcfun ("get_ataninplace_backend" %get-ataninplace-backend) :string)
(cffi:defcfun ("get_atan_backend" %get-atan-backend) :string)
(cffi:defcfun ("get_mse_backend" %get-mse-backend) :string)

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
(cffi:defcfun ("get_uniform_num_backends" %get-uniform-num-backends) :int)
(cffi:defcfun ("get_uniform_backends" %get-uniform-backends) :pointer)
(cffi:defcfun ("get_uniform_inplace_num_backends" %get-uniform-inplace-num-backends) :int)
(cffi:defcfun ("get_uniform_inplace_backends" %get-uniform-inplace-backends) :pointer)
(cffi:defcfun ("get_rand_num_backends" %get-rand-num-backends) :int)
(cffi:defcfun ("get_rand_backends" %get-rand-backends) :pointer)
(cffi:defcfun ("get_rand_inplace_num_backends" %get-rand-inplace-num-backends) :int)
(cffi:defcfun ("get_rand_inplace_backends" %get-rand-inplace-backends) :pointer)
(cffi:defcfun ("get_randn_num_backends" %get-randn-num-backends) :int)
(cffi:defcfun ("get_randn_backends" %get-randn-backends) :pointer)
(cffi:defcfun ("get_randn_inplace_num_backends" %get-randn-inplace-num-backends) :int)
(cffi:defcfun ("get_randn_inplace_backends" %get-randn-inplace-backends) :pointer)
(cffi:defcfun ("get_xavier_num_backends" %get-xavier-num-backends) :int)
(cffi:defcfun ("get_xavier_backends" %get-xavier-backends) :pointer)
(cffi:defcfun ("get_xavier_inplace_num_backends" %get-xavier-inplace-num-backends) :int)
(cffi:defcfun ("get_xavier_inplace_backends" %get-xavier-inplace-backends) :pointer)
(cffi:defcfun ("get_kaiming_num_backends" %get-kaiming-num-backends) :int)
(cffi:defcfun ("get_kaiming_backends" %get-kaiming-backends) :pointer)
(cffi:defcfun ("get_kaiming_inplace_num_backends" %get-kaiming-inplace-num-backends) :int)
(cffi:defcfun ("get_kaiming_inplace_backends" %get-kaiming-inplace-backends) :pointer)
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
(cffi:defcfun ("get_transposition_num_backends" %get-transposition-num-backends) :int)
(cffi:defcfun ("get_transposition_inplace_num_backends" %get-transposition-inplace-num-backends) :int)
(cffi:defcfun ("get_transposition_backends" %get-transposition-backends) :pointer)
(cffi:defcfun ("get_transposition_inplace_backends" %get-transposition-inplace-backends) :pointer)
(cffi:defcfun ("get_neginplace_backends" %get-neginplace-backends) :pointer)
(cffi:defcfun ("get_neginplace_num_backends" %get-neginplace-num-backends) :int)
(cffi:defcfun ("get_neg_backends" %get-neg-backends) :pointer)
(cffi:defcfun ("get_neg_num_backends" %get-neg-num-backends) :int)
(cffi:defcfun ("get_sqrtinplace_backends" %get-sqrtinplace-backends) :pointer)
(cffi:defcfun ("get_sqrtinplace_num_backends" %get-sqrtinplace-num-backends) :int)
(cffi:defcfun ("get_sqrt_backends" %get-sqrt-backends) :pointer)
(cffi:defcfun ("get_sqrt_num_backends" %get-sqrt-num-backends) :int)
(cffi:defcfun ("get_sininplace_backends" %get-sininplace-backends) :pointer)
(cffi:defcfun ("get_sininplace_num_backends" %get-sininplace-num-backends) :int)
(cffi:defcfun ("get_sin_backends" %get-sin-backends) :pointer)
(cffi:defcfun ("get_sin_num_backends" %get-sin-num-backends) :int)
(cffi:defcfun ("get_cosinplace_backends" %get-cosinplace-backends) :pointer)
(cffi:defcfun ("get_cosinplace_num_backends" %get-cosinplace-num-backends) :int)
(cffi:defcfun ("get_cos_backends" %get-cos-backends) :pointer)
(cffi:defcfun ("get_cos_num_backends" %get-cos-num-backends) :int)
(cffi:defcfun ("get_asininplace_backends" %get-asininplace-backends) :pointer)
(cffi:defcfun ("get_asininplace_num_backends" %get-asininplace-num-backends) :int)
(cffi:defcfun ("get_asin_backends" %get-asin-backends) :pointer)
(cffi:defcfun ("get_asin_num_backends" %get-asin-num-backends) :int)
(cffi:defcfun ("get_acosinplace_backends" %get-acosinplace-backends) :pointer)
(cffi:defcfun ("get_acosinplace_num_backends" %get-acosinplace-num-backends) :int)
(cffi:defcfun ("get_acos_backends" %get-acos-backends) :pointer)
(cffi:defcfun ("get_acos_num_backends" %get-acos-num-backends) :int)
(cffi:defcfun ("get_taninplace_backends" %get-taninplace-backends) :pointer)
(cffi:defcfun ("get_taninplace_num_backends" %get-taninplace-num-backends) :int)
(cffi:defcfun ("get_tan_backends" %get-tan-backends) :pointer)
(cffi:defcfun ("get_tan_num_backends" %get-tan-num-backends) :int)
(cffi:defcfun ("get_ataninplace_backends" %get-ataninplace-backends) :pointer)
(cffi:defcfun ("get_ataninplace_num_backends" %get-ataninplace-num-backends) :int)
(cffi:defcfun ("get_atan_backends" %get-atan-backends) :pointer)
(cffi:defcfun ("get_atan_num_backends" %get-atan-num-backends) :int)
(cffi:defcfun ("get_mse_backends" %get-mse-backends) :pointer)
(cffi:defcfun ("get_mse_num_backends" %get-mse-num-backends) :int)

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
