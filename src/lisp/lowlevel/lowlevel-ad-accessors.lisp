(in-package :nnl2.lli.ad)

(cffi:defcenum nnl2-object-type
  (:nnl2-type-ts 0)     ;; nnl2_tensor
  (:nnl2-type-ad 1)     ;; nnl2_ad_tensor  
  (:nnl2-type-unknown 2))
  
(cffi:defcfun ("nnl2_ad_extra_multiplier_getter" extra-multiplier) :float
  (ad-tensor :pointer))
  
(defun (setf extra-multiplier) (new-multiplier ad-tensor)
  (nnl2.ffi:%nnl2-ad-extra-multiplier-setter ad-tensor new-multiplier))
  
(cffi:defcfun ("nnl2_ad_extra_bool_getter" extra-bool) :bool
  (ad-tensor :pointer)) 

(defun (setf extra-bool) (new-bool ad-tensor)
  (nnl2.ffi:%nnl2-ad-extra-bool-setter ad-tensor new-bool))  
  
(cffi:defcfun ("nnl2_ad_extra_integer_getter" extra-integer) :unsigned-char
  (ad-tensor :pointer))  
  
(defun (setf extra-integer) (new-integer ad-tensor)
  (nnl2.ffi:%nnl2-ad-extra-integer-setter ad-tensor new-integer))  
  
(cffi:defcfun ("nnl2_ad_tensor_backward_fn_getter" backward-fn) :pointer
  (ad-tensor :pointer))  
  
(defun (setf backward-fn) (new-fn ad-tensor) 
  (nnl2.ffi:%nnl2-ad-backward-fn-setter ad-tensor new-fn))   
  
(cffi:defcfun ("nnl2_ad_tensor_grad_initialized_getter" grad-initialized) :bool
  (ad-tensor :pointer))
  
(defun (setf grad-initialized) (new-bool ad-tensor)
  (nnl2.ffi:%nnl2-ad-grad-initialized-setter ad-tensor new-bool))    
  
(cffi:defcfun ("nnl2_ad_tensor_magic_number_getter" magic-number) :char
  (ad-tensor :pointer))

(defun (setf magic-number) (new-magic ad-tensor)
  (nnl2.ffi:%nnl2-ad-magic-number-setter ad-tensor new-magic))    
  
(cffi:defcfun ("nnl2_ad_tensor_name_getter" name) :string
  (ad-tensor :pointer))

(defun (setf name) (new-name ad-tensor)
  (nnl2.ffi:%nnl2-ad-name-setter ad-tensor new-name))    
  
(cffi:defcfun ("nnl2_ad_tensor_visited_gen_getter" visited-gen) :unsigned-long
  (ad-tensor :pointer))

(defun (setf visited-gen) (new-gen ad-tensor)
  (nnl2.ffi:%nnl2-ad-visited-gen-setter ad-tensor new-gen))    
  
(cffi:defcfun ("nnl2_ad_tensor_ts_type_getter" object-type) nnl2-object-type
  (ad-tensor :pointer))
  
(defun (setf object-type) (new-obj ad-tensor)
  (nnl2.ffi:%nnl2-ad-object-type-setter ad-tensor new-obj))    
    