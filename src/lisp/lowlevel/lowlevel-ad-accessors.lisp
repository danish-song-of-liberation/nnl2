(in-package :nnl2.lli.ad)

;; NNL2

;; Filepath: nnl2/src/lisp/lowlevel/lowlevel-ad-accessors.lisp
;; File: lowlevel-ad-accessors.lisp

;; Contains lowlevel AD advanced functions

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

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
  
(cffi:defcfun ("nnl2_ad_tensor_visited_gen_getter" visited-gen) :unsigned-long
  (ad-tensor :pointer))

(defun (setf visited-gen) (new-gen ad-tensor)
  (nnl2.ffi:%nnl2-ad-visited-gen-setter ad-tensor new-gen))    
  
(cffi:defcfun ("nnl2_ad_tensor_ts_type_getter" object-type) nnl2-object-type
  (ad-tensor :pointer))
  
(defun (setf object-type) (new-obj ad-tensor)
  (nnl2.ffi:%nnl2-ad-object-type-setter ad-tensor new-obj))    
    
(defmacro iterate-across-tensor-data ((iterator ad-tensor) &body body)
  "Iterates over each element of the AD tensor's data"
  `(nnl2.lli.ts:iatd (,iterator (nnl2.hli.ad:data ,ad-tensor)) ,@body))	

(defmacro iatd ((iterator ad-tensor) &body body)
  "Shorthand for `iterate-across-tensor-data`"
  `(nnl2.lli.ts:iatd (,iterator (nnl2.hli.ad:data ,ad-tensor)) ,@body))	
  
(defmacro parallel-iterate-across-tensor-data ((iterator ad-tensor) &body body)
  "Parallel iteration over each element of the AD tensor's data"
  `(nnl2.lli.ts:piatd (,iterator (nnl2.hli.ad:data ,ad-tensor)) ,@body))		
  
(defmacro piatd ((iterator ad-tensor) &body body)
  "Shorthand for `parallel-iterate-across-tensor-data`"
  `(nnl2.lli.ts:piatd (,iterator (nnl2.hli.ad:data ,ad-tensor)) ,@body))			
 
(defmacro iterate-across-tensor-grad ((iterator ad-tensor) &body body)
  "Iterates over each element of the AD tensor's gradient"
  `(nnl2.lli.ts:iatd (,iterator (nnl2.hli.ad:grad ,ad-tensor)) ,@body))	

(defmacro iatg ((iterator ad-tensor) &body body)
  "Shorthand for `iterate-across-tensor-grad`"
  `(nnl2.lli.ts:iatd (,iterator (nnl2.hli.ad:grad ,ad-tensor)) ,@body))	
  
(defmacro parallel-iterate-across-tensor-grad ((iterator ad-tensor) &body body)
  "Parallel iteration over each element of the AD tensor's gradient"
  `(nnl2.lli.ts:piatd (,iterator (nnl2.hli.ad:grad ,ad-tensor)) ,@body))		
  
(defmacro piatg ((iterator ad-tensor) &body body)
  "Shorthand for `parallel-iterate-across-tensor-grad`"
  `(nnl2.lli.ts:piatd (,iterator (nnl2.hli.ad:grad ,ad-tensor)) ,@body))		

(defmacro iterate-across-tensor-shape ((iterator ad-tensor) &body body)
  "Iterates over the shape vector of the AD tensor's data"
  `(nnl2.lli.ts:iats (,iterator (nnl2.hli.ad:data ,ad-tensor)) ,@body))	 
  
(defmacro iats ((iterator ad-tensor) &body body)
  "Shorthand for `iterate-across-tensor-shape`"
  `(nnl2.lli.ts:iats (,iterator (nnl2.hli.ad:data ,ad-tensor)) ,@body))	 

(defmacro iterate-across-tensor-strides ((iterator ad-tensor) &body body)
  "Iterates over the strides vector of the AD tensor's data"
  `(nnl2.lli.ts:iatst (,iterator (nnl2.hli.ad:data ,ad-tensor)) ,@body))	 
  
(defmacro iatst ((iterator ad-tensor) &body body)
  "Shorthand for `iterate-across-tensor-strides`"
  `(nnl2.lli.ts:iatst (,iterator (nnl2.hli.ad:data ,ad-tensor)) ,@body))	 
  
(defun trefw (tensor indices &key (track-graph nnl2.system:*ad-default-track-graph*) force)
  "Returns a tensor element at specified indices"
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%ad-trefw tensor shape rank nnl2.ffi:ad-reverse-mode track-graph force)))
	
(defun flat (tensor index &key (track-graph nnl2.system:*ad-default-track-graph*) force)
  "Retrieves the tensor element value by linear index"
  (nnl2.ffi:%ad-flat tensor index nnl2.ffi:ad-reverse-mode track-graph force))
  
	