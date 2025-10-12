(in-package :nnl2.log)

(cffi:defcfun ("lisp_call_error" %error) :void
  (msg :string))
  
(cffi:defcfun ("lisp_call_warning" %warning) :void
  (msg :string))  
  
(cffi:defcfun ("lisp_call_fatal" %fatal) :void
  (msg :string))    

(cffi:defcfun ("lisp_call_debug" %debug) :void
  (msg :string))      
  
(cffi:defcfun ("lisp_call_info" %info) :void
  (msg :string))        
  