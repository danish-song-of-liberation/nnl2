(in-package :nnl2.log)

;; NNL2

;; Filepath: nnl2/src/lisp/log/log-ffi.lisp
;; File: log-ffi.lisp

;; Contains import of C logging function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

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
  