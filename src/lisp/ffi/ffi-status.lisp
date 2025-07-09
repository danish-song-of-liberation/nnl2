(in-package :nnl2.ffi)

(cffi:defcfun ("openblas0330woa64static_status" get-openblas0330woa64static-status)
  :int
  "get openblas0330woa64static status.")   
 
(declaim (ftype (function () integer) nnl-ffi-test-1 nnl-ffi-test-2)) 
