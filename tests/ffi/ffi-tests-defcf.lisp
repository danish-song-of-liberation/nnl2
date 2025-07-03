(in-package :nnl2.ffi.tests)

(cffi:defcfun ("__nnl2_test_1" nnl-ffi-test-1)
  :int
  "simple test for nnl2 ffi interface
   should just return 0.")   
 
(cffi:defcfun ("__nnl2_test_2" nnl-ffi-test-2)
  :int
  "simple test for nnl2 ffi interface
   should return 3 + 4 (i.g. 7)") 

(declaim (ftype (function () integer) nnl-ffi-test-1 nnl-ffi-test-2))
