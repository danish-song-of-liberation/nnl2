(in-package :nnl2.ffi.tests)

;; Filepath: nnl2/tests/ffi/ffi-tests-defcf.lisp
;; File: ffi-tests-defcf.lisp

;; Simple tests for ffi interface functions 

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(cffi:defcfun ("__nnl2_test_1" nnl-ffi-test-1)
  :int
  "Simple test for nnl2 ffi interface
   should just return 0.")   
 
(cffi:defcfun ("__nnl2_test_2" nnl-ffi-test-2)
  :int
  "Simple test for nnl2 ffi interface
   should return 3 + 4 (i.g. 7)") 

(declaim (ftype (function () integer) nnl-ffi-test-1 nnl-ffi-test-2))
