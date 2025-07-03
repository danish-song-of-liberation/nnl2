(in-package :nnl2.ffi.tests)

(fiveam:def-suite nnl2.ffi-suite :description "Tests for nnl.math")

(fiveam:in-suite nnl2.ffi-suite)

(fiveam:test nnl.ffi-test/ffi-1
  (fiveam:is (= 0 (nnl-ffi-test-1))))

(fiveam:test nnl.ffi-test/ffi-2
  (fiveam:is (= (+ 3 4) (nnl-ffi-test-2)))) ;; nnl-ffi-test-2 returns 3 + 4 i.e. 7
  