(in-package :nnl2.ffi.tests)

(fiveam:in-suite nnl2.ffi-suite)

(fiveam:test nnl2.ffi-test/ffi-1
  (fiveam:is (= 0 (nnl-ffi-test-1))))

(fiveam:test nnl2.ffi-test/ffi-2
  (fiveam:is (= (+ 3 4) (nnl-ffi-test-2)))) ;; nnl-ffi-test-2 returns 3 + 4 i.e. 7
  