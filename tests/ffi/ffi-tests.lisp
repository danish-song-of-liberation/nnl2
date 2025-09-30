(in-package :nnl2.ffi.tests)

;; Filepath: nnl2/tests/ffi/ffi-tests.lisp
;; File: ffi-tests.lisp

;; Tests for ffi auxiliary functions 

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.ffi-suite)

(fiveam:test nnl2.ffi-test/ffi-1
  (fiveam:is (= 0 (nnl-ffi-test-1))))

(fiveam:test nnl2.ffi-test/ffi-2
  (fiveam:is (= (+ 3 4) (nnl-ffi-test-2)))) ;; nnl-ffi-test-2 returns 3 + 4 i.e. 7
  