(in-package :nnl2.ffi.tests)

;; Filepath: nnl2/tests/ffi/ffi-tests-status.lisp
;; File: ffi-tests-status.lisp

;; Tests for ffi status functions

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.ffi-suite)

(fiveam:test nnl2.ffi-test/get-openblas0330woa64static-status
  (let ((real-status (nnl2.system:alist-symbol-to-bool (nnl2.system:assoc-key 'nnl2.system::*openblas0330woa64static* nnl2.system:+architecture+)))
		(ffi-status (nnl2.ffi:get-openblas0330woa64static-status)))
		
	(fiveam:is-true (and real-status (zerop ffi-status)))))
