;; Filepath: nnl2/tests/system/system-utils-tests.lisp
;; File: system-utils-tests.lisp

;; Tests for auxiliary functions: nnl2.system

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(in-package :nnl2.system.tests)

(fiveam:in-suite nnl2.system-suite)

(fiveam:test nnl2.system-utils/bool-to-int
  "tests functionality of the `nnl2.system:bool-to-int` function"

  (let ((t-call (nnl2.system:bool-to-int t))
		(nil-call (nnl2.system:bool-to-int nil)))
	
	(fiveam:is (= t-call 1))
	(fiveam:is (= nil-call 0))))
