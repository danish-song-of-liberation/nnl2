(in-package :nnl2.system.tests)

(fiveam:in-suite nnl2.system-suite)

(fiveam:test nnl2.system-utils/bool-to-int
  "tests functionality of the `nnl2.system:bool-to-int` function"

  (let ((t-call (nnl2.system:bool-to-int t))
		(nil-call (nnl2.system:bool-to-int nil)))
	
	(fiveam:is (= t-call 1))
	(fiveam:is (= nil-call 0))))
