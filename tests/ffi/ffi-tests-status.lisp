(in-package :nnl2.ffi.tests)

(fiveam:in-suite nnl2.ffi-suite)

(fiveam:test nnl2.ffi-test/get-openblas0330woa64static-status
  (let ((real-status (cdar nnl2.system:+architecture+))
		(ffi-status (nnl2.ffi:get-openblas0330woa64static-status)))
		
	(fiveam:is-true (and real-status (zerop ffi-status)))))
