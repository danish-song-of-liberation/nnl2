(in-package :nnl2.internal)

(defun ts-axpy-regional! (addend summand &key from to (alpha 1.0s0))
  (let* ((rank (nnl2.hli.ts:rank addend))
		 (from (if from from (make-array (list rank))))
		 (to (if to to (make-array (list to))))
		 (from-pntr (nnl2.hli:make-shape-pntr from))
		 (to-pntr (nnl2.hli:make-shape-pntr to)))
		
	(nnl2.ffi:%ts-regional-axpy-inplace addend summand alpha from-pntr to-pntr)))
	