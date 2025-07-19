(in-package :nnl2.hli)

(defun make-foreign-pointer (value dtype)
  (let ((pntr (cffi:foreign-alloc dtype)))
    (setf (cffi:mem-ref pntr dtype) value)
	
	pntr))
