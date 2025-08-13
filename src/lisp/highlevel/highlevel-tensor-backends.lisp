(in-package :nnl2.hli.ts)

(defun use-backend/abs (name)
  (let ((sig (string-upcase	(symbol-name name))))
    (nnl2.ffi:%set-abs-backend sig)
    (nnl2.ffi:%set-abs-inplace-backend sig)))
  
(defun use-backend (name)
  (use-backend/abs name))  
  