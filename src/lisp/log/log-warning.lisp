(in-package :nnl2.log)

(defun warning (msg &rest args)
  (let ((text (apply #'format nil msg args)))
    (%warning text)
	text))
