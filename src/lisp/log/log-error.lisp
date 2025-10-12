(in-package :nnl2.log)

(defun error (msg &rest args)
  (let ((text (apply #'format nil msg args)))
    (%error text)
	text))
  