(in-package :nnl2.log)

(defun info (msg &rest args)
  (let ((text (apply #'format nil msg args)))
    (%info text)
	text))
	