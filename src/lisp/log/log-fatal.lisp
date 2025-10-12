(in-package :nnl2.log)

(defun fatal (msg &rest args)
  (let ((text (apply #'format nil msg args)))
    (%fatal text)
	text))
