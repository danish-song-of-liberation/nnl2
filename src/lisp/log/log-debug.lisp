(in-package :nnl2.log)

(defun debug (msg &rest args)
  (let ((text (apply #'format nil msg args)))
    (%debug text)
	text))
	