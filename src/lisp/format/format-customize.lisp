(in-package :nnl2.format)

(defun customize-tensor-format (&key max-rows max-cols show-rows-after-skip show-cols-after-skip)
  (when max-rows
    (assert (and (not (zerop max-rows)) (plusp max-rows)) nil "```:max-rows``` Should be positive (not zero and non-negative)")
    (setf nnl2.format:*nnl2-max-rows-format* max-rows))
  
  (when max-cols
    (assert (and (not (zerop max-cols)) (plusp max-cols)) nil "```:max-cols``` Should be positive (not zero and non-negative)")
    (setf nnl2.format:*nnl2-max-cols-format* max-cols))

  (when show-rows-after-skip
	(assert (and (not (zerop show-rows-after-skip)) (plusp show-rows-after-skip)) nil "```:show-rows-after-skip``` Should be positive (not zero and non-negative)")
    (setf nnl2.format:*nnl2-show-rows-after-skip* show-rows-after-skip))

  (when show-cols-after-skip
    (assert (and (not (zerop show-cols-after-skip)) (plusp show-cols-after-skip)) nil "```:show-cols-after-skip``` Should be positive (not zero and non-negative)")
    (setf nnl2.format:*nnl2-show-cols-after-skip* show-cols-after-skip)))
	