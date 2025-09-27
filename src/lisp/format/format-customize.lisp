(in-package :nnl2.format)

;; NNL2

;; Filepath: nnl2/src/lisp/format/format-customize.lisp
;; File: format-customize.lisp

;; File contains function to cuztomize format of tensors

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

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
	