;; NNL2

;; Filepath: nnl2/src/lisp/format/format-package.lisp
;; File: format-package.lisp

;; Defining the :nnl2.format package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.format
  (:use :cl)
  (:export
   #:*nnl2-max-rows-format*
   #:*nnl2-max-cols-format*
   #:*nnl2-show-rows-after-skip*
   #:*nnl2-show-cols-after-skip*
   #:customize-tensor-format))
 