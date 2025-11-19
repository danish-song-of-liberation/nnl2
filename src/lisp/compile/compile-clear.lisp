(in-package :nnl2.compile)

;; NNL2

;; Filepath: nnl2/src/lisp/compile/compile-clear.lisp
;; File: compile-clear.lisp

;; Declaration of functions to clear all files from the 
;; nnl2/compile directory when loading the library

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defparameter *compile-files* (uiop:directory-files (concatenate 'string nnl2.intern-system:*current-dir* "compile/"))
  "list of all files in the ''compile/'' subdirectory")

(defun clear-compile ()
  "Deletes all files previously found in the ''compile/'' directory"
  (dolist (file *compile-files*)
    (when (probe-file file) (delete-file file))))

(clear-compile)
