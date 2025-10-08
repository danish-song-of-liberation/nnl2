;; NNL2

;; Filepath: nnl2/src/lisp/convert/convert-package.lisp
;; File: convert-package.lisp

;; Definition of :nnl2.convert package

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defpackage :nnl2.convert
  (:use :cl)
  (:export
    #:auto-install-magicl-choice
    #:nnl2->magicl
	#:nnl2->array
	#:nnl2->list
	#:magicl->nnl2))
  
(in-package :nnl2.convert)  

(defparameter *nnl2-default-magicl-install-status* t
  "Default choice in function ```auto-install-magicl```")
  
(defun auto-install-magicl ()
  (ql:quickload :magicl))  
  
(defun check-status (awanser)
  (or (char= awanser #\Y) (char= awanser #\y)
	  (char= awanser #\N) (char= awanser #\n)))
	  
(defun y-pressed (awanser) (or (char= awanser #\Y) (char= awanser #\y)))	
(defun n-pressed (awanser) (or (char= awanser #\N) (char= awanser #\n)))	  
  
(defun auto-install-magicl-choice ()
  (format t "~%MAGICL was not found. Did you happen to forget ```(ql:quickload :magicl)``` ?~%Do you want to download MAGICL right now automatically? (Default: ~a)~%Y - Yes~%N - No (Returns nil)~%~%"
    (if *nnl2-default-magicl-install-status* "yes" "no"))
	
  (let ((awanser #\?))
    (loop while (not (check-status)) 
		  do (setq awanser (read-char)))
		  
	(cond
	  ((y-pressed awanser) (auto-install-magicl) t)
	  ((n-pressed awanser) nil))))
  