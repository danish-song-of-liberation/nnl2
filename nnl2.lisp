;W.I.P.

(defpackage :nnl2.system
  (:use :cl)
  (:export))
  
(in-package :nnl2.system)

;; GET REKT 
;; PERMISSION DENIED
;; SUBPROCESS RETURNED 1
;; FILE BAZ IS MISSING

(defvar *current-dir* (directory-namestring *load-truename*))

(defun select-gcc-compiler ()
  #+win32
  "gcc"
  #-:win32
  "cc")

(defun compile-c-library (c-dir) 
  (print t)
  (let ((c-file (uiop:merge-pathnames* "nnl2_core.c" c-dir))
        (dll-file (uiop:merge-pathnames* "nnl2_core.dll" c-dir)))
		 
	(assert (probe-file c-file) nil (format nil "File ~a is missing" c-file))
		 
    (let ((command (format nil "~a -shared -o \"~a\" \"~a\" -Wall"
					 (select-gcc-compiler)
					 (uiop:native-namestring dll-file)
					 (uiop:native-namestring c-file))))
			
	  (handler-case 
		  (multiple-value-bind (output error-output exit-code)
			(uiop:run-program command :error-output :interactive :output :string)
			
			(cond 
			  ((zerop exit-code)
				 (return-from compile-c-library 0))
				 
			  (t (return-from compile-c-library 1))))
		  
		(error (e)
		  (error "~a~%" e))))))

(unless (probe-file (format nil "~asrc/c/nnl2_core.dll" *current-dir*)) 
  (uiop:run-program (format nil "gcc -shared -o ~asrc/c/nnl2_core.dll ~asrc/c/nnl2_core.c -Wall"
					  *current-dir*
					  *current-dir*)
					  
    :error-output :interactive))

(asdf:defsystem "nnl2"
  :depends-on (:uiop :cffi)
  :license "MIT"
  :description "Common Lisp (CL) neural networks framework"
  :serial t
  :author "danish-song-of-liberation"
  :homepage "https://github.com/danish-song-of-liberation/nnl2"
  
  :components ((:module "src"
				:components ((:module "lisp"
							  :components ((:file "ffi" :type "lisp"))))))
  
  :perform (asdf:load-op :after (op system)
			  (let* ((c-dir (uiop:merge-pathnames* "src/c/" *current-dir*)))
			    (compile-c-library c-dir))))
  