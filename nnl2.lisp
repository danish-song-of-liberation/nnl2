(defpackage :nnl2.system
  (:use :cl)
  (:export :nnl2-ffi-test-1 :nnl2-ffi-test-2))
  
(in-package :nnl2.system)

(defvar *current-dir* (directory-namestring *load-truename*))

(asdf:defsystem "nnl2"
  :depends-on (:uiop :cffi :fiveam)
  :license "MIT"
  :description "Common Lisp (CL) neural networks framework"
  :serial t
  :author "danish-song-of-liberation"
  :maintainer (:email "nnl.dev@proton.me")
  :homepage "https://github.com/danish-song-of-liberation/nnl2"
  
  :components ((:module "src"
				:components ((:module "lisp"
							  :components ((:module "ffi"
											:components ((:file "ffi-package" :type "lisp")
														 (:file "ffi-with-c" :type "lisp")))))))
							  
			   (:module "tests"
				:components ((:module "ffi"
							  :components ((:file "ffi-tests-package" :type "lisp")
										   (:file "ffi-tests-defcf" :type "lisp")
										   (:file "ffi-tests" :type "lisp")))
							 
							 (:file "tests" :type "lisp"))))
							 
  :perform (asdf:test-op (o c)
			 (uiop:symbol-call :nnl2.tests :run-all-tests)))
  