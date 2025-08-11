(defpackage :nnl2.intern-system
  (:use :cl)
  (:export #:*current-dir*))
  
(in-package :nnl2.intern-system)

(defvar *current-dir* (or (ignore-errors (directory-namestring *load-truename*)) (uiop:getcwd))
  "Contains the project path")  

(asdf:defsystem "nnl2"
  :depends-on (:cffi :fiveam :cl-json)
  :license "MIT"
  :description "Common Lisp (CL) neural networks framework"
  :serial t
  :version "0.1.0"
  :author "danish-song-of-liberation"
  :maintainer (:email "nnl.dev@proton.me")
  :homepage "https://github.com/danish-song-of-liberation/nnl2"
  
  :components ((:module "src"
				:serial t
				:components ((:module "lisp"
							  :serial t
							  :components ((:module "backend-status"
											:serial t
											:components ((:file "backend-status-package" :type "lisp")
														 (:file "openblas0330woa64static-status" :type "lisp")))
											
										   (:module "system"
											:serial t
										    :components ((:file "system-package" :type "lisp")
														 (:file "system-utils" :type "lisp")
														 (:file "system-impl-upd" :type "lisp")
														 (:file "system-config" :type "lisp")
														 (:file "system-greetings" :type "lisp")
														 (:file "system-vars" :type "lisp")))
										   
										   (:module "ffi"
											:serial t
											:components ((:file "ffi-package" :type "lisp")
														 (:file "ffi-with-c" :type "lisp")
														 (:file "ffi-c-core" :type "lisp")
														 (:file "ffi-status" :type "lisp")))
										
										   (:module "highlevel"
										    :serial t
											:components ((:file "highlevel-package" :type "lisp")
														 (:file "highlevel-accessors" :type "lisp")
														 (:file "highlevel-utils" :type "lisp")))))))
							  
			   (:module "tests"
			    :serial t
				:components ((:module "utils"
							  :serial t
							  :components ((:file "utils-package" :type "lisp")
										   (:file "utils-trivial-errors" :type "lisp")
										   (:file "utils-approx-equal" :type "lisp")
										   (:file "utils-log-wrapper" :type "lisp")))
							  
							 (:module "ffi"
							  :serial t
							  :components ((:file "ffi-tests-package" :type "lisp")
										   (:file "ffi-tests-defcf" :type "lisp")
										   (:file "ffi-tests-status" :type "lisp")
										   (:file "ffi-tests" :type "lisp")))
										   
										   
							 (:module "system"
							  :serial t
							  :components ((:file "system-tests-package" :type "lisp")
										   (:file "system-utils-tests" :type "lisp")))
										   
							 (:module "ts"
							  :serial t
							  :components ((:file "ts-tests-package" :type "lisp")
										   (:file "ts-tests-basic-tensors" :type "lisp")
										   (:file "ts-tests-basic-operations" :type "lisp")
										   (:file "ts-tests-basic-operations-inplace" :type "lisp")
										   (:file "ts-tests-trivial-operations" :type "lisp")
										   (:file "ts-tests-trivial-operations-inplace" :type "lisp")
										   (:file "ts-tests-activation-functions" :type "lisp")
										   (:file "ts-tests-activation-functions-inplace" :type "lisp")
										   (:file "ts-tests-wise-element-map" :type "lisp")
										   (:file "ts-tests-wise-element-map-inplace" :type "lisp")
										   (:file "ts-tests-concat" :type "lisp")
										   (:file "ts-tests-correspondence" :type "lisp")
										   (:file "ts-tests-correspondence-inplace" :type "lisp")
										   (:file "ts-tests-broadcasting" :type "lisp")
										   (:file "ts-tests-broadcasting-inplace" :type "lisp")
										   (:file "ts-tests-like-constructors" :type "lisp")
										   (:file "ts-tests-trans" :type "lisp")
										   (:file "ts-tests-subtensor-map" :type "lisp")
										   (:file "ts-tests-subtensor-map-inplace" :type "lisp")))
							 
							 (:file "tests" :type "lisp"))))
							 
  :perform (asdf:test-op (o c)
			 (uiop:symbol-call :nnl2.tests :run-all-tests)))
  
(defvar *nnl2-framework-name* "nnl2" 
  "Stores the name of the framework used for the build. 
   This allows you to identify the build in the debugger 
   and when analyzing the executable file")  
  
(defvar *build-location* "MADE IN NNL2"
  "Same thing.
   Used to track the origin of a build in the debugger")  
   