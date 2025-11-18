(defpackage :nnl2.intern-system
  (:use :cl)
  (:export 
    #:*current-dir*
	#:*kernel-count*
	#:*lparallel-available*))
  
(in-package :nnl2.intern-system)

(defvar *current-dir* (or (ignore-errors (directory-namestring *load-truename*)) (uiop:getcwd))
  "Contains the project path")  

(asdf:defsystem "nnl2"
  :depends-on (:cffi :fiveam)
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
														 (:file "backend-path-util" :type "lisp")
														 (:file "avx128-status" :type "lisp")
														 (:file "avx256-status" :type "lisp")
														 (:file "avx512-status" :type "lisp")
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
														 
										   (:module "threading"
										    :serial t
											:components ((:file "threading-package" :type "lisp")
														 (:file "threading-loops" :type "lisp")
														 (:file "threading-map" :type "lisp")
														 (:file "threading-let" :type "lisp")))
														 
										   (:module "format"
										    :serial t
											:components ((:file "format-package" :type "lisp")
														 (:file "format-parameters" :type "lisp")
														 (:file "format-customize" :type "lisp")))
										
										   (:module "highlevel"
										    :serial t
											:components ((:file "highlevel-package" :type "lisp")
														 (:file "highlevel-accessors" :type "lisp")
														 (:file "highlevel-utils" :type "lisp")
														 (:file "highlevel-tensor-backends" :type "lisp")
														 (:file "highlevel-ad-accessors" :type "lisp")))
														 
										   (:module "lowlevel"
											:serial t
											:components ((:file "lowlevel-package" :type "lisp")
														 (:file "lowlevel-accessors" :type "lisp")
														 (:file "lowlevel-ad-accessors" :type "lisp")))
														 
										   (:module "convert"
										    :serial t
											:components ((:file "convert-package" :type "lisp")
														 (:file "convert-nnl2-array" :type "lisp")
														 (:file "convert-nnl2-list" :type "lisp")
														 (:file "convert-nnl2-magicl" :type "lisp")
														 (:file "convert-magicl-nnl2" :type "lisp")
														 (:file "convert-magicl-array" :type "lisp")
														 (:file "convert-magicl-list" :type "lisp")))
														 
										   (:module "log"
										    :serial t
											:components ((:file "log-package" :type "lisp")
														 (:file "log-ffi" :type "lisp")
														 (:file "log-error" :type "lisp")
														 (:file "log-warning" :type "lisp")
														 (:file "log-fatal" :type "lisp")
														 (:file "log-debug" :type "lisp")
														 (:file "log-info" :type "lisp")))
														 
										   (:module "fusion"
										    :serial t
											:components ((:file "fusion-package" :type "lisp")
														 (:file "fusion-core" :type "lisp")
														 (:file "fusion-rules" :type "lisp")))
														 
										   (:module "gc"
										    :serial t
											:components ((:file "gc-package" :type "lisp")
														 (:file "gc-core" :type "lisp")))
														 
										   (:module "compile"
										    :serial t
											:components ((:file "compile-package" :type "lisp")
													     (:file "compile" :type "lisp")))
														 
										   (:module "internal"
										    :serial t
											:components ((:file "internal-package" :type "lisp")
														 (:file "internal-main" :type "lisp")))))))
							  
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
										   (:file "ts-tests-subtensor-map-inplace" :type "lisp")
										   (:file "ts-tests-real" :type "lisp")))
										   
							 (:module "llits"
							  :serial t 
							  :components ((:file "llits-tests-package" :type "lisp")
										   (:file "llits-tests" :type "lisp")))
										   
							 (:module "convert"
							  :serial t
							  :components ((:file "convert-tests-package" :type "lisp")
										   (:file "convert-tests" :type "lisp")))
							 
							 (:file "tests" :type "lisp"))))
							 
  :perform (asdf:test-op (o c)
			 (uiop:symbol-call :nnl2.tests :run-all-tests)))			 
			 
(defun get-processor-core-count ()
  "Returns the number of processor cores"
  
  (handler-case
      (cond
        ((uiop:os-unix-p)    (parse-integer (string-trim '(#\Newline) (uiop:run-program "nproc" :output :string))))	   
        ((uiop:os-windows-p) (parse-integer (string-trim '(#\Newline) (uiop:run-program "echo %NUMBER_OF_PROCESSORS%" :output :string)))) 
        (t 1))
		
    (error () 1)))
			 
(defvar *kernel-count* (get-processor-core-count))
(defvar *lparallel-available* nil)
			 
(handler-case 
    (progn
      (ql:quickload :sb-cltl2 :silent t)
	  (ql:quickload :lparallel :silent t)
	  (setq *lparallel-available* t))
	  
  (error ()))
  
#+lparallel
(setf lparallel:*kernel* (lparallel:make-kernel *kernel-count*))

(defvar *nnl2-library-name* "nnl2" 
  "Stores the name of the library used for the build. 
   This allows you to identify the build in the debugger 
   and when analyzing the executable file")  
  
(defvar *build-location* "MADE IN NNL2"
  "Same thing.
   Used to track the origin of a build in the debugger")  
   