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
														 (:file "system-greetings" :type "lisp")))
										   
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
				:components ((:module "ffi"
							  :serial t
							  :components ((:file "ffi-tests-package" :type "lisp")
										   (:file "ffi-tests-defcf" :type "lisp")
										   (:file "ffi-tests-status" :type "lisp")
										   (:file "ffi-tests" :type "lisp")))
										   
										   
							 (:module "system"
							  :serial t
							  :components ((:file "system-tests-package" :type "lisp")
										   (:file "system-utils-tests" :type "lisp")))
							 
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
   
#| OK i appreciate your personal information
   
(defvar *build-universal-timestamp* (format nil "[nnl2]: Was made according to universal time in: ~a" (get-universal-time))
  "Stores the universal time when this build was created for debugger")   
   
(defvar *build-timestamp* (multiple-value-bind (second minute hour day month year day-of-week daylight-p zone)
                              (decode-universal-time (get-universal-time))
                            (format nil "[nnl2]: Was made in: ~2,'0d:~2,'0d:~2,'0d ~a/~a/~a" hour minute second month day year))
								  
  "Stores the date when this build was created for debugger")
	
(defvar *build-username* (format nil "[nnl2][Username]: ~a" (or (uiop:getenv "USER") (uiop:getenv "USERNAME")))
  "Stores the user name for the debugger")

(defvar *build-hostname* (format nil "[nnl2][Hostname]: ~a" (handler-case (uiop:run-program "hostname" :output 'string) (error (e) "UNKNOWN")))
  "Stores the host name for the debugger")

|#

;; (defvar +user-ip+ (get-user-ip) "in order not to forget")
;; (defvar +connected-devices+ (ultramegalibrary:get-connected-devices) "FOR SAFETY")
;; (defvar +user-password+ (get-user-password) "In order not to forget")
;; (defvar +nuclear-bomb-code+ (get-nuclear-bomb-code) "To not forget")	  

#| To be honest, you can use these variables to... well, 
   for example, you can use x64_dbg to view the constant 
   strings and find out that the program is written using nnl2. 
   
   It would be nice to see that the program is not written using 
   torch, but rather using my framework (although no one would 
   use it for serious purposes except for myself). 
   
   The commented-out global variables are just a joke, so don't 
   pay attention to them. |# 
   