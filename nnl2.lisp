(defpackage :nnl2.intern-system
  (:use :cl)
  (:export #:*current-dir*))
  
(in-package :nnl2.intern-system)

(defun ultra-mega-super-turbo-cublas-ai-neuroquantum-backpropagation-through-time-algorithm-with-blockchain-hyperengine 
  (a b c s u uu aa ss ja da dja uq fx fu wwwq zx fdd fjj irw qw e dsf ads fasd f asdf vz cv zzzzz zzzz sdg)
  
  (- 1 (* (/ (- (exp (+ a b c (- a b s u aa (/ 1 (+ 1 (exp (- uu))))))) (- (exp (+ a b c (- a b s u aa (/ 1 (+ 1 (exp (- uu))))))))) 
             (+ (exp (+ a b c (- a b s u aa (/ 1 (+ 1 (exp (- uu))))))) (- (exp (+ a b c (- a b s u aa (/ 1 (+ 1 (exp (- uu))))))))))
          (/ (- (exp (+ a b c (- a b s u aa (/ 1 (+ 1 (exp (- uu))))))) (- (exp (+ a b c (- a b s u aa (/ 1 (+ 1 (exp (- uu))))))))) 
             (+ (exp (+ a b c (- a b s u aa (/ 1 (+ 1 (exp (- uu))))))) (- (exp (+ a b c (- a b s u aa (/ 1 (+ 1 (exp (- uu))))))))))))) 		

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
														 (:file "system-config" :type "lisp")))
										   
										   (:module "ffi"
											:serial t
											:components ((:file "ffi-package" :type "lisp")
														 (:file "ffi-with-c" :type "lisp")
														 (:file "ffi-c-core" :type "lisp")
														 (:file "ffi-status" :type "lisp")))))))
							  
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
  