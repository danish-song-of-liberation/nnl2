(in-package :nnl2.system)

;; NNL2

;; Filepath: nnl2/src/lisp/system/system-vars.lisp
;; File: system-vars.lisp

;; File contains definition of system variables

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defparameter +log-tests-folder+ "log/tests/"
  "Relative path to the test log directory
   Not defconstant because it will throw an error")

(defparameter +nnl2-tests-logs-path+ 
  (merge-pathnames +log-tests-folder+ nnl2.intern-system:*current-dir*)
  
  "Not defconstant for the same reason (will throw an error)
   The variable contains the full path to nnl2/log/tests/")
   
(defparameter +nnl2-filepath-log-path+
  (merge-pathnames "nnl2.hli.ts (TESTS)" +nnl2-tests-logs-path+)
  
  "contains the full path to nnl2/log/tests/tests-log")   
	
(defparameter *leaky-relu-default-shift* 0.01s0)	
	