(in-package :nnl2.system)

(defparameter +log-tests-folder+ "log/tests/"
  "Relative path to the test log directory.
   Not defconstant because it will throw an error")

(defparameter +nnl2-tests-logs-path+ 
  (merge-pathnames +log-tests-folder+ nnl2.intern-system:*current-dir*)
  
  "Not defconstant for the same reason (will throw an error)
   The variable contains the full path to nnl2/log/tests/")
   
(defparameter +nnl2-filepath-log-path+
  (merge-pathnames "tensor-tests-log" +nnl2-tests-logs-path+)
  
  "contains the full path to nnl2/log/tests/tests-log")   
