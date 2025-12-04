(in-package :nnl2.hli.nn.utils)

;; NNL2

;; Filepath: nnl2/src/lisp/highlevel/utils/highlevel-utils-nn.lisp
;; File: highlevel-utils-nn.lisp

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun get-param (parameters name &key (test #'string=))
   "Returns a parameter from passed parameters whose name matches name according to test
   
   Args:
       parameters: List of AD tensors (parameters) to search
       name: String name of the parameter to find
       test (&key) (default: #'string=): Optional function of two string arguments for comparison
   
   Returns:
       The first parameter whose name matches name, or nil if not found.
   
   Example:
       (nnl2.hli.nn.utils:get-param parameters \"weights\") -> #<NNL2:TENSOR/FLOAT64 [2x1] ...>"
	   
  (loop for param in parameters
		when (funcall test (nnl2.hli.ad:name param) name)
		return param))

(defun get-names (parameters)
  "Returns the names of all passed parameters
   
   Args:
       parameters: Input parameters
	   
   Example:
       (nnl2.hli.nn.utils:get-names parameters) -> '(\"foo\" \"bar\" ...)"
	   
  (mapcar #'nnl2.hli.ad:name parameters))
  