(in-package :nnl2.threading)

;; NNL2

;; Filepath: nnl2/src/lisp/threading/threading-loops.lisp
;; File: threading-loops.lisp

;; File contains lparallel loop definitions

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defvar *pdotimes-threshold* nil
  "Auto determined threshold for parallel execution")
  
(defvar *pdotimes-min-threshold* 1000
  "Minimum number of iterations to consider parallel execution")  
  
(defvar *pdotimes-multiplier* 500
  "Multiplier per CPU core for threshold calculation")  

(defmacro pdotimes ((iterator times) &body body)
  #+lparallel
  (let ((threashold (or *pdotimes-threshold* (max *pdotimes-min-threshold* (* (lparallel:kernel-worker-count) *pdotimes-multiplier*)))))
    `(if (> ,times ,threashold)
       (lparallel:pdotimes (,iterator ,times) ,@body)
       (dotimes (,iterator ,times) ,@body)))
  
  #-lparallel
  `(dotimes (,iterator ,times) ,@body))
