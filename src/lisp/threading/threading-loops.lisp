(in-package :nnl2.threading)

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
