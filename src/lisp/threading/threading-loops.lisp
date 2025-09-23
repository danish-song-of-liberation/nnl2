(in-package :nnl2.threading)

(defmacro pdotimes ((iterator times) &body body)
  #+lparallel
  `(lparallel:pdotimes (,iterator ,times) ,@body)
  
  #-lparallel
  `(dotimes (,iterator ,times) ,@body))
