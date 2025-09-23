(in-package :nnl2.threading)

(defmacro plet ((&rest vars) &body body)
  #+lparallel
  `(lparallel:plet ,vars ,@body)
  
  #-lparallel
  `(let ,vars ,@body))
  
(defmacro plet-if (condition (&rest vars) &body body)
  #+lparallel
  `(lparallel:plet-if ,condition ,vars ,@body)
  
  #-lparallel
  `(let ,vars ,@body))
  