(in-package :nnl2.threading)

(defun pmapcar (function &rest sequences)
  #+lparallel
  (apply #'lparallel:pmapcar function sequences)
  
  #-lparallel
  (apply #'mapcar function sequences))
  
(defun pmapc (function &rest lists)
  #+lparallel
  (apply #'lparallel:pmapc function lists)
  
  #-lparallel
  (apply #'mapc function lists))
  
(defun pmap (result-type function &rest sequences)
  #+lparallel
  (apply #'lparallel:pmap result-type function sequences)

  #-lparallel
  (apply #'map result-type function sequences))  
  
(defun pmapcan (function &rest lists)
  #+lparallel
  (apply #'lparallel:pmapcan function lists)
  
  #-lparallel
  (apply #'pmapcan function lists))
  
(defun pmaplist (function &rest lists)
  #+lparallel
  (apply #'lparallel:pmaplist function lists)
  
  #-lparallel
  (apply #'maplist function lists))
  
(defun pmap-into (result-sequence function &rest sequences)
  #+lparallel
  (apply #'lparallel:pmap-into result-sequence function sequences)
  
  #-lparallel
  (apply #'map-into result-sequence function sequences))
  
(defun pmapl (function &rest lists)
  #+lparallel
  (apply #'lparallel:pmapl function lists)

  #-lparallel
  (apply #'mapl function lists))  
  