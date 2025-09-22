(defpackage :nnl2.lli
  (:use :cl)
  (:export))
  
(defpackage :nnl2.lli.ts
  (:use :cl)
  (:export 
    #:flat 
    #:trefw 
    #:data 
    #:mem-aref 
    #:iterate-across-tensor-data 
    #:iterate-across-tensor-shape 
    #:alignment))
  