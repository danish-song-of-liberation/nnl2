(defpackage :nnl2.log
  (:use :cl)
  
  (:shadow 
    #:error
	#:warning
	#:debug)
	
  (:export
    #:error
	#:warning
	#:fatal
	#:debug
	#:info))
