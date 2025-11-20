(defpackage :nnl2.optim
  (:use #:cl)
  
  (:export
    #:step!
	#:zero-grad!
	#:free
	#:gd
	#:leto
	#:leto*))
  