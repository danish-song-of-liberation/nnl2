(defpackage :nnl2.blas
  (:use #:cl)
  (:export
    #:dgemm
	#:sgemm
	#:+cblas-row-major+
	#:+cblas-col-major+
	#:+cblas-no-trans+
	#:+cblas-trans+
	#:+cblas-conj-trans+))
	