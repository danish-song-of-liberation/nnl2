(ql:quickload :nnl2)

(use-package :nnl2.hli.ts)

(nnl2.compile:compile ()
  "int qux() { return 0; }")

(cffi:defcfun ("qux" qux) :int)

(print (qux)) ;; 0
