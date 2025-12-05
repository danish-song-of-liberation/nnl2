(ql:quickload :nnl2)

(use-package :nnl2.hli.ts)

(nnl2.compile.aot:function ()
  "int qux() { return 1; }")

(print (qux)) ;; 1
