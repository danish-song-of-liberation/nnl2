(ql:quickload :nnl2)

(use-package :nnl2.hli.ts)

(nnl2.compile.aot:function ()
  "int qux(int a) { return a; }")

(print (qux 123)) ;; 123
