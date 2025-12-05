(ql:quickload :nnl2)

(use-package :nnl2.hli.ts)

(tlet ((a (ones #(10000) :dtype :float32)))
  (print (sum a)))
  