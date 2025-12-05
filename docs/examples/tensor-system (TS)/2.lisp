(ql:quickload :nnl2)

(use-package :nnl2.hli.ts)

(tlet ((a (ones #(3 3) :dtype :float32))
       (b (ones #(3) :dtype :int32)))

  (+= a b)

  (print-tensor a))
