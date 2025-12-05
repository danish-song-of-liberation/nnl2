(ql:quickload :nnl2)

(use-package :nnl2.hli.ts)

(tlet ((a (ones #(3 3)))
       (b (ones #(3 3))))
       
  (tlet ((c (.* a b)))
    c))
	