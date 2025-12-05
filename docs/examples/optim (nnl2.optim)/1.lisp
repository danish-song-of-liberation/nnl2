(ql:quickload :nnl2)

(use-package :nnl2.hli.ad)
(use-package :nnl2.hli.ad.r)

(tlet ((a (ones #(5 5) :requires-grad t))
       (b (ones #(5 5) :requires-grad t)))

  (nnl2.optim:leto ((rlyeh (nnl2.optim:gd (list a b) :lr 0.1)))
    (dotimes (epochs 10)
      (tlet ((c (.+ a b)))
        (backpropagation c)

        (nnl2.optim:step! rlyeh)
        (nnl2.optim:zero-grad! rlyeh)))))
		