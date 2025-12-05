(ql:quickload :nnl2)

(use-package :nnl2.hli.nn)
(use-package :nnl2.hli.ad)
(use-package :nnl2.hli.ad.r.loss)
(use-package :nnl2.lli.ad.utils)

(nnlet ((nn (fnn 1 -> 1 :bias nil :init :kaiming/uniform)))
  (tlet ((x (from-flatten #(1) #(1 1) :requires-grad t))
         (y (from-flatten #(1) #(1 1) :requires-grad t)))

    (nnl2.optim:leto ((a (nnl2.optim:gd (parameters nn) :lr 0.01)))
      (dotimes (epochs 100)
        (tlet* ((prediction (forward nn x))
                (loss       (mse prediction y)))

          (when (zerop (mod epochs 10))
            (format t "Loss in ~d epoch: ~f~%" epochs (extract-scalar loss)))

          (backpropagation loss)

          (nnl2.optim:step! a)
          (nnl2.optim:zero-grad! a))))))
