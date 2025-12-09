(ql:quickload :nnl2)

;; Train a neural network to add numbers

(nnl2.hli.nn:nnlet ((a (nnl2.hli.nn:sequential
                         (nnl2.hli.nn:fnn 2 -> 64 :init :xavier/normal)
                         (nnl2.hli.nn:.tanh :approx t)
                         (nnl2.hli.nn:fnn 64 -> 64 :init :xavier/normal)
                         (nnl2.hli.nn:.sigmoid :approx t)
                         (nnl2.hli.nn:fnn 64 -> 1 :init #'(lambda (x) (nnl2.hli.ts:uniform! (nnl2.hli.ad:data x) :from -1.0 :to 1.0))))))

  (nnl2.optim:leto ((optim (nnl2.optim:gd (nnl2.hli.nn:parameters a) :lr 0.0025)))
    (nnl2.hli.ad:tlet ((x (nnl2.hli.ad:make-tensor #2A((1 1) (2 2) (3 5) (5 5) (2 1) (0 0) (7 7) (4 4))))
                       (y (nnl2.hli.ad:make-tensor #2A((2) (4) (8) (10) (3) (0) (14) (8)) :requires-grad t)))

      (dotimes (epochs 5000)
        (nnl2.hli.ad:tlet* ((forward (nnl2.hli.nn:forward a x))
                            (err (nnl2.hli.ad.r.loss:mse forward y)))

          (nnl2.hli.ad:backpropagation err)

          (when (zerop (mod epochs 10))
            (format t "Epoch ~d. Loss: ~f~%" epochs (nnl2.lli.ad.utils:extract-scalar err)))

          (when (zerop (mod epochs 1000))
            (incf (nnl2.optim:lr optim) 0.0001))

          (nnl2.optim:step! optim)
          (nnl2.optim:zero-grad! optim)))))

  (terpri)

  (nnl2.hli.ad:tlet* ((test (nnl2.hli.ad:make-tensor #2A((1 2) (3 3) (5 5) (6 6))))
                      (forw (nnl2.hli.nn:forward a test)))

    (nnl2.hli.ad:print-data forw)))
	