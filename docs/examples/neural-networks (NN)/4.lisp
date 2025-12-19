(ql:quickload :nnl2)

(use-package :nnl2.hli.nn)

(defun compute-loss (nn input target)
  (nnl2.hli.ad:tlet ((forw (forward nn input)))
    (nnl2.hli.nn.ga.fitness:mse forw target)))

(defun ga-step (population input target &key delta mutate-rate crossover-rate)
  (let* ((elite (nnl2.hli.nn.ga.selection:elitist population #'(lambda (x) (compute-loss x input target))))
         (best (first elite))
         (elite-len (length elite)))

    (cons (copy best) (loop repeat (1- (length population))
                            collect (let* ((parent1 (nth (random elite-len) elite))
                                           (parent2 (nth (random elite-len) elite))
                                           (child (nnl2.hli.nn.ga.crossover:uniform parent1 parent2 :rate crossover-rate))
                                           (mutation (nnl2.hli.nn.ga.mutation:uniform child :rate mutate-rate :delta delta)))

                                      (free child)

                                      mutation)))))

(let ((population (loop repeat 10 collect (fnn 1 -> 1 :init :rand))))
  (nnl2.hli.ad:tlet ((input  (nnl2.hli.ad:make-tensor #2A((1) (2))))
                     (target (nnl2.hli.ad:make-tensor #2A((2) (4)))))

    (dotimes (generation 1000)
      (let ((old-population population))
        (setf population (ga-step population input target
                           :delta 0.01
                           :mutate-rate 0.2
                           :crossover-rate 0.5))

        (let* ((best-network (first population))
               (best-loss    (compute-loss best-network input target)))

          (format t "Generation = ~d; Loss = ~f~%" generation best-loss))

        (mapcar #'free old-population)))))
