(ql:quickload :nnl2)

(use-package :nnl2.hli.ad)
	
(defun ga-step (population targ &key delta mutate-rate elite-count)
  (let* ((elite (nnl2.hli.nn.ga.selection:elitist population #'(lambda (x) (nnl2.hli.nn.ga.fitness:mse x targ))))
         (best (first elite))
         (elite-len (length elite)))

    (cons best (loop repeat (1- (length population))
                     collect (let* ((parent1 (nth (random elite-len) elite))
                                    (parent2 (nth (random elite-len) elite))
                                    (child (nnl2.hli.nn.ga.crossover:uniform parent1 parent2))
                                    (mutation (nnl2.hli.nn.ga.mutation:uniform child :rate mutate-rate :delta delta)))

                               (nnl2.hli.ad:free child)

                               mutation)))))

(defun main (&key (generations 10) (generation 0) new-population target (ga-population 10) (init-fn #'nnl2.hli.ad:rand) (delta 0.5) (elite-count 3) (mutate-rate 0.1) (init-size 10))
  (let ((population (if new-population new-population (loop repeat ga-population collect (funcall init-fn (vector init-size)))))
        (target (if target target (nnl2.hli.ad:make-tensor #(0 1 2 3 4 5 6 7 8 9)))))

    (when new-population (nnl2.gc:gc))

    (apply #'nnl2.gc:push population)

    (let* ((best (first population))
           (best-loss (nnl2.hli.nn.ga.fitness:mse best target)))

      (format t "Generation ~d; Best loss (0 - bad, 1 - good): ~a~%" generation best-loss)

      (when (and (not (zerop generation)) (zerop (mod generation generations)))
        (format t "~%~%Ready! Final tensor (must be 0, 1, 2, .., 8, 9):~%")
        (nnl2.hli.ad:print-data best))

      (when (< generation generations)
        (main  ;; Tail recursion
          :new-population (ga-step population target :delta delta :mutate-rate mutate-rate :elite-count elite-count)
          :target target
          :generations generations
          :generation (1+ generation)
          :ga-population ga-population
          :init-fn init-fn
          :delta delta
          :init-size init-size
          :elite-count elite-count
          :mutate-rate mutate-rate)))))
		  
(main
  :generations 200  ;; Epochs in GA
  :ga-population 10
  :init-fn #'nnl2.hli.ad:rand
  :init-size 10
  :delta 0.5
  :elite-count 3
  :mutate-rate 0.1)
