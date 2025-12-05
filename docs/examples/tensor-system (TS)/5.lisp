(ql:quickload :nnl2)

(use-package :nnl2.hli.ts)

(tlet ((a (ones #(5 5))) (acc 0))
  (nnl2.lli.ts:iatd (i a) ;; alias for nnl2.lli.ts:iterate-across-tensor-data
    (incf acc))

  (print (= acc (size a))))
