(ql:quickload :nnl2)
(ql:quickload :magicl)

(use-package :nnl2.hli.ts)

(tlet ((a (ones #(5 5))))
  (let ((b (nnl2.convert:nnl2->magicl a)))
    (multiple-value-bind (foo bar baz) (magicl:svd b)
      (tlet ((foo2 (nnl2.convert:magicl->nnl2 foo))
             (bar2 (nnl2.convert:magicl->nnl2 bar))
             (baz2 (nnl2.convert:magicl->nnl2 baz)))

        (print-tensor foo2)
        (print-tensor bar2)
        (print-tensor baz2)

        (tlet* ((qux (gemm bar2 baz2))
                (quux (gemm foo2 qux)))

          (print-tensor quux))))))
		  