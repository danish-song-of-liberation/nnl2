(defpackage :nnl2.hli
  (:use :cl)
  (:export :make-foreign-pointer))
  
(defpackage :nnl2.hli.ts
  (:use :cl)
  (:shadow /=)
  (:export :make-shape-pntr :empty :empty-with-pntr :zeros :zeros-with-pntr :ones :ones-with-pntr :tlet
   :full :full-with-pntr :print-tensor :rank :dtype :int-dtype :shape-pointer :shape
   :gemm :gemm! :+= :-= :size :size-in-bytes :.+ :.- :*= :/= :.* :./ :^= :.^ :.exp! :.exp))
  