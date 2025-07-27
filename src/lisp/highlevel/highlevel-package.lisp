(defpackage :nnl2.hli
  (:use :cl)
  (:export :make-foreign-pointer))
  
(defpackage :nnl2.hli.ts
  (:use :cl)
  (:shadow /=)
  (:export :make-shape-pntr :empty :empty-with-pntr :zeros :zeros-with-pntr :ones :ones-with-pntr :tlet
   :full :full-with-pntr :print-tensor :rank :dtype :int-dtype :shape-pointer :shape :.log! :.log
   :gemm :gemm! :+= :-= :size :size-in-bytes :.+ :.- :*= :/! :.* :./ :^= :.^ :.exp! :.exp :tref
   :scale! :scale :zeros-like :ones-like :full-like :.max! :.min! :.max :.min :.abs! :.abs
   :.map! :empty-like :.map))
  