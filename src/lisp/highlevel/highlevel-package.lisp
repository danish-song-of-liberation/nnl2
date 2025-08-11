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
   :.map! :empty-like :.map :hstack :vstack :tlet* :.relu! :.relu :.leaky-relu! :.leaky-relu
   :.sigmoid! :.sigmoid :.tanh! :.tanh :concat :randn :randn-like :xavier :transpose! :transpose
   :sum :norm :copy :.+/incf! :.+/incf :.-/decf! :.-/decf :.+/gnrl :.+/gnrl! :.-/gnrl :.-/gnrl!
   :.*/mulf! :.*/mulf :.//divf! :.//divf :.*/gnrl! :.*/gnrl :.//gnrl! :.//gnrl :.^/powf! :.^/powf
   :.^/gnrl! :.^/gnrl :.max/maxf! :.max/gnrl! :.max/maxf :.max/gnrl :.min/minf! :.min/gnrl!
   :.min/minf :.min/gnrl :.+/broadcasting! :.+/broadcasting :.-/broadcasting :.-/broadcasting!
   :.*/broadcasting! :.*/broadcasting :.//broadcasting! :.//broadcasting :.^/broadcasting!
   :.^/broadcasting :*nnl2-tensor-types* :ts-type-to-lisp :lisp-type-to-ts :from-flatten
   :make-tensor :/map :/map! :axpy :axpy!))
  