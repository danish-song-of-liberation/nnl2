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
   :make-tensor :/map :/map! :axpy :axpy! :use-backend :use-backend/abs :use-backend/inplace-fill
   :use-backend/+= :use-backend/-= :use-backend/*= :use-backend//! :use-backend/^= 
   :use-backend/.log! :use-backend/.max! :use-backend/.min! :use-backend/scale! 
   :use-backend/.+ :use-backend/.- :use-backend/.* :use-backend/./ :use-backend/.^ 
   :use-backend/.log :use-backend/.min :use-backend/.max :use-backend/scale :use-backend/gemm
   :use-backend/empty :use-backend/.abs :use-backend/xavier :use-backend/randn 
   :use-backend/full :use-backend/zeros :use-backend/sum :use-backend/l2norm 
   :use-backend/copy :use-backend/.relu :use-backend/.relu! :use-backend/.leaky-relu 
   :use-backend/.leaky-relu! :use-backend/.sigmoid :use-backend/.sigmoid! 
   :use-backend/.tanh :use-backend/.tanh! :use-backend/transpose :use-backend/transpose!
   :get-backend/empty :get-backend/zeros :get-backend/ones :get-backend/full :get-backend/gemm
   :get-backend/gemm! :get-backend/+= :get-backend/-= :get-backend/.+ :get-backend/.-
   :get-backend/*= :get-backend//! :get-backend/.* :get-backend/./ :get-backend/^=
   :get-backend/.^ :get-backend/.exp! :get-backend/.exp :get-backend/.log! :get-backend/.log
   :get-backend/scale! :get-backend/scale :get-backend/.max! :get-backend/.max :get-backend/.min!
   :get-backend/.min :get-backend/.abs! :get-backend/.abs :get-backend/hstack :get-backend/vstack
   :get-backend/.relu! :get-backend/.relu :get-backend/.leaky-relu! :get-backend/.leaky-relu
   :get-backend/.sigmoid :get-backend/.sigmoid! :get-backend/.tanh :get-backend/.tanh!
   :get-backend/concat :get-backend/randn :get-backend/xavier :get-backend/transpose
   :get-backend/transpose! :get-backend/sum :get-backend/norm :get-backend/copy
   :get-backend/axpy! :get-backend/axpy :get-backends/empty :get-backends/zeros
   :get-backends/full :get-backends/ones :get-backends/gemm :get-backend/gemm!
   :get-backends/+= :get-backends/-= :get-backends/.+ :get-backends/.- :get-backends/*=
   :get-backends//! :get-backends/.* :get-backends/./ :get-backends/^= :get-backends/.^
   :get-backends/.exp! :get-backends/.exp :get-backends/.log! :get-backends/.log
   :get-backends/scale :get-backends/scale! :get-backends/.max! :get-backends/.min!
   :get-backends/.max :get-backends/.min :get-backends/.abs :get-backends/.abs!
   :get-backends/hstack :get-backends/vstack :get-backends/.relu :get-backends/.relu!
   :get-backends/.leaky-relu :get-backends/.leaky-relu! :get-backends/.sigmoid
   :get-backends/.sigmoid! :get-backends/.tanh :get-backends/.tanh! :get-backends/concat
   :get-backends/randn :get-backends/xavier :get-backends/transpose! :get-backends/transpose
   :get-backends/sum :get-backends/norm :get-backends/copy :get-backends/axpy! :get-backends/axpy))
   