(defpackage :nnl2.ffi
  (:use :cl)
  (:export :get-openblas0330woa64static-status :%empty :%zeros :%ones :%full :free-tensor :print-tensor
   :%dgemm :%sgemm :%gemm :get-tensor-rank :get-tensor-dtype :get-int-tensor-dtype :get-pointer-to-tensor-shape
   :%gemm! :%+= :%-= :%get-size :%get-size-in-bytes :%+ :%- :%*= :%/= :%* :%/ :%^= :%.^ :%.exp! :%.exp :%.log!
   :%.log :%tref :%tref-setter :%scale! :%scale :%zeros-like :%ones-like :%full-like :%.max! :%.min! :%.max :%.min
   :%.abs! :%.abs :get-tensor-data :%empty-like :%hstack :%vstack :%.relu! :%.relu :%.leaky-relu! :%.leaky-relu
   :%.sigmoid! :%.sigmoid :%.tanh! :%.tanh :%concat :%randn :%randn-like :%xavier :%transpose! :%transpose :%sum
   :%l2norm :%copy :%.+/incf! :%.+/incf :%.-/decf! :%.-/decf :%.*/mulf! :%.*/mulf :%.//divf! :%.//divf :%.^/powf! 
   :%.^/powf :%.max/maxf! :%.max/maxf :%.min/minf! :%.min/minf :%.+/broadcasting! :%.+/broadcasting :%.-/broadcasting!
   :%.-/broadcasting :%.*/broadcasting! :%.*/broadcasting :%.//broadcasting! :%.//broadcasting :%.^/broadcasting!
   :%.^/broadcasting :%.max/broadcasting :%.max/broadcasting! :%.min/broadcasting :%.min/broadcasting!
   :%make-tensor-from-flatten :%axpy/axpf! :%axpy/axpf :%axpy/broadcasting! :%axpy/broadcasting :%axpy :%axpy!
   :%set-abs-backend :%set-abs-inplace-backend :%set-inplace-fill-backend))
																					 
