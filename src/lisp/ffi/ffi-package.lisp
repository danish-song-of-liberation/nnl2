(defpackage :nnl2.ffi
  (:use :cl)
  (:export 
   :get-openblas0330woa64static-status :%empty :%zeros :%ones :%full :free-tensor :print-tensor
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
   :%set-abs-backend :%set-abs-inplace-backend :%set-inplace-fill-backend :%set-empty-backend :%set-zeros-backend
   :%set-ones-backend :%set-sgemminplace-backend :%set-dgemminplace-backend :%set-addinplace-backend 
   :%set-subinplace-backend :%set-add-backend :%set-sub-backend :%set-mulinplace-backend :%set-divinplace-backend
   :%set-mul-backend :%set-div-backend :%set-powinplace-backend :%set-expinplace-backend :%set-pow-backend
   :%set-exp-backend :%set-loginplace-backend :%set-log-backend :%set-scaleinplace-backend :%set-scale-backend
   :%set-maxinplace-backend :%set-mininplace-backend :%set-max-backend :%set-min-backend :%set-hstack-backend
   :%set-vstack-backend :%set-reluinplace-backend :%set-relu-backend :%set-leakyreluinplace-backend
   :%set-leakyrelu-backend :%set-sigmoidinplace-backend :%set-sigmoid-backend :%set-tanhinplace-backend
   :%set-tanh-backend :%set-concat-backend :%set-randn-backend :%set-xavier-backend :%set-transposeinplace-backend
   :%set-transpose-backend :%set-sum-backend :%set-l2norm-backend :%set-copy-backend :%set-add-incf-inplace-backend
   :%set-add-incf-backend :%set-sub-decf-inplace-backend :%set-sub-decf-backend :%set-mul-mulf-inplace-backend
   :%set-mul-mulf-backend :%set-div-divf-inplace-backend :%set-div-divf-backend :%set-pow-powf-inplace-backend
   :%set-pow-powf-backend :%set-max-maxf-inplace-backend :%set-max-maxf-backend :%set-min-minf-inplace-backend
   :%set-min-minf-backend :%set-add-broadcasting-inplace-backend :%set-add-broadcasting-backend
   :%set-sub-broadcasting-inplace-backend :%set-sub-broadcasting-backend :%set-mul-broadcasting-inplace-backend
   :%set-mul-broadcasting-backend :%set-div-broadcasting-inplace-backend :%set-div-broadcasting-backend
   :%set-pow-broadcasting-inplace-backend :%set-pow-broadcasting-backend :%set-max-broadcasting-inplace-backend
   :%set-min-broadcasting-backend :%set-fill-tensor-with-data-backend :%set-axpy-inplace-backend
   :%set-axpy-backend :%set-axpf-inplace-backend :%set-axpf-backend :%set-axpy-broadcasting-inplace-backend
   :%set-axpy-broadcasting-backend)) 
