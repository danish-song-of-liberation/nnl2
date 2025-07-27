(defpackage :nnl2.ffi
  (:use :cl)
  (:export :get-openblas0330woa64static-status :%empty :%zeros :%ones :%full :free-tensor :print-tensor
   :%dgemm :%sgemm :%gemm :get-tensor-rank :get-tensor-dtype :get-int-tensor-dtype :get-pointer-to-tensor-shape
   :%gemm! :%+= :%-= :%get-size :%get-size-in-bytes :%+ :%- :%*= :%/= :%* :%/ :%^= :%.^ :%.exp! :%.exp :%.log!
   :%.log :%tref :%tref-setter :%scale! :%scale :%zeros-like :%ones-like :%full-like :%.max! :%.min! :%.max :%.min
   :%.abs! :%.abs))
																					 
