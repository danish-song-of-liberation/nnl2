(in-package :nnl2.hli.ts.tests)

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/empty (&key dtype int-dtype)
  "Checks that the `nnl2.hli.ts:empty` function creates a tensor with the specified properties"
  
  (let ((expected-size (* 4 4))
		(expected-shape '(4 4)))
		
    (nnl2.hli.ts:tlet ((tensor (nnl2.hli.ts:empty #(4 4) :dtype dtype)))
	
      (nnl2.hli.ts.tests:assert-tensor-properties 
	     :tensor tensor 
		 :expected-size expected-size
		 :expected-dtype dtype
		 :expected-shape expected-shape)
		 
	  (fiveam:is (= (nnl2.hli.ts:int-dtype tensor) int-dtype) "Checking if the data type matches the enum"))))

(fiveam:test nnl2.hli.ts/empty/float64
  (check-nnl2.hli.ts/empty :dtype :float64 :int-dtype nnl2.hli.ts.tests:+enum-float64+))
	  
(fiveam:test nnl2.hli.ts/empty/float32
  (check-nnl2.hli.ts/empty :dtype :float32 :int-dtype nnl2.hli.ts.tests:+enum-float32+))
  
(fiveam:test nnl2.hli.ts/empty/int32
  (check-nnl2.hli.ts/empty :dtype :int32 :int-dtype nnl2.hli.ts.tests:+enum-int32+))
    
  