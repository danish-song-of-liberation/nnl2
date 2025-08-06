(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-activation-functions.lisp
;; File: ts-tests-activation-functions.lisp

;; Tests activation functions

;; Note: 
;;   uses the function `check-nnl2.hli.ts/trivial-operation` from 
;;   the file `ts-tests-trivial-operations.lisp` (in the same folder)

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defparameter *default-.relu-operation-shape* '(5 5))
(defparameter *default-.leaky-relu-operation-shape* '(5 5))
(defparameter *default-.sigmoid-operation-shape* '(5 5))
(defparameter *default-.tanh-operation-shape* '(5 5))

;; -- `.relu` tests section --		

(fiveam:test nnl2.hli.ts/.relu/float64/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.relu-operation-shape* :val -1.0d0 :expected 0.0d0 :op #'nnl2.hli.ts:.relu))					
	
(fiveam:test nnl2.hli.ts/.relu/float64/2
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.relu-operation-shape* :val 1.0d0 :expected 1.0d0 :op #'nnl2.hli.ts:.relu))						
	
(fiveam:test nnl2.hli.ts/.relu/float32/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.relu-operation-shape* :val -1.0s0 :expected 0.0s0 :op #'nnl2.hli.ts:.relu))	

(fiveam:test nnl2.hli.ts/.relu/float32/2
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.relu-operation-shape* :val 1.0s0 :expected 1.0s0 :op #'nnl2.hli.ts:.relu))	  

(fiveam:test nnl2.hli.ts/.relu/int32/1
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.relu-operation-shape* :val -1 :expected 0 :op #'nnl2.hli.ts:.relu))			
  
(fiveam:test nnl2.hli.ts/.relu/int32/2
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.relu-operation-shape* :val 1 :expected 1 :op #'nnl2.hli.ts:.relu))	
  
;; -- `.leaky-relu` tests section --		
	
(fiveam:test nnl2.hli.ts/.leaky-relu/float64/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.leaky-relu-operation-shape* :val -1.0d0 :expected (coerce -0.01 'double-float) :op #'nnl2.hli.ts:.leaky-relu :tolerance 0.001))					
	
(fiveam:test nnl2.hli.ts/.leaky-relu/float64/2
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.leaky-relu-operation-shape* :val 1.0d0 :expected 1.0d0 :op #'nnl2.hli.ts:.leaky-relu))						
	
(fiveam:test nnl2.hli.ts/.leaky-relu/float32/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.leaky-relu-operation-shape* :val -1.00 :expected -0.01 :op #'nnl2.hli.ts:.leaky-relu :tolerance 0.001))	

(fiveam:test nnl2.hli.ts/.leaky-relu/float32/2
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.leaky-relu-operation-shape* :val 1.0s0 :expected 1.0s0 :op #'nnl2.hli.ts:.leaky-relu))	  

(fiveam:test nnl2.hli.ts/.leaky-relu/int32/1
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.leaky-relu-operation-shape* :val -1 :expected (coerce -0.01 'double-float) :op #'nnl2.hli.ts:.leaky-relu :tolerance 0.001))			
  
(fiveam:test nnl2.hli.ts/.leaky-relu/int32/2
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.leaky-relu-operation-shape* :val -100 :expected -1 :op #'nnl2.hli.ts:.leaky-relu))	
    
(fiveam:test nnl2.hli.ts/.leaky-relu/int32/3
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.leaky-relu-operation-shape* :val 1 :expected 1 :op #'nnl2.hli.ts:.leaky-relu))	
 
;; -- `.sigmoid` tests section --

(fiveam:test nnl2.hli.ts/.sigmoid/float64/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.sigmoid-operation-shape* :val 0.7d0 :expected 0.6681877721681662d0 :op #'nnl2.hli.ts:.sigmoid :tolerance 0.00001))					 
 
(fiveam:test nnl2.hli.ts/.sigmoid/float64/2
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.sigmoid-operation-shape* :val 0.0d0 :expected 0.50d0 :op #'nnl2.hli.ts:.sigmoid :tolerance 0.0001)) 
 
(fiveam:test nnl2.hli.ts/.sigmoid/float32/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.sigmoid-operation-shape* :val 0.7s0 :expected 0.66818774s0 :op #'nnl2.hli.ts:.sigmoid :tolerance 0.001))					 
 
(fiveam:test nnl2.hli.ts/.sigmoid/float32/2
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.sigmoid-operation-shape* :val 0.0s0 :expected 0.50s0 :op #'nnl2.hli.ts:.sigmoid :tolerance 0.01)) 
  
(fiveam:test nnl2.hli.ts/.sigmoid/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.sigmoid-operation-shape* :val 1 :expected 0.7310585786300049d0 :op #'nnl2.hli.ts:.sigmoid :tolerance 0.0001))	  

;; -- `.tanh` tests section --

(fiveam:test nnl2.hli.ts/.tanh/float32/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.tanh-operation-shape* :val 0.3s0 :expected 0.29131s0 :op #'nnl2.hli.ts:.tanh :tolerance 0.01))				 
 
(fiveam:test nnl2.hli.ts/.tanh/float64/2
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.tanh-operation-shape* :val 0.0d0 :expected 0.0d0 :op #'nnl2.hli.ts:.tanh))					 
  
(fiveam:test nnl2.hli.ts/.tanh/float32/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.tanh-operation-shape* :val 0.3s0 :expected 0.29131s0 :op #'nnl2.hli.ts:.tanh :tolerance 0.01))

(fiveam:test nnl2.hli.ts/.tanh/float32/1
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.tanh-operation-shape* :val 0.0s0 :expected 0.0s0 :op #'nnl2.hli.ts:.tanh))  
  
(fiveam:test nnl2.hli.ts/.tanh/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.tanh-operation-shape* :val 1 :expected 0.7615941559557649d0 :op #'nnl2.hli.ts:.tanh :tolerance 0.001))				 
   
;; the continuation of the file is `ts-tests-activation-functions-inplace.lisp` 
;; with the implementation of activation function tests in place   
   