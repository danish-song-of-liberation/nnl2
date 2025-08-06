(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-trivial-operations-inplace.lisp
;; File: ts-tests-trivial-operations-inplace.lisp

;; Continuation of the code with trivial operations, only this time in-place

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defparameter *default-.exp!-operation-shape* '(5 5))
(defparameter *default-.log!-operation-shape* '(5 5))
(defparameter *default-.abs!-operation-shape* '(5 5))

;; -- `.exp!` tests section --		
	
(fiveam:test nnl2.hli.ts/.exp!/float64
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.exp!-operation-shape* :val 2.0d0 :expected 7.38d9 :op #'nnl2.hli.ts:.exp! :inplace t))					
	
(fiveam:test nnl2.hli.ts/.exp!/float32
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.exp!-operation-shape* :val 2.0s0 :expected 7.389 :op #'nnl2.hli.ts:.exp! :inplace t))			

(fiveam:test nnl2.hli.ts/.exp!/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.exp!-operation-shape* :val 0 :expected 1 :op #'nnl2.hli.ts:.exp! :inplace t))			

;; -- `.log!` tests section --

(fiveam:test nnl2.hli.ts/.log!/float64
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.exp-operation-shape* :val 3.0d0 :expected 1.0986122886681098d0 :op #'nnl2.hli.ts:.log! :inplace t))					
	
(fiveam:test nnl2.hli.ts/.log!/float32
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.exp-operation-shape* :val 3.0s0 :expected 1.0986123 :op #'nnl2.hli.ts:.log! :inplace t))			

(fiveam:test nnl2.hli.ts/.log!/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.exp-operation-shape* :val 3 :expected 1.0d98 :op #'nnl2.hli.ts:.log! :inplace t))		

;; -- `.abs!` tests section --

(fiveam:test nnl2.hli.ts/.abs!/float64
  (check-nnl2.hli.ts/trivial-operation :dtype :float64 :shape *default-.abs!-operation-shape* :val -1.0d0 :expected 1.0d0 :op #'nnl2.hli.ts:.abs! :inplace t))					
	
(fiveam:test nnl2.hli.ts/.abs!/float32
  (check-nnl2.hli.ts/trivial-operation :dtype :float32 :shape *default-.abs!-operation-shape* :val -1.0s0 :expected 1.0s0 :op #'nnl2.hli.ts:.abs! :inplace t))			

(fiveam:test nnl2.hli.ts/.abs!/int32
  (check-nnl2.hli.ts/trivial-operation :dtype :int32 :shape *default-.abs!-operation-shape* :val -1 :expected 1 :op #'nnl2.hli.ts:.abs! :inplace t))		
