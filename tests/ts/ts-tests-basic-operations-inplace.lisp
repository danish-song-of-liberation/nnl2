(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-basic-operations-inplace.lisp
;; File: ts-tests-basic-operations-inplace.lisp

;; The file is a continuation of the ts-tests-basic-operations.lisp file 
;; and does something similar, but with in-place operations (+= -=)

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defparameter *default-+=-operation-shape* '(5 5))
(defparameter *default--=-operation-shape* '(5 5))
(defparameter *default-*=-operation-shape* '(5 5))
(defparameter *default-/!-operation-shape* '(5 5))
(defparameter *default-^=-operation-shape* '(5 5))
(defparameter *default-.max!-operation-shape* '(5 5))
(defparameter *default-.min!-operation-shape* '(5 5))

;; -- `+=` tests section --		
	
(fiveam:test nnl2.hli.ts/+=/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-+=-operation-shape* :first-value 2.0d0 :second-value 1.0d0 :expected-value 3.0d0 :function #'nnl2.hli.ts:+= :inplace t))					
	
(fiveam:test nnl2.hli.ts/+=/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-+=-operation-shape* :first-value 2.0s0 :second-value 1.0s0 :expected-value 3.0s0 :function #'nnl2.hli.ts:+= :inplace t))					
						
(fiveam:test nnl2.hli.ts/+=/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-+=-operation-shape* :first-value 2 :second-value 1 :expected-value 3 :function #'nnl2.hli.ts:+= :inplace t))					
	
;; -- `-=` tests section --		
	
(fiveam:test nnl2.hli.ts/-=/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default--=-operation-shape* :first-value 2.0d0 :second-value 1.0d0 :expected-value 1.0d0 :function #'nnl2.hli.ts:-= :inplace t))					
	
(fiveam:test nnl2.hli.ts/-=/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default--=-operation-shape* :first-value 2.0s0 :second-value 1.0s0 :expected-value 1.0s0 :function #'nnl2.hli.ts:-= :inplace t))					
						
(fiveam:test nnl2.hli.ts/-=/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default--=-operation-shape* :first-value 2 :second-value 1 :expected-value 1 :function #'nnl2.hli.ts:-= :inplace t))					
		
;; -- `*=` tests section --				

(fiveam:test nnl2.hli.ts/*=/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-*=-operation-shape* :first-value 2.0d0 :second-value 4.0d0 :expected-value 8.0d0 :function #'nnl2.hli.ts:*= :inplace t))					
	
(fiveam:test nnl2.hli.ts/*=/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-*=-operation-shape* :first-value 2.0s0 :second-value 4.0s0 :expected-value 8.0s0 :function #'nnl2.hli.ts:*= :inplace t))					
						
(fiveam:test nnl2.hli.ts/*=/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-*=-operation-shape* :first-value 2 :second-value 4 :expected-value 8 :function #'nnl2.hli.ts:*= :inplace t))					
  
;; -- `/!` tests section (div in-place) --

(fiveam:test nnl2.hli.ts//!/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-/!-operation-shape* :first-value 8.0d0 :second-value 4.0d0 :expected-value 2.0d0 :function #'nnl2.hli.ts:/! :inplace t))					
	
(fiveam:test nnl2.hli.ts//!/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-/!-operation-shape* :first-value 8.0s0 :second-value 4.0s0 :expected-value 2.0s0 :function #'nnl2.hli.ts:/! :inplace t))					
						
(fiveam:test nnl2.hli.ts//!/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-/!-operation-shape* :first-value 8 :second-value 4 :expected-value 2 :function #'nnl2.hli.ts:/! :inplace t))			 
				
;; -- `^=` tests section (pow in-place) --

(fiveam:test nnl2.hli.ts/^=/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-^=-operation-shape* :first-value 4.0d0 :second-value 3.0d0 :expected-value 64.0d0 :function #'nnl2.hli.ts:^= :inplace t))					
	
(fiveam:test nnl2.hli.ts/^=/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-^=-operation-shape* :first-value 4.0s0 :second-value 3.0s0 :expected-value 64.0s0 :function #'nnl2.hli.ts:^= :inplace t))					
						
(fiveam:test nnl2.hli.ts/^=/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-^=-operation-shape* :first-value 4 :second-value 3 :expected-value 64 :function #'nnl2.hli.ts:^= :inplace t))			 
											
;; -- `.max!` tests section --

(fiveam:test nnl2.hli.ts/.max!/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-.max!-operation-shape* :first-value 5.0d0 :second-value 3.0d0 :expected-value 5.0d0 :function #'nnl2.hli.ts:.max! :inplace t))					
	
(fiveam:test nnl2.hli.ts/.max!/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-.max!-operation-shape* :first-value 5.0s0 :second-value 3.0s0 :expected-value 5.0s0 :function #'nnl2.hli.ts:.max! :inplace t))					
						
(fiveam:test nnl2.hli.ts/.max!/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-.max!-operation-shape* :first-value 5 :second-value 3 :expected-value 5 :function #'nnl2.hli.ts:.max! :inplace t))			 

;; -- `.min!` tests section --											
											
(fiveam:test nnl2.hli.ts/.min!/float64
  (check-nnl2.hli.ts/operation :dtype :float64 :shape *default-.min!-operation-shape* :first-value 5.0d0 :second-value 3.0d0 :expected-value 3.0d0 :function #'nnl2.hli.ts:.min! :inplace t))					
	
(fiveam:test nnl2.hli.ts/.min!/float32
  (check-nnl2.hli.ts/operation :dtype :float32 :shape *default-.min!-operation-shape* :first-value 5.0s0 :second-value 3.0s0 :expected-value 3.0s0 :function #'nnl2.hli.ts:.min! :inplace t))					
						
(fiveam:test nnl2.hli.ts/.min!/int32
  (check-nnl2.hli.ts/operation :dtype :int32 :shape *default-.min!-operation-shape* :first-value 5 :second-value 3 :expected-value 3 :function #'nnl2.hli.ts:.min! :inplace t))													
											
;; I'll repeat it just in case. 
;;
;; I don't consider this part of the code DRY because 
;; creating a separate, cumbersome macro for creating 
;; tests would be impractical and violate KISS, and 
;; the degree of violation of KISS as a principle would 
;; exceed DRY.											
											