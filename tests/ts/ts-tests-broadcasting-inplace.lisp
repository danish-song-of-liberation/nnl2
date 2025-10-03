(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-broadcasting-inplace.lisp
;; File: ts-tests-broadcasting-inplace.lisp

;; This file is a continuation of the file `ts-tests-broadcasting.lisp` 
;; (in the same folder), but it implements in-place operations

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defparameter *default-.+/broadcasting!/a-operation-shape* '(4 4 4) "Shape for first (a) tensor in broadcasting in-place")
(defparameter *default-.+/broadcasting!/b-operation-shape* '(4) "Shape for second (b) tensor in broadcasting in-place")	  
  
(defparameter *default-.-/broadcasting!/a-operation-shape* '(4 4) "Shape for first (a) tensor in broadcasting in-place")
(defparameter *default-.-/broadcasting!/b-operation-shape* '(4) "Shape for second (b) tensor in broadcasting in-place")	  

(defparameter *default-.*/broadcasting!/a-operation-shape* '(3 3 2) "Shape for first (a) tensor in broadcasting in-place")
(defparameter *default-.*/broadcasting!/b-operation-shape* '(3 1) "Shape for second (b) tensor in broadcasting in-place")	  

(defparameter *default-.//broadcasting!/a-operation-shape* '(3 3) "Shape for first (a) tensor in broadcasting in-place")
(defparameter *default-.//broadcasting!/b-operation-shape* '(3) "Shape for second (b) tensor in broadcasting in-place")	  

(defparameter *default-.^/broadcasting!/a-operation-shape* '(3 3) "Shape for first (a) tensor in broadcasting in-place")
(defparameter *default-.^/broadcasting!/b-operation-shape* '(3) "Shape for second (b) tensor in broadcasting in-place")	  

(defparameter *default-.max/broadcasting!/a-operation-shape* '(3 4) "Shape for first (a) tensor in broadcasting in-place")
(defparameter *default-.max/broadcasting!/b-operation-shape* '(3) "Shape for second (b) tensor in broadcasting in-place")	  

(defparameter *default-.min/broadcasting!/a-operation-shape* '(3 4) "Shape for first (a) tensor in broadcasting in-place")
(defparameter *default-.min/broadcasting!/b-operation-shape* '(4) "Shape for second (b) tensor in broadcasting in-place")	  

(defparameter *default-axpy/broadcasting!/a-operation-shape* '(3 4) "Shape for first (a) tensor in broadcasting in-place")
(defparameter *default-axpy/broadcasting!/b-operation-shape* '(4) "Shape for second (b) tensor in broadcasting in-place")	  

;; -- `.+/broadcasting!` tests section --

(fiveam:test nnl2.hli.ts/.+/broadcasting!/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.+/broadcasting!/a-operation-shape* :shape-2 *default-.+/broadcasting!/b-operation-shape* :fill-1 2.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:+= :expected-val 5.0d0 :inplace t))									
  
(fiveam:test nnl2.hli.ts/.+/broadcasting!/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.+/broadcasting!/a-operation-shape* :shape-2 *default-.+/broadcasting!/b-operation-shape* :fill-1 2.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:+= :expected-val 5.0s0 :inplace t))									
    
(fiveam:test nnl2.hli.ts/.+/broadcasting!/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.+/broadcasting!/a-operation-shape* :shape-2 *default-.+/broadcasting!/b-operation-shape* :fill-1 2 :fill-2 3 :op #'nnl2.hli.ts:+= :expected-val 5 :inplace t))									
    		
;; -- `.-/broadcasting!` tests section --

(fiveam:test nnl2.hli.ts/.-/broadcasting!/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.-/broadcasting!/a-operation-shape* :shape-2 *default-.-/broadcasting!/b-operation-shape* :fill-1 2.0d0 :fill-2 1.0d0 :op #'nnl2.hli.ts:-= :expected-val 1.0d0 :inplace t))									
  
(fiveam:test nnl2.hli.ts/.-/broadcasting!/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.-/broadcasting!/a-operation-shape* :shape-2 *default-.-/broadcasting!/b-operation-shape* :fill-1 2.0s0 :fill-2 1.0s0 :op #'nnl2.hli.ts:-= :expected-val 1.0s0 :inplace t))									
    
(fiveam:test nnl2.hli.ts/.-/broadcasting!/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.-/broadcasting!/a-operation-shape* :shape-2 *default-.-/broadcasting!/b-operation-shape* :fill-1 2 :fill-2 1 :op #'nnl2.hli.ts:-= :expected-val 1 :inplace t))									
    			
;; -- `.*/broadcasting!` tests section --

(fiveam:test nnl2.hli.ts/.*/broadcasting!/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.*/broadcasting!/a-operation-shape* :shape-2 *default-.*/broadcasting!/b-operation-shape* :fill-1 2.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:*= :expected-val 6.0d0 :inplace t))									
  
(fiveam:test nnl2.hli.ts/.*/broadcasting!/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.*/broadcasting!/a-operation-shape* :shape-2 *default-.*/broadcasting!/b-operation-shape* :fill-1 2.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:*= :expected-val 6.0s0 :inplace t))									
    
(fiveam:test nnl2.hli.ts/.*/broadcasting!/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.*/broadcasting!/a-operation-shape* :shape-2 *default-.*/broadcasting!/b-operation-shape* :fill-1 2 :fill-2 3 :op #'nnl2.hli.ts:*= :expected-val 6 :inplace t))													
								
;; -- `.//broadcasting!` tests section --

(fiveam:test nnl2.hli.ts/.//broadcasting!/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.//broadcasting!/a-operation-shape* :shape-2 *default-.//broadcasting!/b-operation-shape* :fill-1 15.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:/! :expected-val 5.0d0 :inplace t))									
  
(fiveam:test nnl2.hli.ts/.//broadcasting!/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.//broadcasting!/a-operation-shape* :shape-2 *default-.//broadcasting!/b-operation-shape* :fill-1 15.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:/! :expected-val 5.0s0 :inplace t))									
    
(fiveam:test nnl2.hli.ts/.//broadcasting!/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.//broadcasting!/a-operation-shape* :shape-2 *default-.//broadcasting!/b-operation-shape* :fill-1 15 :fill-2 3 :op #'nnl2.hli.ts:/! :expected-val 5 :inplace t))													
								
;; -- `.^/broadcasting!` tests section --

(fiveam:test nnl2.hli.ts/.^/broadcasting!/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.^/broadcasting!/a-operation-shape* :shape-2 *default-.^/broadcasting!/b-operation-shape* :fill-1 4.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:^= :expected-val 64.0d0 :inplace t))									
  
(fiveam:test nnl2.hli.ts/.^/broadcasting!/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.^/broadcasting!/a-operation-shape* :shape-2 *default-.^/broadcasting!/b-operation-shape* :fill-1 4.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:^= :expected-val 64.0s0 :inplace t))									
    
(fiveam:test nnl2.hli.ts/.^/broadcasting!/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.^/broadcasting!/a-operation-shape* :shape-2 *default-.^/broadcasting!/b-operation-shape* :fill-1 4 :fill-2 3 :op #'nnl2.hli.ts:^= :expected-val 64 :inplace t))									
    															
;; -- `.max/broadcasting!` tests section --	

(fiveam:test nnl2.hli.ts/.max/broadcasting!/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.max/broadcasting!/a-operation-shape* :shape-2 *default-.max/broadcasting!/b-operation-shape* :fill-1 4.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:.max! :expected-val 4.0d0 :inplace t))									
  
(fiveam:test nnl2.hli.ts/.max/broadcasting!/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.max/broadcasting!/a-operation-shape* :shape-2 *default-.max/broadcasting!/b-operation-shape* :fill-1 4.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:.max! :expected-val 4.0s0 :inplace t))									
    
(fiveam:test nnl2.hli.ts/.max/broadcasting!/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.max/broadcasting!/a-operation-shape* :shape-2 *default-.max/broadcasting!/b-operation-shape* :fill-1 4 :fill-2 3 :op #'nnl2.hli.ts:.max! :expected-val 4 :inplace t))									
    							
;; -- `.min/broadcasting!` tests section --							
	
(fiveam:test nnl2.hli.ts/.min/broadcasting!/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-.min/broadcasting!/a-operation-shape* :shape-2 *default-.min/broadcasting!/b-operation-shape* :fill-1 4.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:.min! :expected-val 3.0d0 :inplace t))									
  
(fiveam:test nnl2.hli.ts/.min/broadcasting!/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-.min/broadcasting!/a-operation-shape* :shape-2 *default-.min/broadcasting!/b-operation-shape* :fill-1 4.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:.min! :expected-val 3.0s0 :inplace t))									
    
(fiveam:test nnl2.hli.ts/.min/broadcasting!/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-.min/broadcasting!/a-operation-shape* :shape-2 *default-.min/broadcasting!/b-operation-shape* :fill-1 4 :fill-2 3 :op #'nnl2.hli.ts:.min! :expected-val 3 :inplace t))									
    																			
;; -- `axpy/broadcasting!` tests section --																							
																				
(fiveam:test nnl2.hli.ts/axpy/broadcasting!/float64
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float64 :shape-1 *default-axpy/broadcasting!/a-operation-shape* :shape-2 *default-axpy/broadcasting!/b-operation-shape* :fill-1 4.0d0 :fill-2 3.0d0 :op #'nnl2.hli.ts:axpy! :expected-val 7.0d0 :inplace t))									
  
(fiveam:test nnl2.hli.ts/axpy/broadcasting!/float32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :float32 :shape-1 *default-axpy/broadcasting!/a-operation-shape* :shape-2 *default-axpy/broadcasting!/b-operation-shape* :fill-1 4.0s0 :fill-2 3.0s0 :op #'nnl2.hli.ts:axpy! :expected-val 7.0s0 :inplace t))									
    
(fiveam:test nnl2.hli.ts/axpy/broadcasting!/int32
  (check-nnl2.hli.ts/broadcasting-operation :dtype :int32 :shape-1 *default-axpy/broadcasting!/a-operation-shape* :shape-2 *default-axpy/broadcasting!/b-operation-shape* :fill-1 4 :fill-2 3 :op #'nnl2.hli.ts:axpy! :expected-val 7 :inplace t))																													
																				