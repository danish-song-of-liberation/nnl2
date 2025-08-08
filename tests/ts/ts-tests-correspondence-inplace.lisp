(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-correspondence-inplace.lisp
;; File: ts-tests-correspondence-inplace.lisp

;; Continuation of the ts-tests-correspondence.lisp file with 
;; approximately the same content but with in-place test implementations

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defparameter *default-.+/incf!-operation-shape* '(5 5))	
(defparameter *default-.-/decf!-operation-shape* '(5 5))	
(defparameter *default-.*/mulf!-operation-shape* '(5 5))	
(defparameter *default-.//divf!-operation-shape* '(5 5))	
(defparameter *default-.^/powf!-operation-shape* '(5 5))	
(defparameter *default-.max/maxf!-operation-shape* '(5 5))	
(defparameter *default-.min/minf!-operation-shape* '(5 5))	

;; -- `.+/incf!` tests section --

(fiveam:test nnl2.hli.ts/.+/incf!/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.+/incf!-operation-shape* :fill 1.0d0 :val 1.0d0 :expected 2.0d0 :op #'nnl2.hli.ts:+= :inplace t))									
  
(fiveam:test nnl2.hli.ts/.+/incf!/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.+/incf!-operation-shape* :fill 1.0s0 :val 1.0s0 :expected 2.0s0 :op #'nnl2.hli.ts:+= :inplace t))									
    
(fiveam:test nnl2.hli.ts/.+/incf!/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.+/incf!-operation-shape* :fill 1 :val 1 :expected 2 :op #'nnl2.hli.ts:+= :inplace t))									
      
;; -- `.-/decf!` tests section --	  

(fiveam:test nnl2.hli.ts/.-/decf!/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.-/decf!-operation-shape* :fill 2.0d0 :val 1.0d0 :expected 1.0d0 :op #'nnl2.hli.ts:-= :inplace t))									
  
(fiveam:test nnl2.hli.ts/.-/decf!/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.-/decf!-operation-shape* :fill 2.0s0 :val 1.0s0 :expected 1.0s0 :op #'nnl2.hli.ts:-= :inplace t))									
    
(fiveam:test nnl2.hli.ts/.-/decf!/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.-/decf!-operation-shape* :fill 2 :val 1 :expected 1 :op #'nnl2.hli.ts:-= :inplace t))									

;; -- `.*/mulf!` tests section --			  
		  
(fiveam:test nnl2.hli.ts/.*/mulf!/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.*/mulf!-operation-shape* :fill 4.0d0 :val 2.0d0 :expected 8.0d0 :op #'nnl2.hli.ts:*= :inplace t))									
  
(fiveam:test nnl2.hli.ts/.*/mulf!/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.*/mulf!-operation-shape* :fill 4.0s0 :val 2.0s0 :expected 8.0s0 :op #'nnl2.hli.ts:*= :inplace t))									
    
(fiveam:test nnl2.hli.ts/.*/mulf!/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.*/mulf!-operation-shape* :fill 4 :val 2 :expected 8 :op #'nnl2.hli.ts:*= :inplace t))									  
		        	  	  
;; -- `.//divf!` tests section --			  
		  
(fiveam:test nnl2.hli.ts/.//divf!/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.//divf!-operation-shape* :fill 4.0d0 :val 2.0d0 :expected 2.0d0 :op #'nnl2.hli.ts:/! :inplace t))									
  
(fiveam:test nnl2.hli.ts/.//divf!/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.//divf!-operation-shape* :fill 4.0s0 :val 2.0s0 :expected 2.0s0 :op #'nnl2.hli.ts:/! :inplace t))									
    
(fiveam:test nnl2.hli.ts/.//divf!/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.//divf!-operation-shape* :fill 4 :val 2 :expected 2 :op #'nnl2.hli.ts:/! :inplace t))									  		  
		  	  
;; -- `.^/powf!` tests section --			  
		  
(fiveam:test nnl2.hli.ts/.^/powf!/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.^/powf!-operation-shape* :fill 4.0d0 :val 3.0d0 :expected 64.0d0 :op #'nnl2.hli.ts:^= :inplace t))									
  
(fiveam:test nnl2.hli.ts/.^/powf!/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.^/powf!-operation-shape* :fill 4.0s0 :val 3.0s0 :expected 64.0s0 :op #'nnl2.hli.ts:^= :inplace t))									
    
(fiveam:test nnl2.hli.ts/.^/powf!/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.^/powf!-operation-shape* :fill 4 :val 3 :expected 64 :op #'nnl2.hli.ts:^= :inplace t))									  		  			  

;; -- `.max/maxf!` tests section --		
	
(fiveam:test nnl2.hli.ts/.max/maxf!/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.max/maxf!-operation-shape* :fill 4.0d0 :val 5.0d0 :expected 5.0d0 :op #'nnl2.hli.ts:.max! :inplace t))									
  
(fiveam:test nnl2.hli.ts/.max/maxf!/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.max/maxf!-operation-shape* :fill 4.0s0 :val 5.0s0 :expected 5.0s0 :op #'nnl2.hli.ts:.max! :inplace t))									
    
(fiveam:test nnl2.hli.ts/.max/maxf!/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.max/maxf!-operation-shape* :fill 4 :val 5 :expected 5 :op #'nnl2.hli.ts:.max! :inplace t))									  		  
	
;; -- `.min/minf!` tests section --		
	
(fiveam:test nnl2.hli.ts/.min/minf!/float64
  (check-nnl2.hli.ts/correspondence-operation :dtype :float64 :shape *default-.min/minf!-operation-shape* :fill 4.0d0 :val 5.0d0 :expected 4.0d0 :op #'nnl2.hli.ts:.min! :inplace t))									
  
(fiveam:test nnl2.hli.ts/.min/minf!/float32
  (check-nnl2.hli.ts/correspondence-operation :dtype :float32 :shape *default-.min/minf!-operation-shape* :fill 4.0s0 :val 5.0s0 :expected 4.0s0 :op #'nnl2.hli.ts:.min! :inplace t))									
    
(fiveam:test nnl2.hli.ts/.min/minf!/int32
  (check-nnl2.hli.ts/correspondence-operation :dtype :int32 :shape *default-.min/minf!-operation-shape* :fill 4 :val 5 :expected 4 :op #'nnl2.hli.ts:.min! :inplace t))									  		 				
		