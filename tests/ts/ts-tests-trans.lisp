(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-trans.lisp
;; File: ts-tests-trans.lisp

;; Contains tests with transposition and pseudo in-place transposition

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defun check-nnl2.hli.ts/transpose (&key dtype)
  "A simple transpose test"
  
  (nnl2.tests.utils:make-test dtype "Transpose"
    (let* ((lst '((1 2 3) (4 5 6) (7 8 9)))
		   (trans-lst (apply #'mapcar #'list lst))
		   (lisp-type (nnl2.hli.ts:ts-type-to-lisp dtype)))
      
	  (nnl2.hli.ts:tlet* ((tensor (nnl2.hli.ts:make-tensor lst :dtype dtype))
						  (trans-tensor (nnl2.hli.ts:transpose tensor)))
						  
	    (dotimes (i (length lst))
		  (dotimes (j (length (first lst)))
		    (fiveam:is (= (nnl2.hli.ts:tref trans-tensor i j) (coerce (nth i (nth j lst)) lisp-type)))))))))	
	
(defun check-nnl2.hli.ts/transpose! (&key dtype)
  "A simple transpose! (transpose pseudo-in-place) test"
  
  (nnl2.tests.utils:make-test dtype "Transpose pseudo-in-place"
    (let* ((lst '((1 2 3) (4 5 6) (7 8 9)))
		   (trans-lst (apply #'mapcar #'list lst))
		   (lisp-type (nnl2.hli.ts:ts-type-to-lisp dtype)))
      
	  (nnl2.hli.ts:tlet* ((tensor (nnl2.hli.ts:make-tensor lst :dtype dtype)))
	  
		(nnl2.hli.ts:transpose! tensor)
						  
	    (dotimes (i (length lst))
		  (dotimes (j (length (first lst)))
		    (fiveam:is (= (nnl2.hli.ts:tref tensor i j) (coerce (nth i (nth j lst)) lisp-type)))))))))		
	
;; -- `transpose` tests section --	
  
(fiveam:test nnl2.hli.ts/transpose/float64
  (check-nnl2.hli.ts/transpose :dtype :float64)) 
	  
(fiveam:test nnl2.hli.ts/transpose/float32
  (check-nnl2.hli.ts/transpose :dtype :float32)) 
	  
(fiveam:test nnl2.hli.ts/transpose/int32
  (check-nnl2.hli.ts/transpose :dtype :int32)) 
	  	  
;; -- `transpose! (transposition pseudo-in-place)` tests section --	
  
(fiveam:test nnl2.hli.ts/transpose!/float64
  (check-nnl2.hli.ts/transpose! :dtype :float64)) 
	  
(fiveam:test nnl2.hli.ts/transpose!/float32
  (check-nnl2.hli.ts/transpose! :dtype :float32)) 
	  
(fiveam:test nnl2.hli.ts/transpose!/int32
  (check-nnl2.hli.ts/transpose! :dtype :int32)) 
	  	  