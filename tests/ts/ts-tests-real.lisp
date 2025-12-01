(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-real.lisp
;; File: ts-tests-real.lisp

;; File contains tests with real tensor operation that will be used 
;; by the user, rather than repetitive tests of a single function

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(fiveam:test nnl2.hli.ts-real-test/1
  (nnl2.tests.utils:make-test :float64 "Simple Gradient Descent"
  
    (nnl2.hli.ts:tlet ((target (nnl2.hli.ts:make-tensor #2A((1 0 1) (0 0 1) (1 0 1))))
					   (a (nnl2.hli.ts:uniform #(3 3))) (b (nnl2.hli.ts:uniform #(3 3))))
					   
	  (flet ((d/mse (x y) (nnl2.hli.ts:.* (nnl2.hli.ts:.- x y) 2))
			 (forward (x y) (nnl2.hli.ts:gemm a b)))
			 
		(let ((lr 0.1))	 
		  (dotimes (epochs 100)
		    (nnl2.hli.ts:tlet* ((fp (forward a b))
							    (loss (d/mse fp target))
							    (dl/da (nnl2.hli.ts:gemm loss b :transb :nnl2trans))
							    (dl/db (nnl2.hli.ts:gemm a loss :transa :nnl2trans)))
							  
			  (nnl2.hli.ts:axpy! a dl/da :alpha (- lr))
			  (nnl2.hli.ts:axpy! b dl/db :alpha (- lr)))))))))
			  