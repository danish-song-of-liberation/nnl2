(in-package :nnl2.convert.tests)

;; Filepath: nnl2/tests/convert/convert-tests.lisp
;; File: convert-tests.lisp

;; Contains :nnl2.convert tests

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.convert.tests-suite)

(fiveam:test nnl2.convert.test/nnl2->list
  (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:ones #(1) :dtype :float32))
					 (b (list 1.0s0)))
					 
    (fiveam:is (= (nth 0 b) (nnl2.hli.ts:tref a 0)))))
	
(fiveam:test nnl2.convert.test/nnl2->vector
  (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:ones #(1) :dtype :float32))
					 (b (make-array '(1) :initial-element 1.0s0)))
					 
    (fiveam:is (= (aref b 0) (nnl2.hli.ts:tref a 0)))))	
	