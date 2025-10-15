(in-package :nnl2.lli.ts.tests)

;; Filepath: nnl2/tests/llits/llits-tests.lisp
;; File: llits-tests.lisp

;; Contains :nnl2.lli.ts tests

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.lli.ts.tests-suite)

(fiveam:test nnl2.lli.ts-test/flat
  (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:rand #(5 5) :dtype :float64)))
    (let ((val     (nnl2.hli.ts:tref a 0 0))
		  (flatval (nnl2.lli.ts:flat a 0)))
		  
      (fiveam:is (= val flatval)))))

(fiveam:test nnl2.lli.ts-test/trefw
  (nnl2.hli.ts:tlet ((a (nnl2.hli.ts:rand #(5 5) :dtype :float64)))
    (let ((trefval  (nnl2.hli.ts:tref a 0 0))
		  (trefwval (nnl2.lli.ts:trefw a 0 0)))
		  
	  (fiveam:is (= trefval trefwval)))))
	  