(in-package :nnl2.hli.ts.tests)

;; NNL2

;; Filepath: nnl2/tests/ts/ts-tests-subtensor-map-inplace.lisp
;; File: ts-tests-subtensor-map-inplace.lisp

;; Continues tests /map but in-place (/map!)

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(fiveam:in-suite nnl2.hli.ts-suite)

(defparameter *default-/map!-operation-shape* '(5 5))

;; -- `/map!` tests section --	

(fiveam:test nnl2.hli.ts//map!/float64
  (/map-test :dtype :float64 :shape *default-/map!-operation-shape* :op #'nnl2.hli.ts:/map! :inplace t))		

(fiveam:test nnl2.hli.ts//map!/float32
  (/map-test :dtype :float32 :shape *default-/map!-operation-shape* :op #'nnl2.hli.ts:/map! :inplace t))	
   
(fiveam:test nnl2.hli.ts//map!/int32
  (/map-test :dtype :int32 :shape *default-/map!-operation-shape* :op #'nnl2.hli.ts:/map! :inplace t))	
  