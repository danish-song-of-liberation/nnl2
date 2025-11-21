(in-package :nnl2.internal)

;; NNL2

;; Filepath: nnl2/src/lisp/internal/internal-main.lisp
;; File: internal-main.lisp

;; Contains internal C functions

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun ts-axpy-regional! (addend summand &key from to (alpha 1.0s0))
  "Performs in-place AXPY operation on a subregion of a tensor"
  (let* ((rank (nnl2.hli.ts:rank addend))
		 (from (if from from (make-array (list rank))))
		 (to (if to to (make-array (list to))))
		 (from-pntr (nnl2.hli:make-shape-pntr from))
		 (to-pntr (nnl2.hli:make-shape-pntr to)))
		
	(nnl2.ffi:%ts-regional-axpy-inplace addend summand alpha from-pntr to-pntr)))
	
(cffi:defcfun ("nnl2_ad_tensor_share_data" ad-share-data) :pointer
  (ad-tensor :pointer))  
	
	