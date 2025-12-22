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
	
(defun ts-concat-vectors (&rest vectors)
  (let* ((len (length vectors))
		 (dtype (nnl2.hli.ts:dtype (car vectors)))
		 (vectors-pntr (cffi:foreign-alloc :pointer :count len)))
		 
    (dotimes (i len)
	  (setf (cffi:mem-aref vectors-pntr :pointer i) (nth i vectors)))
	  
	(let ((result (nnl2.ffi:%vector-concat vectors-pntr len dtype)))
	  (cffi:foreign-free vectors-pntr)
	  result)))
	
(defun ad-concat-vectors (&rest vectors)
  (let* ((len (length vectors))
		 (dtype (nnl2.hli.ad:dtype (car vectors)))
		 (vectors-pntr (cffi:foreign-alloc :pointer :count len)))
		 
    (dotimes (i len)
	  (setf (cffi:mem-aref vectors-pntr :pointer i) (nth i vectors)))
	  
	(let ((result (nnl2.ffi:%ad-vector-concat vectors-pntr len dtype)))
	  (cffi:foreign-free vectors-pntr)
	  result)))	
	
(defun ts-vector-as-parameter (vector indices &key (start 0))
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr indices)
    (nnl2.ffi:%vector-as-parameter shape rank start vector)))
	
(defun ts-assign-row (dst src &key (seq-index 0))
  (nnl2.ffi:%internal-assign-row dst seq-index src))	
  
(defun ad-assign-row (dst src &key (seq-index 0) (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-internal-assign-row dst seq-index src track-graph))	  

(defun ts-assign-row-add (src dst &key (seq-index 0))
  (nnl2.ffi:%internal-assign-row-add dst seq-index src))	
  
(defun ts-timestep-view (src time)
  (nnl2.ffi:%internal-timestep-view src time))  
  
(defun ad-timestep-view (src time)
  (nnl2.ffi:%ad-internal-timestep-view src time))    
  