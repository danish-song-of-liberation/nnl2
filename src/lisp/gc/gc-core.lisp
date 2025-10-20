;; NNL2

;; Filepath: nnl2/src/lisp/gc/gc-core.lisp
;; File: gc-core.lisp

;; Contains a main logic of nnl2 garbage collector

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(in-package :nnl2.gc)

(defparameter *gc* nil
  "List of tensors to free")
  
(defparameter *profile* nil
  "Should add profiling to free tensors")
  
(defun push (&rest tensors)
  "Adds tensors to GC. Last argument can be a finalizer function."
  (if (and tensors (functionp (car (last tensors))))
    (let ((finalizer (car (last tensors))))
	  (nbutlast tensors 2)
      (dolist (tensor tensors) 
	    (when *profile* (nnl2.log:info "===> Added ~a tensor into garbage collector" tensor)) 
		(cl:push (cons tensor finalizer) *gc*)))
	  
    (dolist (tensor tensors) 
	  (when *profile* (nnl2.log:info "===> Added ~a tensor into garbage collector" tensor)) 
	  (cl:push (cons tensor 'none) *gc*))))

(defun reset ()
  "Resets garbage collector"
  (setq *gc* nil))	
	
(defun gc ()
  "WARNING: 
      This feature is greatly simplified and 
	  is not yet complete. it will be completed 
	  in the future, but for now it only supports 
	  :nnl2.hli.ts tensors.
  
   Clears all tensors placed in nnl2.gc:*gc*"
   
  (dolist (tensor *gc*)
    (let ((finalizer (cdr tensor)) (tensor-to-free (car tensor)))
	  (when (functionp finalizer) 
	    (when *profile* (nnl2.log:info "<=== Calling finalizer: ~d" finalizer))
		(funcall finalizer tensor-to-free))
		
	  (when *profile* (nnl2.log:info "<=== Freeing tensor: ~a" tensor-to-free))
	  (nnl2.hli.ts:free tensor-to-free)))
	
  (reset))  
	
