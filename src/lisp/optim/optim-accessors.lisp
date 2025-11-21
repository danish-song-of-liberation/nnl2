(in-package :nnl2.optim)

;; NNL2

;; Filepath: nnl2/src/lisp/optim/optim-accessors.lisp
;; File: optim-accessors.lisp

;; Standard optimizers accessors

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(deftype nnl2-optim () 
  #+sbcl      'sb-sys:system-area-pointer
  #+clisp     'fi:foreign-data
  #+ccl       'ccl:macptr
  #+ecl       'si:foreign-data
  #+abcl      'system:foreign-pointer
  #+lispworks 'fli:pointer
  #+allegro   'excl:foreign-pointer)
  
(cffi:defcfun ("nnl2_optim_step" step!) :void
  (optim :pointer))
  
(cffi:defcfun ("nnl2_optim_zero" zero-grad!) :void
  (optim :pointer))  
  
(cffi:defcfun ("nnl2_optim_free" free) :void
  (optim :pointer))  
  
(defun gd (tensors &key (lr nnl2.system:*default-learning-rate*))
  "Create Gradient Descent optimizer
   
   Args:
       tensors: List of tensors or single tensor to optimize
	   lr (&key) (default: nnl2.system:*default-learning-rate*): Learning rate
	   
   Returns:
       Pointer to GD Optimizer object"
	   
  (let* ((tensors (if (listp tensors) (nnl2.hli:flatten tensors) (list tensors)))
         (len (length tensors))
         (mem (cffi:foreign-alloc :pointer :count len)))
    
    (dotimes (i len) 
      (setf (cffi:mem-aref mem :pointer i) (nth i tensors)))
    
    (let ((optimizer (nnl2.ffi:%optim-make-gd mem len (coerce lr 'single-float))))
      (tg:finalize optimizer #'(lambda () (cffi:foreign-free mem)))
      optimizer)))
	  
(defmacro leto ((&rest bindings) &body body)
  (let ((vars (mapcar #'car bindings)))
    `(let ,bindings
	   (unwind-protect
	       (progn ,@body)
		   
		 (progn 
		   ,@(loop for var in vars
		           collect `(when (typep ,var 'nnl2-optim) (free ,var))))))))  
  
(defmacro leto* ((&rest bindings) &body body)
  (if (null bindings)
   `(progn ,@body)
    (let* ((binding (first bindings))
           (var (if (consp binding) (car binding) binding))
           (value (if (consp binding) (cadr binding) nil)))
     `(let (,binding)
        (unwind-protect
          (leto* ,(rest bindings) ,@body)
          (when (typep ,var 'nnl2-optim) (free ,var)))))))  	  
	  
	  