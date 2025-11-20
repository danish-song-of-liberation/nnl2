(in-package :nnl2.optim)

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
  
(defun gd (tensors &key (lr 0.1))
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
	  
	  