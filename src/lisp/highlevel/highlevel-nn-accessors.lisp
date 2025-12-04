(in-package :nnl2.hli.nn)

(deftype nnl2-nn () 
  #+sbcl      'sb-sys:system-area-pointer
  #+clisp     'fi:foreign-data
  #+ccl       'ccl:macptr
  #+ecl       'si:foreign-data
  #+abcl      'system:foreign-pointer
  #+lispworks 'fli:pointer
  #+allegro   'excl:foreign-pointer)

(progn
  (defconstant +nn-type-fnn+ 0)
  (defconstant +nn-type-unknown+ 1))

(defparameter *nn-default-init-type* :kaiming/normal)

(cffi:defcfun ("nnl2_ann_free" free) :void
  (nn :pointer))

(defun forward (nn &rest args)
  (ecase (nnl2.ffi:%nn-get-type nn)
    (0 (apply #'nnl2.ffi:%nn-fnn-forward nn args))))
  
(defmacro nnlet ((&rest bindings) &body body)
  (let ((vars (mapcar #'car bindings)))
    `(let ,bindings
	   (unwind-protect
	       (progn ,@body)
		   
		 (progn 
		   ,@(loop for var in vars
		           collect `(when (typep ,var 'nnl2-nn) (free ,var))))))))

(defmacro nnlet* ((&rest bindings) &body body)
  (if (null bindings)
   `(progn ,@body)
    (let* ((binding (first bindings))
           (var (if (consp binding) (car binding) binding))
           (value (if (consp binding) (cadr binding) nil)))
     `(let (,binding)
        (unwind-protect
          (nnlet* ,(rest bindings) ,@body)
          (when (typep ,var 'nnl2-nn) (free ,var)))))))
	
(defun extract-parameters (params num-params)
  (let ((mem (cffi:foreign-alloc :pointer :count num-params))
		(lst-params nil))
		
    (dotimes (i num-params)
	  (push (cffi:mem-aref params :pointer i) lst-params))
	  
	(cffi:foreign-free mem)
	(nnl2.ffi:%nn-free-parameters params)
	
	lst-params))
	
(defun parameters (nn)
  (extract-parameters (nnl2.ffi:%nn-get-parameters nn) (nnl2.ffi:%nn-get-num-parameters nn)))

(defmacro fnn (in-features arrow out-features &key (bias t) (dtype nnl2.system:*default-tensor-type*) (init *nn-default-init-type*))
  (declare (ignore arrow))
  `(let* ((nn (nnl2.ffi:%create-nn-fnn ,in-features ,out-features ,bias ,dtype ,(if (keywordp init) init :identity))))
     ,(when (not (keywordp init))
        `(dotimes (i (length (parameters nn)))
           (funcall ,init (nth i (parameters nn)))))
		   
     nn))
  