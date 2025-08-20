(in-package :nnl2.hli.ts)

(defun symbol-to-uppercase-string (symbol)
  (string-upcase (symbol-name symbol)))

(defun uppercase-string-to-symbol (upstring)
  (intern (string-downcase upstring) :keyword))

(defun use-backend/.abs (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-abs-backend sig)))
	
(defun use-backend/.abs! (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-abs-inplace-backend sig)))
	
(defun use-backend/full (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-inplace-fill-backend sig)))  
  
(defun use-backend/empty (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-empty-backend sig)))
  
(defun use-backend/zeros (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-zeros-backend sig)))  

(defun use-backend/ones (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-ones-backend sig)))	
	
(defun use-backend/full-like (name) (use-backend/full name))	
(defun use-backend/empty-like (name) (use-backend/empty name))	
(defun use-backend/zeros-like (name) (use-backend/zeros name))	
(defun use-backend/ones-like (name) (use-backend/ones name))	
	
(defun use-backend/gemm (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-sgemminplace-backend sig)
	(nnl2.ffi:%set-dgemminplace-backend sig)))
	
(defun use-backend/+= (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-addinplace-backend sig)
	(nnl2.ffi:%set-add-incf-inplace-backend sig)
	(nnl2.ffi:%set-add-broadcasting-inplace-backend sig)))
	
(defun use-backend/-= (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-subinplace-backend sig)
	(nnl2.ffi:%set-sub-decf-inplace-backend sig)
	(nnl2.ffi:%set-sub-broadcasting-inplace-backend sig)))
 
(defun use-backend/*= (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-mulinplace-backend sig)
	(nnl2.ffi:%set-mul-mulf-inplace-backend sig)
	(nnl2.ffi:%set-mul-broadcasting-inplace-backend sig)))
  
(defun use-backend//! (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-divinplace-backend sig)
	(nnl2.ffi:%set-div-divf-inplace-backend sig)
	(nnl2.ffi:%set-div-broadcasting-inplace-backend sig)))
	
(defun use-backend/^= (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-powinplace-backend sig)
	(nnl2.ffi:%set-pow-powf-inplace-backend sig)
	(nnl2.ffi:%set-pow-broadcasting-inplace-backend sig)))

(defun use-backend/.log! (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-loginplace-backend sig)))	

(defun use-backend/scale! (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-scaleinplace-backend sig)))	

(defun use-backend/.min! (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-mininplace-backend sig)
	(nnl2.ffi:%set-min-minf-inplace-backend sig)
	(nnl2.ffi:%set-min-broadcasting-inplace-backend sig)))

(defun use-backend/.max! (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-maxinplace-backend sig)
	(nnl2.ffi:%set-max-maxf-inplace-backend sig)
	(nnl2.ffi:%set-max-broadcasting-inplace-backend sig)))

(defun use-backend/.+ (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-add-backend sig)
	(nnl2.ffi:%set-add-incf-backend sig)
	(nnl2.ffi:%set-add-broadcasting-backend sig)))
    
(defun use-backend/.- (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-sub-backend sig)
	(nnl2.ffi:%set-sub-decf-backend sig)
	(nnl2.ffi:%set-sub-broadcasting-backend sig)))
 
(defun use-backend/.* (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-div-backend sig)
	(nnl2.ffi:%set-mul-mulf-backend sig)
	(nnl2.ffi:%set-mul-broadcasting-backend sig)))
    
(defun use-backend/./ (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-div-backend sig)
	(nnl2.ffi:%set-div-divf-backend sig)
	(nnl2.ffi:%set-div-broadcasting-backend sig)))

(defun use-backend/.^ (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-pow-backend sig)
	(nnl2.ffi:%set-pow-powf-backend sig)
	(nnl2.ffi:%set-pow-broadcasting-backend sig)))

(defun use-backend/.log (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-log-backend sig)))  
 	
(defun use-backend/scale (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-scale-backend sig)))		
	
(defun use-backend/.min (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-min-backend sig)
	(nnl2.ffi:%set-min-minf-backend sig)
	(nnl2.ffi:%set-min-broadcasting-backend sig)))
	
(defun use-backend/.max (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-max-backend sig)
    (nnl2.ffi:%set-max-maxf-backend sig)
	(nnl2.ffi:%set-max-broadcasting-backend sig)))
	
(defun use-backend/hstack (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-hstack-backend sig)))		

(defun use-backend/vstack (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-vstack-backend sig)))

(defun use-backend/.relu! (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-reluinplace-backend sig)))		
	
(defun use-backend/.relu (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-relu-backend sig)))			
	
(defun use-backend/.leaky-relu! (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-leakyreluinplace-backend sig)))		
	
(defun use-backend/.leaky-relu (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-leakyrelu-backend sig)))	
	
(defun use-backend/.sigmoid! (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-sigmoidinplace-backend sig)))		
	
(defun use-backend/.sigmoid (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-sigmoid-backend sig)))	
	
(defun use-backend/.tanh! (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-tanhinplace-backend sig)))		
	
(defun use-backend/.tanh (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-tanh-backend sig)))	
	
(defun use-backend/concat (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-concat-backend sig)))		
	
(defun use-backend/randn (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-randn-backend sig)))			

(defun use-backend/randn-like (name) 
  (use-backend/randn name))	

(defun use-backend/xavier (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-xavier-backend sig)))		

(defun use-backend/transpose! (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-transposeinplace-backend sig)))		
	
(defun use-backend/transpose (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-transpose-backend sig)))

(defun use-backend/sum (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-sum-backend sig)))

(defun use-backend/l2norm (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-l2norm-backend sig)))

(defun use-backend/copy (name)	
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-copy-backend sig)))

(defun use-backend (name)
  (use-backend/+= name)
  (use-backend/-= name)
  (use-backend/*= name)
  (use-backend//! name)
  (use-backend/^= name)
  (use-backend/.log! name)
  (use-backend/.max! name)
  (use-backend/.min! name)
  (use-backend/scale! name)
  (use-backend/.+ name)
  (use-backend/.- name)
  (use-backend/.* name)
  (use-backend/./ name)
  (use-backend/.^ name)
  (use-backend/.log name)
  (use-backend/.min name)
  (use-backend/.max name)
  (use-backend/scale name)
  (use-backend/empty name)
  (use-backend/.abs name)
  (use-backend/xavier name)
  (use-backend/randn name)
  (use-backend/full name)	
  (use-backend/empty name)
  (use-backend/zeros name)
  (use-backend/sum name)
  (use-backend/l2norm name)
  (use-backend/copy name)
  (use-backend/gemm name)
  (use-backend/.relu name)
  (use-backend/.relu! name)
  (use-backend/.leaky-relu name)
  (use-backend/.leaky-relu! name)
  (use-backend/.sigmoid name)
  (use-backend/.sigmoid! name)
  (use-backend/.tanh name)
  (use-backend/.tanh! name)
  (use-backend/transpose name)
  (use-backend/transpose! name))
  
(defun get-backend/empty ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-empty-backend)))    
  
(defun get-backend/zeros ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-zeros-backend)))

(defun get-backend/ones ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-ones-backend)))  
  
(defun get-backend/full ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-inplace-fill-backend)))  
  
(defun get-backend/full-like () (get-backend/full))	
(defun get-backend/empty-like () (get-backend/empty))	
(defun get-backend/zeros-like () (get-backend/zeros))	
(defun get-backend/ones-like () (get-backend/ones))
  
(defun get-backend/gemm ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-gemm-backend)))

(defun (setf get-backend/gemm) (name)
  (use-backend/gemm name))  
    
(defun get-backend/gemm! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-gemm-backend)))    
  
(defun (setf get-backend/gemm!) (name)
  (use-backend/gemm! name))  
 
(defun get-backend/+= ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-addinplace-backend))) 

(defun (setf get-backend/+=) (name)
  (use-backend/+= name))  
 
(defun get-backend/-= ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-subinplace-backend)))

(defun (setf get-backend/-=) (name)
  (use-backend/-= name))  
  
(defun get-backend/.+ ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-add-backend)))

(defun (setf get-backend/.+) (name)
  (use-backend/.+ name))
 
(defun get-backend/.- ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-sub-backend)))  

(defun (setf get-backend/.-) (name)
  (use-backend/.- name))  
 
(defun get-backend/*= ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-mulinplace-backend)))       

(defun (setf get-backend/*=) (name)
  (use-backend/*= name))

(defun get-backend//! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-divinplace-backend)))  

(defun (setf get-backend//!) (name)
  (use-backend//! name))

(defun get-backend/.* ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-mul-backend)))    

(defun (setf get-backend/.*) (name)
  (use-backend/.* name))

(defun get-backend/./ ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-div-backend)))      
  
(defun (setf get-backend/./) (name)
  (use-backend/./ name))  
  
(defun get-backend/^= ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-powinplace-backend)))      

(defun (setf get-backend/^=) (name)
  (use-backend/^= name))
  
(defun get-backend/.^ ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-pow-backend)))      
  
(defun (setf get-backend/.^) (name)
  (use-backend/.^ name))  
  
(defun get-backend/.exp! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-expinplace-backend)))   
  
(defun (setf get-backend/.exp!) (name)
  (use-backend/.exp! name))  
  
(defun get-backend/.exp ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-exp-backend)))   

(defun (setf get-backend/.exp) (name)
  (use-backend/.exp name))

(defun get-backend/.log! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-loginplace-backend)))      

(defun (setf get-backend/.log!) (name)
  (use-backend/.log! name))
	
(defun get-backend/.log ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-log-backend)))    	
	
(defun (setf get-backend/.log) (name)
  (use-backend/.log name))	
	
(defun get-backend/scale! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-scaleinplace-backend)))    		

(defun (setf get-backend/scale!) (name)
  (use-backend/scale! name))

(defun get-backend/scale ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-scale-backend)))    		
	
(defun (setf get-backend/scale) (name)
  (use-backend/scale name))
	
(defun get-backend/.max! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-maxinplace-backend)))    		

(defun (setf get-backend/.max!) (name)
  (use-backend/.max! name))

(defun get-backend/.min! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-mininplace-backend)))  

(defun (setf get-backend/.min!) (name)
  (use-backend/.min! name))

(defun get-backend/.max ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-max-backend)))    		

(defun (setf get-backend/.max) (name)
  (use-backend/.max name))

(defun get-backend/.min ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-min-backend))) 

(defun (setf get-backend/.min) (name)
  (use-backend/.min name))

(defun get-backend/.abs! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-absinplace-backend)))   
  
(defun (setf get-backend/.abs!) (name)
  (use-backend/.abs! name))  
  
(defun get-backend/.abs ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-abs-backend)))  
  
(defun (setf get-backend/.abs) (name)
  (use-backend/.abs name))  
  
(defun get-backend/hstack ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-hstack-backend)))   
  
(defun (setf get-backend/hstack) (name)
  (use-backend/hstack name))  
  
(defun get-backend/vstack ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-vstack-backend)))    
  
(defun (setf get-backend/vstack) (name)
  (use-backend/vstack name))  
  
(defun get-backend/.relu! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-reluinplace-backend)))     
  
(defun (setf get-backend/.relu!) (name)
  (use-backend/.relu! name))  
  
(defun get-backend/.relu ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-relu-backend)))       
  
(defun (setf get-backend/.relu) (name)
  (use-backend/.relu name))  
  
(defun get-backend/.leaky-relu! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-leakyreluinplace-backend)))     
  
(defun (setf get-backend/.leaky-relu!) (name)
  (use-backend/.leaky-relu name))  
  
(defun get-backend/.leaky-relu ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-leakyrelu-backend)))   
 
(defun (setf get-backend/.leaky-relu) (name)
  (use-backend/.leaky-relu name))
 
(defun get-backend/.sigmoid! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-sigmoidinplace-backend)))     
 
(defun (setf get-backend/.sigmoid!) (name)
  (use-backend/.sigmoid! name))
 
(defun get-backend/.sigmoid ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-sigmoid-backend)))     
  
(defun (setf get-backend/.sigmoid) (name)
  (use-backend/.sigmoid name))  
  
(defun get-backend/.tanh! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-tanhinplace-backend)))     
  
(defun (setf get-backend/.tanh!) (name)
  (use-backend/.tanh! name))  
  
(defun get-backend/.tanh ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-tanh-backend)))    
  
(defun (setf get-backend/.tanh) (name)
  (use-backend/.tanh name))  
  
(defun get-backend/concat ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-concat-backend)))   
  
(defun (setf get-backend/concat) (name)
  (use-backend/concat name))  
  
(defun get-backend/randn ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-randn-backend)))  
 
(defun (setf get-backend/randn) (name)
  (use-backend/randn name)) 
 
(defun get-backend/randn-like () 
  (get-backend/randn))

(defun (setf get-backend/randn-like) (name)
  (use-backend/randn-like name))  
 
(defun get-backend/xavier ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-xavier-backend)))

(defun (setf get-backend/xavier) (name)
  (use-backend/xavier name))  
  
(defun get-backend/transpose! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-transposeinplace-backend))) 
    
(defun (setf get-backend/transpose!) (name)
  (use-backend/transpose! name))	
	
(defun get-backend/transpose ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-transpose-backend))) 
    
(defun (setf get-backend/transpose) (name)
  (use-backend/transpose name))
	
(defun get-backend/sum ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-sum-backend)))	
  
(defun (setf get-backend/sum) (name)
  (use-backend/sum name))  
  
(defun get-backend/norm (&key (p :l2))
  (case p
    (:l2 (uppercase-string-to-symbol (nnl2.ffi:%get-sum-backend)))
	(otherwise (error "Incorrect :p key in norm~%"))))

(defun (setf get-backend/norm) (name &key (p :l2))
  (use-backend/norm name :p p))
	
(defun get-backend/copy ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-copy-backend)))

(defun (setf get-backend/copy) (name)
  (use-backend/copy name))  
  
(defun get-backend/axpy! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-axpy-inplace-backend)))
  
(defun (setf get-backend/axpy!) (name)
  (use-backend/axpy! name))  
  
(defun get-backend/axpy ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-axpy-backend)))

(defun (setf get-backend/axpy) (name)
  (use-backend/axpy name))
  
(defun get-backends/empty ()
  (let ((num-backends (nnl2.ffi:%get-empty-num-backends))
	    (backends (nnl2.ffi:%get-empty-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))
  
(defun get-backends/zeros ()
  (let ((num-backends (nnl2.ffi:%get-zeros-num-backends))
	    (backends (nnl2.ffi:%get-zeros-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))  
  
(defun get-backends/full ()
  (let ((num-backends (nnl2.ffi:%get-inplace-fill-num-backends))
	    (backends (nnl2.ffi:%get-inplace-fill-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))    
  
(defun get-backends/ones ()
  (let ((num-backends (nnl2.ffi:%get-ones-num-backends))
	    (backends (nnl2.ffi:%get-ones-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))    
    
(defun get-backends/full-like () (get-backends/full))	
(defun get-backends/empty-like () (get-backends/empty))	
(defun get-backends/zeros-like () (get-backends/zeros))	
(defun get-backends/ones-like () (get-backends/ones))
	
(defun get-backends/gemm ()
  (let ((num-backends (nnl2.ffi:%get-gemm-num-backends))
	    (backends (nnl2.ffi:%get-gemm-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))    
		  
(defun get-backends/gemm! ()
  (let ((num-backends (nnl2.ffi:%get-gemm-num-backends))
	    (backends (nnl2.ffi:%get-gemm-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))   

(defun get-backends/+= ()
  (let ((num-backends (nnl2.ffi:%get-addinplace-num-backends))
	    (backends (nnl2.ffi:%get-addinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))   
 
(defun get-backends/-= ()
  (let ((num-backends (nnl2.ffi:%get-subinplace-num-backends))
	    (backends (nnl2.ffi:%get-subinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))   
 
(defun get-backends/.+ ()
  (let ((num-backends (nnl2.ffi:%get-add-num-backends))
	    (backends (nnl2.ffi:%get-add-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))  
 
(defun get-backends/.- ()
  (let ((num-backends (nnl2.ffi:%get-sub-num-backends))
	    (backends (nnl2.ffi:%get-sub-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))  
		  
(defun get-backends/*= ()
  (let ((num-backends (nnl2.ffi:%get-mulinplace-num-backends))
	    (backends (nnl2.ffi:%get-mulinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))  
		  		  
(defun get-backends//! ()
  (let ((num-backends (nnl2.ffi:%get-divinplace-num-backends))
	    (backends (nnl2.ffi:%get-divinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))  
		  		  				  
(defun get-backends/.* ()
  (let ((num-backends (nnl2.ffi:%get-mul-num-backends))
	    (backends (nnl2.ffi:%get-mul-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))  
		  								  
(defun get-backends/./ ()
  (let ((num-backends (nnl2.ffi:%get-div-num-backends))
	    (backends (nnl2.ffi:%get-div-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
		  
(defun get-backends/^= ()
  (let ((num-backends (nnl2.ffi:%get-powinplace-num-backends))
	    (backends (nnl2.ffi:%get-powinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 

(defun get-backends/.^ ()
  (let ((num-backends (nnl2.ffi:%get-pow-num-backends))
	    (backends (nnl2.ffi:%get-pow-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
		  
(defun get-backends/.exp! ()
  (let ((num-backends (nnl2.ffi:%get-expinplace-num-backends))
	    (backends (nnl2.ffi:%get-expinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 		  

(defun get-backends/.exp ()
  (let ((num-backends (nnl2.ffi:%get-exp-num-backends))
	    (backends (nnl2.ffi:%get-exp-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	
	
(defun get-backends/.log! ()
  (let ((num-backends (nnl2.ffi:%get-loginplace-num-backends))
	    (backends (nnl2.ffi:%get-loginplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 		  

(defun get-backends/.log ()
  (let ((num-backends (nnl2.ffi:%get-log-num-backends))
	    (backends (nnl2.ffi:%get-log-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	
		  		  
(defun get-backends/scale! ()
  (let ((num-backends (nnl2.ffi:%get-scaleinplace-num-backends))
	    (backends (nnl2.ffi:%get-scaleinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/scale ()
  (let ((num-backends (nnl2.ffi:%get-scale-num-backends))
	    (backends (nnl2.ffi:%get-scale-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
		  
(defun get-backends/.max! ()
  (let ((num-backends (nnl2.ffi:%get-maxinplace-num-backends))
	    (backends (nnl2.ffi:%get-maxinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/.min! ()
  (let ((num-backends (nnl2.ffi:%get-mininplace-num-backends))
	    (backends (nnl2.ffi:%get-mininplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 

(defun get-backends/.max ()
  (let ((num-backends (nnl2.ffi:%get-max-num-backends))
	    (backends (nnl2.ffi:%get-max-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/.min ()
  (let ((num-backends (nnl2.ffi:%get-min-num-backends))
	    (backends (nnl2.ffi:%get-min-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
	
(defun get-backends/.abs! ()
  (let ((num-backends (nnl2.ffi:%get-absinplace-num-backends))
	    (backends (nnl2.ffi:%get-absinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/.abs ()
  (let ((num-backends (nnl2.ffi:%get-abs-num-backends))
	    (backends (nnl2.ffi:%get-abs-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
	
(defun get-backends/hstack ()
  (let ((num-backends (nnl2.ffi:%get-hstack-num-backends))
	    (backends (nnl2.ffi:%get-hstack-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/vstack ()
  (let ((num-backends (nnl2.ffi:%get-vstack-num-backends))
	    (backends (nnl2.ffi:%get-vstack-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
		
(defun get-backends/.relu! ()
  (let ((num-backends (nnl2.ffi:%get-reluinplace-num-backends))
	    (backends (nnl2.ffi:%get-reluinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/.relu ()
  (let ((num-backends (nnl2.ffi:%get-relu-num-backends))
	    (backends (nnl2.ffi:%get-relu-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
		  	  
(defun get-backends/.leaky-relu! ()
  (let ((num-backends (nnl2.ffi:%get-leakyreluinplace-num-backends))
	    (backends (nnl2.ffi:%get-leakyreluinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/.leaky-relu ()
  (let ((num-backends (nnl2.ffi:%get-leakyrelu-num-backends))
	    (backends (nnl2.ffi:%get-leakyrelu-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
		  
(defun get-backends/.sigmoid! ()
  (let ((num-backends (nnl2.ffi:%get-sigmoidinplace-num-backends))
	    (backends (nnl2.ffi:%get-sigmoidinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/.sigmoid ()
  (let ((num-backends (nnl2.ffi:%get-sigmoid-num-backends))
	    (backends (nnl2.ffi:%get-sigmoid-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
		  		  
(defun get-backends/.tanh! ()
  (let ((num-backends (nnl2.ffi:%get-tanhinplace-num-backends))
	    (backends (nnl2.ffi:%get-tanhinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 	

(defun get-backends/.tanh ()
  (let ((num-backends (nnl2.ffi:%get-tanh-num-backends))
	    (backends (nnl2.ffi:%get-tanh-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i))))) 
		  
(defun get-backends/concat ()
  (let ((num-backends (nnl2.ffi:%get-concat-num-backends))
	    (backends (nnl2.ffi:%get-concat-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))
		  
(defun get-backends/randn ()
  (let ((num-backends (nnl2.ffi:%get-randn-num-backends))
	    (backends (nnl2.ffi:%get-randn-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))		  

(defun get-backends/randn-like () 
  (get-backends/randn))
		  
(defun get-backends/xavier ()
  (let ((num-backends (nnl2.ffi:%get-xavier-num-backends))
	    (backends (nnl2.ffi:%get-xavier-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))

(defun get-backends/transpose! ()
  (let ((num-backends (nnl2.ffi:%get-transposeinplace-num-backends))
	    (backends (nnl2.ffi:%get-transposeinplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))
		  		  
(defun get-backends/transpose ()
  (let ((num-backends (nnl2.ffi:%get-transpose-num-backends))
	    (backends (nnl2.ffi:%get-transpose-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))
		  
(defun get-backends/sum ()
  (let ((num-backends (nnl2.ffi:%get-sum-num-backends))
	    (backends (nnl2.ffi:%get-sum-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))
		  
(defun get-backends/l2norm ()
  (let ((num-backends (nnl2.ffi:%get-l2norm-num-backends))
	    (backends (nnl2.ffi:%get-l2norm-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))

(defun get-backends/norm (&key (p :l2))
  (case p 
    (:l2 (get-backends/l2norm))
	(otherwise (error "Incorrect :p key in norm~%"))))
	
(defun get-backends/copy ()
  (let ((num-backends (nnl2.ffi:%get-copy-num-backends))
	    (backends (nnl2.ffi:%get-copy-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))
		  
(defun get-backends/axpy! ()
  (let ((num-backends (nnl2.ffi:%get-axpy-inplace-num-backends))
	    (backends (nnl2.ffi:%get-axpy-inplace-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))		  
		  
(defun get-backends/axpy ()
  (let ((num-backends (nnl2.ffi:%get-axpy-num-backends))
	    (backends (nnl2.ffi:%get-axpy-backends)))
		
    (loop for i from 0 below num-backends
		  collect (uppercase-string-to-symbol (cffi:mem-aref backends :string i)))))
		  