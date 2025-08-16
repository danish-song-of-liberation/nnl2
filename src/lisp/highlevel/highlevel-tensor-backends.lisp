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
  
(defun use-backend/empty (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-empty-backend sig)))

(defun use-backend/ones (name)
  (let ((sig (symbol-to-uppercase-string name)))
    (nnl2.ffi:%set-ones-backend sig)))	
	
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
  
(defun get-backend/gemm ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-gemm-backend)))  
    
(defun get-backend/gemm! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-gemm-backend)))    
 
(defun get-backend/+= ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-addinplace-backend)))    
 
(defun get-backend/-= ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-subinplace-backend)))    
  
(defun get-backend/.+ ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-add-backend)))    
 
(defun get-backend/.- ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-sub-backend)))     
 
(defun get-backend/*= ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-mulinplace-backend)))       

(defun get-backend//! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-divinplace-backend)))  

(defun get-backend/.* ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-mul-backend)))    

(defun get-backend/./ ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-div-backend)))      
  
(defun get-backend/^= ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-powinplace-backend)))      
  
(defun get-backend/.^ ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-pow-backend)))      
  
(defun get-backend/.exp! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-expinplace-backend)))   
  
(defun get-backend/.exp ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-exp-backend)))   

(defun get-backend/.log! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-loginplace-backend)))      
	
(defun get-backend/.log ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-log-backend)))    	
	
(defun get-backend/scale! ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-scaleinplace-backend)))    		

(defun get-backend/scale ()
  (uppercase-string-to-symbol (nnl2.ffi:%get-scale-backend)))    		
		