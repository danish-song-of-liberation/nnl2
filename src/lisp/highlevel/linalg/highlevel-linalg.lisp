(in-package :nnl2.hli.ts.linalg)

(defun gesvd (a order jobu jobvt m n lda ldu ldvt) 
  "Compute Singular Value Decomposition (SVD) using standard LAPACK algorithm.
   
   Performs the decomposition A = U * Σ * V^T, where:
   A is an m×n input matrix
   U is an m×m (or m×min(m,n)) orthogonal matrix of left singular vectors
   Σ is a diagonal matrix with singular values (size min(m,n))
   V^T is an n×n (or min(m,n)×n) orthogonal matrix of right singular vectors (transposed)
   
   Args:
       a: Input matrix tensor [m × n] (any floating-point dtype)
       order: Storage order, either :nnl2rowmajor or :nnl2colmajor
	   
       jobu: Character specifying computation of U matrix:
             #\\A: All m columns of U
             #\\S: First min(m,n) columns of U
             #\\O: First min(m,n) columns of U overwrite input A
             #\\N: U is not computed
			 
       jobvt: Character specifying computation of V^T matrix:
              #\\A: All n rows of V^T
              #\\S: First min(m,n) rows of V^T
              #\\O: First min(m,n) rows of V^T overwrite input A
              #\\N: V^T is not computed
			  
       m: Number of rows (automatically determined if NIL)
       n: Number of columns (automatically determined if NIL)
       lda: Leading dimension of A (automatically determined if NIL)
       ldu: Leading dimension of U (automatically determined if NIL)
       ldvt: Leading dimension of VT (automatically determined if NIL)
   
     Returns:
         Four values:
             1. s: Singular values tensor [min(m,n)] in descending order
			 2. u: Left singular vectors matrix (size depends on jobu)
			 3. vt: Right singular vectors matrix (size depends on jobvt)
			 4. info: Return code (0 = success, >0 = convergence failure, <0 = illegal argument)
   
   Notes:
     All output tensors have same dtype as input tensor
     When jobu='O' or jobvt='O', input matrix A is overwritten
     Leading dimensions are automatically computed based on order if not provided
     Uses standard LAPACK gesvd algorithm (slower but more stable for some cases)
   
   Example:
     (gesvd a :nnl2rowmajor #\\A #\\A) ; Full SVD, row-major
     (gesvd a :nnl2colmajor #\\S #\\S) ; Economy SVD, column-major
   
   See also: gesdd, svd"
   
  (declare (optimize (speed 3) (safety 1)))
	
  (declare (type nnl2.hli.ts::nnl2-tensor a)
           (type keyword order)
           (type character jobu jobvt))

  (let* ((shape-a (nnl2.hli.ts:shape a :as :vector))
         (m (or m (aref shape-a 0)))
         (n (or n (aref shape-a 1)))
         (min-mn (min m n))  
         (lda (or lda (if (eq order :nnl2rowmajor) n m)))     
         (s (nnl2.hli.ts:empty (list min-mn) :dtype (nnl2.hli.ts:dtype a)))   
		 
         (u (cond ((char-equal jobu #\N) (make-tensor nil))
                  ((char-equal jobu #\O) (cffi:null-pointer))
                  (t (let ((u-rows m) (u-cols (if (char-equal jobu #\A) m min-mn)))
                       (nnl2.hli.ts:empty `(,u-rows ,u-cols) :dtype (nnl2.hli.ts:dtype a))))))
         
         (vt (cond ((char-equal jobvt #\N) (make-tensor nil))
                   ((char-equal jobvt #\O) (cffi:null-pointer))
                   (t (let ((vt-rows (if (char-equal jobvt #\A) n min-mn)) (vt-cols n))
                        (nnl2.hli.ts:empty `(,vt-rows ,vt-cols) :dtype (nnl2.hli.ts:dtype a))))))
         
         (ldu (or ldu (cond ((char-equal jobu #\N) 1)
                            ((char-equal jobu #\O) 1)
                            ((eq order :nnl2rowmajor) (if (char-equal jobu #\A) m min-mn))
                            (t m))))
				
         (ldvt (or ldvt (cond ((char-equal jobvt #\N) 1)
                              ((char-equal jobvt #\O) 1)
                              ((eq order :nnl2rowmajor) n)
                              (t (if (char-equal jobvt #\A) n min-mn)))))
         
         (superb (nnl2.hli.ts:empty `(,(max 1 (* 5 min-mn))) :dtype (nnl2.hli.ts:dtype a))))
    
    (let ((info (nnl2.ffi:%gesvd order 
                                 (char-code jobu) 
                                 (char-code jobvt)
                                 m n
                                 a lda
                                 s
                                 u ldu
                                 vt ldvt
                                 superb)))
	  
      (values s 
	    (if (cffi:null-pointer-p u) nil u)
        (if (cffi:null-pointer-p vt) nil vt)
		info))))

(defun gesdd (a order jobz m n lda ldu ldvt) 
  "Compute Singular Value Decomposition (SVD) using Divide-and-Conquer algorithm
   
   Performs the decomposition A = U * Σ * V^T using LAPACK's divide-and-conquer
   algorithm (gesdd), which is typically faster than standard gesvd for large
   matrices but uses more workspace memory
   
   Args:
       a: Input matrix tensor [m × n] (any floating-point dtype)
       order: Storage order, either :nnl2rowmajor or :nnl2colmajor
	   
       jobz: Character specifying computation of singular vectors:
             #\\A: All m columns of U and all n rows of V^T
             #\\S: First min(m,n) columns of U and rows of V^T
			 
             #\\O: Overwrites A with singular vectors:
                    If m >= n: First n columns of U overwrite A
                    If m < n: First m rows of V^T overwrite A
					
             #\\N: Neither U nor V^T are computed
			 
       m: Number of rows (automatically determined if NIL)
       n: Number of columns (automatically determined if NIL)
       lda: Leading dimension of A (automatically determined if NIL)
       ldu: Leading dimension of U (automatically determined if NIL)
       ldvt: Leading dimension of VT (automatically determined if NIL)
   
     Returns:
         Four values:
             1. s: Singular values tensor [min(m,n)] in descending order
             2. u: Left singular vectors matrix (size depends on jobz)
             3. vt: Right singular vectors matrix (size depends on jobz)
             4. info: Return code (0 = success, >0 = convergence failure, <0 = illegal argument)
   
   Notes:
       All output tensors have same dtype as input tensor
       When jobz='O', input matrix A is overwritten with singular vectors
       Requires integer workspace tensor (iwork) of size 8*min(m,n)
       Typically faster than gesvd for matrices with min(m,n) > 25
       Uses LAPACK's divide-and-conquer algorithm (DBDSDC/SBDSDC)
   
   Example:
     (gesdd a :nnl2rowmajor #\\A) ; Full SVD, row-major
     (gesdd a :nnl2colmajor #\\S) ; Economy SVD, column-major
   
   See also: gesvd, svd"
   
  (declare (optimize (speed 3) (safety 1)))
  
  (declare (type nnl2.hli.ts::nnl2-tensor a)
           (type keyword order)
           (type character jobz))

  (let* ((shape-a (nnl2.hli.ts:shape a :as :vector))
         (m (or m (aref shape-a 0)))
         (n (or n (aref shape-a 1)))
         (min-mn (min m n))  
         (lda (or lda (if (eq order :nnl2rowmajor) n m)))     
         (s (nnl2.hli.ts:empty (list min-mn) :dtype (nnl2.hli.ts:dtype a)))   
         
         (u (cond ((char-equal jobz #\N) (make-tensor nil))
                  ((char-equal jobz #\O) (cffi:null-pointer))
                  (t (let ((u-rows m) (u-cols (if (char-equal jobz #\A) m min-mn)))
                       (nnl2.hli.ts:empty `(,u-rows ,u-cols) :dtype (nnl2.hli.ts:dtype a))))))
         
         (vt (cond ((char-equal jobz #\N) (make-tensor nil))
                   ((char-equal jobz #\O) (cffi:null-pointer))
                   (t (let ((vt-rows (if (char-equal jobz #\A) n min-mn)) (vt-cols n))
                        (nnl2.hli.ts:empty `(,vt-rows ,vt-cols) :dtype (nnl2.hli.ts:dtype a))))))
         
         (ldu (or ldu (cond ((char-equal jobz #\N) 1)
                            ((char-equal jobz #\O) 1)
                            ((eq order :nnl2rowmajor) (if (char-equal jobz #\A) m min-mn))
                            (t m))))
         
         (ldvt (or ldvt (cond ((char-equal jobz #\N) 1)
                              ((char-equal jobz #\O) 1)
                              ((eq order :nnl2rowmajor) (if (char-equal jobz #\A) n min-mn))
                              (t (if (char-equal jobz #\A) n min-mn)))))
         
         (iwork (nnl2.hli.ts:empty (list (* 8 min-mn)) :dtype :int32)))
    
    (let ((info (nnl2.ffi:%gesdd order 
                                 (char-code jobz)
                                 m n
                                 a lda
                                 s
                                 u ldu
                                 vt ldvt
                                 iwork)))
      
      (values s 
              (if (cffi:null-pointer-p u) nil u)
              (if (cffi:null-pointer-p vt) nil vt)
			  info))))

(defun svd (a &key (lapack :gesdd) (order :nnl2rowmajor) (jobu #\A) (jobvt #\A) (jobz #\A) m n lda ldu ldvt)
  "Singular Value Decomposition (SVD) operation
   A = U * Σ * V^T
   
   Args:
     a: Input matrix tensor [m × n]
	 
     lapack: Algorithm to use:
             :gesvd - Standard SVD algorithm
             :gesdd - Divide-and-Conquer SVD algorithm 
			 Default: :gesdd
			 
     order: Storage order (:nnl2rowmajor or :nnl2colmajor)
     
     Parameters for gesvd:
		 jobu: Options for computing U matrix (for gesvd):
			   #\A - all m columns of U
			   #\S - first min(m,n) columns of U  
			   #\O - first min(m,n) columns of U overwrite input A
			   #\N - no columns of U computed
			   
		 jobvt: Options for computing V^T matrix (for gesvd):
				#\A - all n rows of V^T
				#\S - first min(m,n) rows of V^T
				#\O - first min(m,n) rows of V^T overwrite input A
				#\N - no rows of V^T computed
            
     Parameters for gesdd:
		 jobz: Options for computing singular vectors (for gesdd):
			   #\A - all m columns of U and all n rows of V^T
			   #\S - first min(m,n) columns of U and rows of V^T
			   #\O - 
			   ;    If m >= n: first n columns of U overwrite A, V^T is computed
			   ;    If m < n: first m rows of V^T overwrite A, U is computed
			   #\N - neither U nor V^T are computed
			   
   m, n: Matrix dimensions (automatically determined if not provided)
   lda, ldu, ldvt: Leading dimensions (automatically determined if not provided)
   
   Returns:
       Multiple values: (s u vt info) where:
       s: singular values tensor [min(m,n)]
       u: left singular vectors tensor (size depends on jobu/jobz)
       vt: right singular vectors tensor (size depends on jobvt/jobz)
	   info: return code (0 = success)"
   
  (ecase lapack 
    (:gesvd (gesvd a order jobu jobvt m n lda ldu ldvt))
    (:gesdd (gesdd a order jobz m n lda ldu ldvt))))
	
(declaim (inline svd))  

(defun diag (tensor &key (k 0))
  "Extract or construct a diagonal matrix/vector
   
   Args:
       tensor: Input tensor (vector or matrix)
       k (&key) (default: 0): Diagonal index
       k = 0: main diagonal
       k > 0 : K-th diagonal above the main
       k < 0 : K-th diagonal below the main

   Returns:
       If input is vector: New square matrix with diagonal elements
       If input is matrix: New vector containing diagonal elements"
	   
  (if (= (nnl2.hli.ts:rank tensor) 1)
    (nnl2.ffi:%diag-vector-matrix tensor k)
	(nnl2.ffi:%diag-matrix-vector tensor k)))
		
(declaim (inline diag))
		
(defun luf (a &key (order :nnl2rowmajor) m n lda ipiv)
  "Compute LU factorization with partial pivoting
  
   Performs the decomposition A = P * L * U, where
   P is a permutation matrix
   L is lower triangular with unit diagonal elements
   U is upper triangular
  
   Args:
       a: Input matrix tensor [m × n] (any floating-point dtype)
       order: Storage order, either :nnl2rowmajor or :nnl2colmajor
       m: Number of rows (automatically determined if NIL)
       n: Number of columns (automatically determined if NIL)
       lda: Leading dimension of A (automatically determined if NIL)
       ipiv: Existing tensor for pivot indices or NIL to create new
  
   Returns:
       Three values:
           1. lu: Matrix containing LU factors in compact form
           2. ipiv: Pivot indices tensor [min(m,n), INT32]
           3. info: Return code (0 = success, >0 = singular matrix, <0 = illegal argument)
  
   Notes:
       Input matrix A is overwritten with LU factors in compact form:
         - Upper triangle (including diagonal) contains U
         - Lower triangle (excluding diagonal) contains multipliers for L
         - Diagonal elements of L are implied to be 1.0
       
       Pivot indices are 1-based (LAPACK convention)
       If info > 0, U(info,info) is exactly zero (matrix is singular)
  
   Example:
     (multiple-value-bind (lu ipiv info) 
         (getrf a :nnl2colmajor)
       (when (zerop info)
         ;; Use lu and ipiv for solving linear systems
         ))"
  
  (declare (optimize (speed 3) (safety 1)))
  
  (declare (type nnl2.hli.ts::nnl2-tensor a)
           (type keyword order))
  
  (let* ((shape-a (nnl2.hli.ts:shape a :as :vector))
         (m (or m (aref shape-a 0)))
         (n (or n (aref shape-a 1)))
         (min-mn (min m n))
         (lda (or lda (if (eq order :nnl2rowmajor) n m)))
         
         (ipiv-tensor (if (null ipiv)
                        (nnl2.hli.ts:empty (list min-mn) :dtype :int32)
                        (progn
                          (assert (equal (nnl2.hli.ts:shape ipiv :as :list) (list min-mn)) nil "IPIV tensor must have shape (~D)" min-mn)
                          ipiv)))
         
         (lu (nnl2.hli.ts:copy a)))
    
    (let ((info (nnl2.ffi:%getrf order 
                                 m n
                                 lu lda
                                 ipiv-tensor)))
      
      (values lu ipiv-tensor info))))		
		
(defun eye (rows cols &key (dtype nnl2.system:*default-tensor-type*))
  "Create identity matrix tensor
   
   Args:
       rows: Number of rows 
	   cols: Number of cols 
	   dtype (&key) (default: nnl2.system:*default-tensor-type*): Data type
	   
   Example:
       (nnl2.hli.ts:tlet ((a (nnl2.hli.ts.linalg:eye 3 3))) ...) ;; -> [[1 0 0] [0 1 0] [0 0 1]]"
	   
  (nnl2.ffi:%eye rows cols dtype))		

(cffi:defcfun ("nnl2_eye_like" eye-like) :pointer 
  (tensor :pointer))
  
(defun trefw-crutch (tensor &rest at)
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr at)
    (let ((elem (nnl2.ffi:%lowlevel-tref-with-coords tensor shape rank)))
	  (if (cffi:null-pointer-p elem)
	    (error "Pointer can't be NULL")
		(cffi:mem-ref elem (nnl2.hli.ts:type/nnl2->cffi (nnl2.hli.ts:dtype tensor)))))))  
  
(defun (setf trefw-crutch) (with tensor &rest at)
  (multiple-value-bind (shape rank) (nnl2.hli:make-shape-pntr at)
    (let* ((dtype (nnl2.hli.ts:dtype tensor))
		   (cffi-type (nnl2.hli.ts:type/nnl2->cffi dtype))
		   (lisp-type (nnl2.hli.ts:type/nnl2->lisp dtype))
		   (filler-pntr (cffi:foreign-alloc cffi-type)))

     (setf (cffi:mem-ref filler-pntr cffi-type) (coerce with lisp-type))
	 
	 (nnl2.ffi:%lowlevel-tref-with-coords-setter tensor shape rank filler-pntr))))  
  
(defun lufu (a)
  (let* ((n (nnl2.hli.ts:nrows a))
		 (u (nnl2.hli.ts:zeros (list n n) :dtype (nnl2.hli.ts:dtype a))))
		 
	(dotimes (i n)
      (dotimes (j n)
        (when (<= i j)
          (setf (trefw-crutch u i j) (trefw-crutch a i j)))))	 
		 
	u))
	
(defun lufl (lu)
  (let* ((n (nnl2.hli.ts:nrows lu))
         (l (nnl2.hli.ts:zeros (list n n) :dtype (nnl2.hli.ts:dtype lu))))
		 
    (dotimes (i n)
      (dotimes (j n)
        (cond
          ((> i j) (setf (trefw-crutch l i j) (trefw-crutch lu i j)))
          ((= i j) (setf (trefw-crutch l i j) 1)))))
		  
    l))	
	
(defun lufp (ipiv n &key (dtype nnl2.system:*default-tensor-type*))
  (let ((p (eye n n :dtype dtype)))
    (dotimes (k (nnl2.hli.ts:nrows ipiv))
      (let ((pivot (1- (trefw-crutch ipiv k)))) 
        (unless (= k pivot)
          (nnl2.hli.ts.utils:swap-rows! p k pivot))))

	p))

(defun lu (a &key (p t))
  (multiple-value-bind (lu ipiv) 
      (luf a)
	  
	(let ((u (lufu lu))
		  (l (lufl lu))
		  (plu nil))
		  
	  (when p
		(setf plu (lufp ipiv (nnl2.hli.ts:nrows ipiv))))
		  
	  (nnl2.hli.ts:free lu)	  
	  (nnl2.hli.ts:free ipiv)	  
		  
	  (if p 
	    (values l u plu)
		(values l u)))))

(defun triu (tensor &key (k 0))
  (nnl2.ffi:%triu tensor k))
  
(defun tril (tensor &key (k 0))
  (nnl2.ffi:%tril tensor k))  
