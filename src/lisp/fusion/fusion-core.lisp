(in-package :nnl2.fusion)

;; NNL2

;; Filepath: nnl2/src/lisp/fusion/fusion-core.lisp
;; File: fusion-core.lisp

;; File contains fusion logic

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defparameter *rules* nil
  "List of fusion rules 
   Each rule is a cons (pattern . replacement)")

(defun add-rule (pattern replacement)
  "Add a new fusion rule to the system
  
   Args: 
       pattern: The pattern to match against forms
       replacement: The replacement form when pattern matches
	   
   Returns: 
       Updated rules list"
   
  (push (cons pattern replacement) *rules*))

(defun symbol-equal (a b)
  "Check if two symbols are equal by name (ignoring package)
  
   Args: 
       a, b: Symbols to compare
   
   Returns: 
       T if symbols have same name, NIL otherwise"
   
  (and (symbolp a) (symbolp b) (string= (symbol-name a) (symbol-name b))))

(defun extract-variable-type (pattern-symbol)
  "Extract variable name and type from pattern symbol like `?name=type`
  
   Args: 
       pattern-symbol: Symbol starting with `?` optionally followed by `=type`
   
   Returns: 
       Two values: variable name and type (or NIL if no type specified)"
  
  (let ((name (symbol-name pattern-symbol)))
    (when (and (> (length name) 0) (char= (char name 0) #\?))
      (let ((equal-pos (position #\= name)))
        (if equal-pos
          (values (intern (subseq name 0 equal-pos)) (intern (subseq name (1+ equal-pos))))
          (values pattern-symbol nil))))))

(defun check-variable-type (value type)
  "Check if value matches the specified type
  
   Args: 
       value: Value to check
       type: Type symbol (real, nnl2.hli.ts:nnl2-tensor, symbol, number, list, body)
		 
   Returns: 
       T if value matches type, NIL otherwise"
   
  (case type
    (scalar (numberp value))
    (real (numberp value))
	
    (tensor (and (symbolp value) 
              (or 
			    (search "NNL2-TENSOR" (string value) :test #'char-equal)
				(search "NNL2.HLI.TS::NNL2-TENSOR" (string value) :test #'char-equal)
				(search "TENSOR" (string value) :test #'char-equal)
                (search "ARRAY" (string value) :test #'char-equal))))
				
    (symbol (symbolp value))
    (number (numberp value))
    (list (listp value))
    (body (listp value))
    (t t)))

(defun pattern-match (pattern form bindings)
  "Match pattern against form, collecting variable bindings
  
   Args: 
       pattern: Pattern to match
       form: Form to match against
       bindings: Current variable bindings (alist)
		 
   Returns: 
       Two values: Updated bindings and success flag"

  (cond
    ((and (null pattern) (null form)) (values bindings t))
    ((or (null pattern) (null form)) (values nil nil)) 
    ((eq pattern '?) (values (cons (cons '? form) bindings) t))
    ((eq pattern '&body) (values (cons (cons '&body form) bindings) t))
	
    ((and (symbolp pattern) (multiple-value-bind (var-name var-type) (extract-variable-type pattern) var-name))
       (multiple-value-bind (var-name var-type) (extract-variable-type pattern)
         (if (or (null var-type) (check-variable-type form var-type))
           (values (cons (cons var-name form) bindings) t)
           (values nil nil))))
		  
    ((and (symbolp pattern) (let ((name (symbol-name pattern))) (and (> (length name) 0) (char= (char name 0) #\?))))
       (values (cons (cons pattern form) bindings) t))
	   
    ((and (listp pattern) (listp form))
       (multiple-value-bind (new-bindings success) (pattern-match (car pattern) (car form) bindings)
         (if success
           (pattern-match (cdr pattern) (cdr form) new-bindings)
           (values nil nil))))
		   
    ((and (symbolp pattern) (symbolp form))
       (if (symbol-equal pattern form)
         (values bindings t)
         (values nil nil)))
		 
    (t 
	  (if (equal pattern form)
        (values bindings t)
        (values nil nil)))))
		
(defun collect-vars (expr in-arg-position vars)
  "Collect variable names from expression into hash table
  
   Args:
       expr: Expression to analyze
       in-arg-position: T if symbol is in argument position (not function name)
       vars: Hash table to collect variable names into
   
   Returns:
       Modified vars hash table"
  
  (cond
    ((symbolp expr)
       (when (and in-arg-position (not (special-operator-p expr)) (not (fboundp expr)))
         (setf (gethash (symbol-name expr) vars) t)))
    
    ((listp expr)
       (case (car expr) 
         ((quote function) nil)
       
         ((let let* tlet tlet*)
            (dolist (binding (cadr expr))
              (when (listp binding)
                (collect-vars (car binding) nil vars) 
                (collect-vars (cadr binding) t vars)))
				
            (mapc #'(lambda (x) (collect-vars x t vars)) (cddr expr)))

         (t (mapc #'(lambda (x) (collect-vars x t vars)) (cdr expr)))))))

(defun used-variables (form)
  "Find all variable names used in a form
  
   Args:
       form: Form to analyze
   
   Returns:
       List of variable names (as strings)"
  
  (let ((vars (make-hash-table :test 'equal)))
    (when (and (symbolp form) (not (special-operator-p form)) (not (fboundp form)))
      (setf (gethash (symbol-name form) vars) t))
    
    (collect-vars form nil vars)
    (loop for var being the hash-keys of vars collect var)))

(defun pattern-variables (pattern)
  "Extract all variable symbols from a pattern
  
   Args:
       pattern: Pattern to analyze
   
   Returns:
       List of variable symbols (starting with '?')"
  
  (let ((vars '()))
    (labels ((extract-vars (expr)
               (cond
                 ((and (symbolp expr) (let ((name (symbol-name expr))) (and (> (length name) 0) (char= (char name 0) #\?))))
                    (push expr vars))
					
                 ((listp expr) (mapc #'extract-vars expr)))))
				 
      (extract-vars pattern)
	  
     vars)))

(defun process-item (item bindings)
  "Process a single item in replacement template, substituting variables
  
   Args:
       item: Item to process (symbol, list, or other)
       bindings: Variable bindings (alist)
   
   Returns:
       Processed item with variables substituted"
  
  (cond
    ((eq item '&body)
       (let ((body-binding (assoc '&body bindings)))
         (if body-binding (cdr body-binding) '&body)))
		 
    ((and (symbolp item) (let ((name (symbol-name item))) (and (> (length name) 0) (char= (char name 0) #\?))))
       (let* ((item-name (symbol-name item))
	   
              (var-name (if (position #\= item-name)
                          (subseq item-name 0 (position #\= item-name))
                          item-name))
						  
              (binding (find-if #'(lambda (b) (string= var-name (symbol-name (car b)))) bindings)))
			  
         (if binding (cdr binding) item)))
		 
    ((listp item) (mapcar #'(lambda (x) (process-item x bindings)) item))
	
    (t item)))

(defun apply-replacement (replacement bindings)
  "Apply variable substitutions to replacement template
  
   Args:
       replacement: Replacement template
       bindings: Variable bindings (alist)
   
   Returns:
       Form with variables substituted"
  
  (if (listp replacement)
    (mapcar #'(lambda (item) (process-item item bindings)) replacement)
    (process-item replacement bindings)))

(defun pattern-var->name (pattern-var)
  "Extract clean variable name from pattern variable symbol
  
   Args:
       pattern-var: Pattern variable symbol (starting with '?')
   
   Returns:
       Variable name as string (without '?' and type suffix)"
  
  (let ((name (symbol-name pattern-var)))
    (subseq name 1 (or (position #\= name) (length name)))))

(defun try-fusion (form)
  "Try to apply fusion rules to a form
  
   Args:
       form: Form to transform
   
   Returns:
       Transformed form if fusion applied, NIL otherwise"
  
  (dolist (rule *rules*)
    (multiple-value-bind (bindings success) (pattern-match (car rule) form nil)
      (when success
        (let* ((pattern (car rule))
               (body-binding (assoc '&body bindings))
               (used-vars (when body-binding (used-variables (cdr body-binding))))
               (pattern-vars (pattern-variables pattern))
               (intermediate-vars (remove '&body pattern-vars :test #'eq))

               (final-var (when (and 
								  (listp pattern) 
								  (member (car pattern) '(tlet* nnl2.hli.ts:tlet*))
                                  (listp (second pattern))
                                  (listp (car (last (second pattern)))))
								  
                            (caar (last (second pattern)))))
               
               (intermediate-vars-only (if final-var
                                         (remove final-var intermediate-vars :test #'eq)
                                         intermediate-vars))

               (intermediate-names (mapcar #'pattern-var->name intermediate-vars-only))

               (used-intermediate (find-if #'(lambda (pattern-var-name) (find pattern-var-name used-vars :test #'string=))
                                     intermediate-names)))

          (unless used-intermediate
            (return-from try-fusion 
              (apply-replacement (cdr rule) bindings)))))))
			  
  nil)

(defun transform (form &key (enable-fusion t))
  "Recursively transform a form, applying fusion where possible
  
   Args:
       form: Form to transform
       enable-fusion: Whether to apply fusion rules
   
   Returns:
       Transformed form"
  
  (cond
    ((listp form)
       (let ((fused (if enable-fusion (try-fusion form) nil)))
         (if fused
           (transform fused :enable-fusion enable-fusion)
           (case (car form)
             ((let let* tlet tlet*)
                (let* ((bindings (second form))
				
                       (processed-bindings
                         (mapcar #'(lambda (binding)
                                     (if (listp binding)
                                       (list (car binding) (transform (cadr binding) :enable-fusion enable-fusion))
                                       binding))
                           bindings))

                       (processed-body (mapcar #'(lambda (body-form) (transform body-form :enable-fusion enable-fusion)) (cddr form))))
					   
                `(,(car form) ,processed-bindings ,@processed-body)))
				
             (t (mapcar #'(lambda (item) (transform item :enable-fusion enable-fusion)) form))))))
			 
    (t form)))

(defmacro with-fusion ((&key (enable t) (fastcall nil) (timing nil) (expand nil)) &body body)
  "Macro to enable code fusion transformations
  
   Args:
       enable: Whether to apply fusion (default T)
       fastcall: Wrap in fastcall optimization (default NIL)
       timing: Measure and print execution time (default NIL)
       expand: Print macroexpansion without executing (default NIL)
   
   Returns:
       Transformed and optionally optimized code"
  
  (let* ((transformed-body `(progn ,@(mapcar #'(lambda (form) (transform form :enable-fusion enable)) body)))
  
         (core-body (if fastcall 
                       `(nnl2.hli:fastcall ,transformed-body)
                       transformed-body))
					   
         (final-body (if timing
                       `(let ((start (get-internal-real-time)))
                          (prog1 ,core-body
                            (format t "Execution time: ~f seconds~%" (/ (- (get-internal-real-time) start) internal-time-units-per-second))))
							
                        core-body)))
    
    (if expand
      `(progn
		 (nnl2.log:info "======== Macroexpand-1 output ========~%~a" ',final-body)
		 (nnl2.log:info "======== End Macroexpand-1 ========"))	
		 
      final-body)))
		
(defun reset-rules ()
  "Clear all fusion rules"
  (setq *rules* nil))

(reset-rules)
