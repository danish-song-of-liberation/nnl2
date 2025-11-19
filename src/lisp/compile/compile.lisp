(in-package :nnl2.compile)

;; NNL2

;; Filepath: nnl2/src/lisp/compile/compile.lisp
;; File: compile.lisp

;; Contains the main logic of the :nnl2.compile package, :nnl2.compile.aot. 
;; The main functions are compile (compiles any C code) and 
;; function (compiles any C function and automatically generates 
;; a cffi wrapper), and the logic is mainly focused on them

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun get-library-extension ()
  "Returns the shared library file extension based on the operating system"
  (cond
    ((uiop:os-macosx-p) ".dylib")
    ((uiop:os-unix-p) ".so")
    ((uiop:os-windows-p) ".dll")
    (t (error "Unsupported operating system"))))
	
(defun build-output-path (output-dir name)
  "Constructs the full path to the output library file"
  (concatenate 'string output-dir name (get-library-extension)))	
  
(defun build-source-path (current-dir name)
  "Constructs the path to the source C file"
  (concatenate 'string current-dir "compile/user-code-" name ".c"))

(defun write-includes (stream includes)
  "Writes #include directives to the stream based on provided keys"
  (let ((include-map '((:ad . "../src/c/nnl2/ad.h")
                       (:ts . "../src/c/nnl2/ts.h")
                       (:core . "../src/c/nnl2/core.h")
                       (:friendly . "../src/c/nnl2/friendly.h"))))
					   
    (dolist (include-key includes)
      (let ((include-path (cdr (assoc include-key include-map))))
        (when include-path
          (format stream "#include \"~a\"~%" include-path))))))

(defun write-source-code (path code includes &key (if-exists :supersede) (if-does-not-exist :create))
  "Writes the source code to a file with appropriate includes"
  (with-open-file (stream path :direction :output :if-does-not-exist if-does-not-exist :if-exists if-exists)			  
    (write-includes stream includes)
    (princ code stream))
	
  path)		  
  
(defun build-compiler-command (source-path output-path name optimize-level flags)
  "Builds the compiler command string"
  (let ((output-file (build-output-path output-path name)))
    (format nil "gcc -shared -o ~a ~a~@[ ~a~] -O~d~@[ ~a~]"
            output-file
            source-path
            (when (uiop:os-unix-p) "-fPIC")
            optimize-level
            flags)))

(defun compile-shared-library (compiler-command log-p)
  "Compiles the shared library and returns the results"
  (multiple-value-bind (output error-output exit-code) 
      (uiop:run-program compiler-command :ignore-error-status t)
    
    (when log-p
      (nnl2.log:debug "Compiler output: ~a" output)
      (nnl2.log:debug "Compiler error output: ~a" error-output)
      (nnl2.log:debug "Compiler exit code: ~d" exit-code))
    
    (values output error-output exit-code)))

(defun load-foreign-library (library-path log-p)
  "Loads the shared library via cffi"
  (cffi:load-foreign-library library-path)
  (when log-p (nnl2.log:info "Library loaded from ~a" library-path)))	
  
(defun compile-and-load (code &key (include '()) (log t) (name *default-compile-name*) (optimize *default-optimize-level*) (if-exists :supersede)
                                   (if-does-not-exist :create) (nnl2-include '()) (flags "") (delete t))
								   
  "Compiles C code into a shared library and loads it"
  
  (let* ((current-dir nnl2.intern-system:*current-dir*)
         (source-path (build-source-path current-dir name))
         (output-path (concatenate 'string current-dir "compile/"))
         (library-path (build-output-path output-path name))
         (all-includes (union include nnl2-include)))
    
    ;; Write source code to file
    (write-source-code source-path code all-includes :if-exists if-exists :if-does-not-exist if-does-not-exist)
  
    (when log (nnl2.log:info "Source file saved at ~a" source-path))
    
    ;; Compile to shared library
    (let ((compiler-command (build-compiler-command source-path output-path name optimize flags)))
      (multiple-value-bind (output error-output exit-code) (compile-shared-library compiler-command log)
        (unless (zerop exit-code)
          (error "Compilation failed with exit code ~d~%Error output: ~a~%Compiler command: ~a~%" exit-code error-output compiler-command))
        
        (when log (nnl2.log:info "Library created at ~a" library-path))
        
        ;; Load the library
        (load-foreign-library library-path log)
		
		(when delete (delete-file source-path))
        
        (values library-path exit-code)))))  

(defmacro compile ((&key include log (name "unnamed") (optimize 2) debug (if-exists :supersede) 
						 (if-does-not-exist :create) nnl2-include (flags "") (delete t)) code)	
						 
  "Macro wrapper for the compile-and-load function for backward compatibility"
  
  `(compile-and-load ,code :include ,include :log ,log
                           :name ,name :optimize ,optimize
                           :if-exists ,if-exists :if-does-not-exist ,if-does-not-exist
                           :nnl2-include ,nnl2-include :flags ,flags
						   :delete ,delete))
	
(in-package :nnl2.compile.aot)
	
(defun trim-whitespace (string)
  "Remove whitespace from both ends of a string"
  (string-trim '(#\Space #\Tab #\Newline) string))

(defun split-into-words (string)
  "Split string into words by spaces"
  (loop with length = (length string)
        with start = 0
        with words = '()
        for i from 0 to length
        when (or (= i length) (char= (char string i) #\Space))
        do (let ((word (trim-whitespace (subseq string start i))))
             (when (> (length word) 0)
               (push word words))
             (setf start (1+ i)))
        finally (return (nreverse words))))
		
(define-condition name-extraction-error (error)
  ((text :initarg :text :reader error-text))
  (:report (lambda (condition stream)
             (format stream "Failed to extract name: ~a" (error-text condition)))))

(defun extract-function-name (c-code)
  "Extract function name from C function declaration"
  (let ((open-paren-pos (position #\( c-code)))
    (unless open-paren-pos
      (error 'name-extraction-error :text "No opening parenthesis found"))
    
    (let ((name-part (trim-whitespace (subseq c-code 0 open-paren-pos))))
      (if (find #\Space name-part)
          ;; "int foo" -> "foo"
          (let ((last-space-pos (position #\Space name-part :from-end t)))
            (subseq name-part (1+ last-space-pos)))
          name-part))))
		  
(defun extract-return-type-words (c-code)
  "Extract words comprising the return type from C function declaration"
  (let ((open-paren-pos (position #\( c-code)))
    (unless open-paren-pos
      (error 'name-extraction-error :text "No opening parenthesis found"))
    
    (let ((prefix (trim-whitespace (subseq c-code 0 open-paren-pos))))
      (split-into-words prefix))))
	 
(defparameter *type-mappings*
  '(("char" . :char)
    ("int" . :int)
    ("double" . :double)
    ("float" . :float)
    ("long" . :long)
    ("bool" . :bool)
    ("void" . :void)
    ("string" . :string)))

(defun detect-pointer-type (type-string)
  "Check if type string represents a pointer and return appropriate type"
  (cond ((find #\* type-string) :pointer) ((string= "char*" type-string) :string) (t nil)))
	
(defun parse-type (type-words)
  "Parse type specification from list of words"
  (let ((unsigned-p (member "unsigned" type-words :test #'string=))
        (type-name (find-if #'(lambda (word) (assoc word *type-mappings* :test #'string=)) type-words))
        (pointer-type (some #'detect-pointer-type type-words)))
    
    (cond (pointer-type pointer-type) (type-name (cdr (assoc type-name *type-mappings* :test #'string=))) (t :void))))	
	
(defun parse-parameter (param-words)
  "Parse parameter specification into (name type) list"
  (when param-words
    (let ((param-name (read-from-string (first (last param-words))))
          (param-type (parse-type param-words)))
		  
      (list param-name param-type))))	
	 
(define-condition signature-parse-error (error)
  ((text :initarg :text :reader error-text))
  (:report (lambda (condition stream)
             (format stream "Failed to parse signature: ~a" (error-text condition)))))
	 
(defun tokenize-function-signature (c-code)
  "Parse function parameters into list of parameter specifications."
  (let ((start-pos (position #\( c-code))
        (end-pos (position #\) c-code)))
    
    (unless (and start-pos end-pos)
      (error 'signature-parse-error :text "Malformed function signature"))
    
    (let ((params-text (trim-whitespace (subseq c-code (1+ start-pos) end-pos)))
          (parameters nil) (current-param nil) (current-word "") (depth 0)) 
      
      (when (> (length params-text) 0)
        (loop for char across params-text
              do (cond
                   ((char= char #\() (incf depth))
                   ((char= char #\)) (decf depth))
				   
                   ((and (char= char #\,) (= depth 0))
                      (push current-word current-param)
                      (push (nreverse current-param) parameters)
                      (setf current-param '()
                            current-word ""))
							
                   ((and (char= char #\Space) (= depth 0))
                      (when (> (length current-word) 0)
                        (push current-word current-param)
                        (setf current-word "")))
					  
                   (t
                     (setf current-word (concatenate 'string current-word (string char)))))
					 
              finally (progn
                        (when (> (length current-word) 0) (push current-word current-param))
                        (when current-param (push (nreverse current-param) parameters)))))
      
      (nreverse parameters))))
	 
(defun generate-cffi-definition (c-code &key name)
  "Generate CFFI defcfun form from C function declaration"
  (let* ((function-name (extract-function-name c-code))
         (return-type-words (extract-return-type-words c-code))
         (parameter-specs (tokenize-function-signature c-code))
         (return-type (parse-type return-type-words))
         (parameters (remove nil (mapcar #'parse-parameter parameter-specs)))
         (lisp-name (or name (read-from-string function-name))))
    
    `(cffi:defcfun (,function-name ,lisp-name) ,return-type
       ,@parameters)))
	   
(defun add-c-headers (code headers)
  "Add C header includes to code string"
  
  (if headers
    (concatenate 'string
                 (format nil "~{~{#include \"~A\"~}~^~%~}" (mapcar #'list headers))
                 (string #\Newline)
                 code)
				   
    code))
	  
(defmacro function ((&rest compiler-options) c-code &key name declaration include)
  "Define a foreign function with optional compilation and CFFI binding."
  (let ((full-code (add-c-headers c-code (eval include))))
    ;; Compile c code
    (eval `(nnl2.compile:compile ,compiler-options ,full-code))
    
    ;; Return appropriate form
    (if declaration
      declaration
      (generate-cffi-definition c-code :name name))))
	  