(in-package :nnl2.backends)

;; NNL2

;; Filepath: nnl2/src/lisp/backend-status/backend-path-util.lisp
;; File: backend-path-util.lisp

;; Contains the get-test-path function 

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(defun get-test-path (dir-path test &key (type "c"))
  "Returns the full path to the test file in the project structure
  
   dir-path: Base path to the project's root directory
   test: Name of the test file (without extension)
   type (key): File extension (default .c)
   
   Returns full path to the test file
   
   Example 1: (get-test-path #P\"/bar/baz/\" \"foo\") -> #P\"/bar/baz/src/c/backends_tests/foo.c\"
   Example 2: (get-test-path #P\"/qux/quux/\" \"garple\" :type \"h\") -> #P\"/qux/quux/src/c/backends_tests/garple.h\""
   
  (handler-case 
      (let ((path (uiop:merge-pathnames* 
					;; Creating a relative path to a test file
					(make-pathname :directory '(:relative "src" "c" "backends_tests")
								   :name test
								   :type type)
								   
					dir-path)))
							
		path)	
		
	(error (e)
	  (error "(~a): ~a~%" #'get-test-h e))))
	  