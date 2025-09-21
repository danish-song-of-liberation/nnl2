(in-package :nnl2.backends)

(defun get-test-path (dir-path test &key (type "c"))
  (handler-case 
      (let ((path (uiop:merge-pathnames* 
					(make-pathname :directory '(:relative "src" "c" "backends_tests")
								   :name test
								   :type type)
								   
					dir-path)))
					
		path)	
		
	(error (e)
	  (error "(~a): ~a~%" #'get-test-h e))))
	  