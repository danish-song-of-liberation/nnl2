(in-package :nnl2.system)

(defun greet-the-user ()
  (format t "~%~%~%Welcome to nnl2!
			 ~%This message is informational and appears only once.~%If you want to call it again, write `(nnl2.system:greet-the-user)`
			 ~%You have successfully launched the framework for the first time~%Start working with the framework by reading the documentation at /nnl2/doc (Here's the full path: ~asrc)
			 ~%If you want to help with the project or encounter a problem, please write to issues or nnl.dev@proton.me
			 ~%You can check out the nnl2 benchmarks against torch and numpy in the folder /nnl2/benchmarks (Here's the full path: ~asrc)
			 ~%The framework is under active development, and the greetings are not yet fully complete
			 ~%~%~%" 
			 
	nnl2.intern-system:*current-dir*
	nnl2.intern-system:*current-dir*))
			   
(when *first-launch*	
  (greet-the-user))		   
