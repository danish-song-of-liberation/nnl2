(in-package :nnl2.system)

(defun load-c-library (c-dir)
  (let ((dll-path (uiop:merge-pathnames* "nnl2_core.dll" c-dir)))
    
	(assert (probe-file dll-path) nil (format nil "File ~a is missing" dll-path))

	(cffi:load-foreign-library dll-path)))
		
(let* ((c-dir (uiop:merge-pathnames* "src/c/" *current-dir*)))
  (load-c-library c-dir))  
