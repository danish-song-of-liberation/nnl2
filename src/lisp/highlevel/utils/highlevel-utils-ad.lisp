(in-package :nnl2.hli.ad.utils)

(defun narrow (ad-tensor &key (dim 0) len start (track-graph nnl2.system:*ad-default-track-graph*))
  (nnl2.ffi:%ad-narrow ad-tensor dim start len nnl2.ffi:ad-reverse-mode track-graph))
  