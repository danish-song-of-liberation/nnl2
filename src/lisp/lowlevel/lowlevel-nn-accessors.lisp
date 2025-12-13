(in-package :nnl2.lli.nn)

(defmacro fnn (in-features arrow out-features &key w b (handle-as :copy) (bias t) (dtype `(nnl2.hli.ad:dtype ,w)))
  (declare (ignore arrow))
  `(nnl2.ffi:%nn-manual-fnn ,in-features ,out-features ,bias ,dtype ,(when `,w `,w) ,(when `,b `,b) ,handle-as))
  
(defmacro rnncell (input-size arrow hidden-size &key wxh whh bxh bhh (handle-as :copy) (bias t) (dtype `(nnl2.hli.ad:dtype ,wxh)))
  (declare (ignore arrow))
  `(nnl2.ffi:%nn-manual-rnn-cell ,input-size ,hidden-size ,bias ,dtype ,(when wxh `,wxh) ,(when whh `,whh) ,(when (and bias bxh) `,bxh) ,(when (and bias bhh) `,bhh) ,handle-as))  
  