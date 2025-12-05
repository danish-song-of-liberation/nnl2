(ql:quickload :nnl2)

(use-package :nnl2.hli.ts)

(nnl2.compile.aot:function (:optimize 3)
  "void bar(nnl2_tensor* tensor) {
       size_t numel = nnl2_numel(tensor);

       switch(tensor -> dtype) {
           case FLOAT64: {
               nnl2_float64* data = (nnl2_float64*)tensor -> data;

               for(size_t it = 0; it < numel; it++)
                   data[it] = sin(data[it]);

               return;
           }

           default: {
               NNL2_TYPE_ERROR(tensor -> dtype);
               return;
           }
       }
  }" :include '("math.h" "../src/c/nnl2_core.c" "../src/c/nnl2/friendly.h"))

(tlet* ((a (ones #(5 5) :dtype :float64)))
  (bar a)
  (print-tensor a))
  