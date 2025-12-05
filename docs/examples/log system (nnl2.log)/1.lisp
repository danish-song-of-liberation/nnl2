(ql:quickload :nnl2)

;; (nnl2.log:fatal "hello world") Exits the program

(nnl2.log:error "hello world")
(nnl2.log:warning "hello world")
(nnl2.log:info "hello world")
(nnl2.log:debug "hello world")

(nnl2.log:info "2 + 3 = ~d" (+ 2 3))
