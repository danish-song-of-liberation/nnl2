(in-package :nnl2.fusion)

;; NNL2

;; Filepath: nnl2/src/lisp/fusion/fusion-rules.lisp
;; File: fusion-rules.lisp

;; File contains definition of fusion rules

;; In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
;; nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2

(add-rule (list '.+ '?a (list '.* '?b '?c=real)) '(nnl2.hli.ts:axpy ?a ?b :alpha ?c))
(add-rule (list '+= '?a (list '.* '?b '?c=real)) '(nnl2.hli.ts:axpy! ?a ?b :alpha ?c))

(add-rule '(nnl2.hli.ts:tlet ((?a (nnl2.hli.ts:.+ ?b (nnl2.hli.ts:.* ?c ?d=real)))) &body) 
          '(nnl2.hli.ts:tlet ((?a (nnl2.hli.ts:axpy ?b ?c :alpha ?d))) &body))
		  
(add-rule '(nnl2.hli.ts:tlet* ((?a (nnl2.hli.ts:.* ?b ?c)) (?d (nnl2.hli.ts:.+ ?f ?a))) &body)
          '(nnl2.hli.ts:tlet ((?d (nnl2.hli.ts:axpy ?f ?b :alpha ?c))) &body))

(add-rule '(nnl2.hli.ts:tlet ((?a (nnl2.hli.ts:.* ?d ?b=real))) (nnl2.hli.ts:+= ?c ?a))
		  '(nnl2.hli.ts:axpy! ?c ?d :alpha ?b))

(add-rule '(nnl2.hli.ts:tlet ((?a (nnl2.hli.ts:.- ?b (nnl2.hli.ts:.* ?c ?d=real)))) &body) 
          '(nnl2.hli.ts:tlet ((?a (nnl2.hli.ts:axpy ?b ?c :alpha (- ?d)))) &body))
		  
(add-rule '(nnl2.hli.ts:tlet* ((?a (nnl2.hli.ts:.* ?b ?c)) (?d (nnl2.hli.ts:.- ?f ?a))) &body)
          '(nnl2.hli.ts:tlet ((?d (nnl2.hli.ts:axpy ?f ?b :alpha (- ?c)))) &body))

(add-rule '(nnl2.hli.ts:tlet ((?a (nnl2.hli.ts:.* ?d ?b=real))) (nnl2.hli.ts:-= ?c ?a))
		  '(nnl2.hli.ts:axpy! ?c ?d :alpha (- ?b)))

		  