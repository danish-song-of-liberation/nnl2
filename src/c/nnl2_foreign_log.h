#ifndef NNL2_FOREIGN_LOG_H
#define NNL2_FOREIGN_LOG_H

void lisp_call_error(char* msg) {
	NNL2_ERROR(msg);
}

void lisp_call_warning(char* msg) {
	NNL2_WARN(msg);
}

void lisp_call_fatal(char* msg) {
	NNL2_FATAL(msg);
}

void lisp_call_debug(char* msg) {
	NNL2_DEBUG(msg);
}

void lisp_call_info(char* msg) {
	NNL2_INFO(msg);
}

#endif
