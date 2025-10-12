#ifndef NNL2_FOREIGN_LOG_H
#define NNL2_FOREIGN_LOG_H

/**
 * @file lisp_logging.c
 * @brief Logging functions for Lisp system integration
 * @date 2025
 * @copyright MIT License
 */

/** @brief 
 * Log an error message from Lisp code
 *
 ** @param msg 
 * The error message to log
 */
void lisp_call_error(char* msg) {
	NNL2_ERROR(msg);
}

/** @brief 
 * Log a warning message from Lisp code
 *
 ** @param msg 
 * The warning message to log
 */
void lisp_call_warning(char* msg) {
	NNL2_WARN(msg);
}

/** @brief 
 * Log a fatal error message from Lisp code
 *
 ** @param msg 
 * The fatal error message to log
 *
 ** @warning
 * Immediately terminates the program execution
 */
void lisp_call_fatal(char* msg) {
	NNL2_FATAL(msg);
}

/** @brief 
 * Log a debug message from Lisp code
 *
 ** @param msg 
 * The debug message to log
 */
void lisp_call_debug(char* msg) {
	NNL2_DEBUG(msg);
}

/** @brief 
 * Log an info message from Lisp code
 *
 ** @param msg 
 * The info message to log
 */
void lisp_call_info(char* msg) {
	NNL2_INFO(msg);
}

#endif
