// NNL2

/** @file nnl2_backend_system_docs.h
 ** @brief Common documentation for the backend system
 *
 ** Filepath: nnl2/src/c/nnl2_backend_system_docs.h
 ** File: nnl2_backend_system_docs.h
 **
 ** The file contains the documentation of the backend system
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/
 
/** @defgroup backend_system
 * Backend System
 * 
 ** @brief Documentation for the backend registration and selection
 **
 ** @details
 * The default implementation of any function uses the 
 * Implementation* type (instead of a list pointer, that 
 * is, Implementation[]). 
 *
 * All function implementations 
 * are passed to it as pointers, along with their priority 
 * and backend name. Later, automatic dispatching is performed, 
 * the best backend is selected, and manual chaining functions 
 * are set, such as chaining the current backend for addition.
 **
 **
 ** @see REGISTER_BACKEND
 ** @see SET_BACKEND_BY_NAME
 ** @see MAKE_CURRENT_BACKEND
 ** @see CURRENT_BACKEND
 ** @see NAIVE_BACKEND_NAME
 ** @see AVX128_BACKEND_NAME
 ** @see AVX256_BACKEND_NAME
 ** @see BLAS_BACKEND_NAME "BLAS"
 */
 