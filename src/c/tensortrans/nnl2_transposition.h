#ifndef NNL2_TRANSPOSITION_H
#define NNL2_TRANSPOSITION_H

Tensor* nnl2_naive_transposition(const Tensor* tensor) {
	(void)tensor;
	
	return NULL;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for transposition (view) operation
 * @details
 * Array follows the common backend registration pattern for transposition (view)
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for transposition (view)
 * 
 * @see nnl2_naive
 * @see nnl2_naive_transposition
 */
Implementation transposition_backends[] = {
    REGISTER_BACKEND(nnl2_naive_transposition, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for transposition operation (view)
 * @ingroup backend_system 
 */
transpositionfn nnl2_transposition;

/** 
 * @brief Makes the transposition backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(transposition);

/** 
 * @brief Sets the backend for transposition operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for transposition
 * @see ESET_BACKEND_BY_NAME
 */
void set_transposition_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transposition_backends, nnl2_transposition, backend_name, CURRENT_BACKEND(transposition));
}

/** 
 * @brief Gets the name of the active backend for transposition operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_transposition_backend() {
	return current_backend(transposition);
}

/** 
 * @brief Function declaration for getting all available transposition backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(transposition);

/**
 * @brief Function declaration for getting the number of available transposition backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transposition);

#endif /** NNL2_TRANSPOSITION_H **/
