#ifndef NNL2_TRANSPOSITION_INPLACE_H
#define NNL2_TRANSPOSITION_INPLACE_H

void nnl2_naive_transposition_inplace(Tensor* tensor) {
	(void)tensor;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for transposition_inplace (view) operation
 * @details
 * Array follows the common backend registration pattern for transposition_inplace (view)
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for transposition_inplace (view)
 * 
 * @see nnl2_naive
 * @see nnl2_naive_transposition_inplace
 */
Implementation transposition_inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_transposition_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for transposition-inplace operation (view)
 * @ingroup backend_system 
 */
transpositioninplacefn nnl2_transposition_inplace;

/** 
 * @brief Makes the transposition-inplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(transposition_inplace);

/** 
 * @brief Sets the backend for transposition-inplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for transposition-inplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_transposition_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transposition_inplace_backends, nnl2_transposition, backend_name, CURRENT_BACKEND(transposition_inplace));
}

/** 
 * @brief Gets the name of the active backend for transposition_inplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_transposition_inplace_backend() {
	return CURRENT_BACKEND(transposition_inplace);
}

/** 
 * @brief Function declaration for getting all available transposition (inplace) backends (view)
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(transposition_inplace);

/**
 * @brief Function declaration for getting the number of available transposition inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transposition_inplace);

#endif /** NNL2_TRANSPOSITION_INPLACE_H **/
