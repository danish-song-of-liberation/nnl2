#ifndef NNL2_AD_TRANSPOSITION_H
#define NNL2_AD_TRANSPOSITION_H

/** @file nnl2_ad_transposition.h
 ** @brief AD implementation for O(1) transposition operation (view-based)
 ** @date 2025
 ** @copyright MIT
 **/

static void nnl2_ad_reverse_derivative_transposition(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor);

/** @brief
 * Reverse mode backward pass for view-based transposition operation
 *
 ** @param tensor
 * The output tensor from transposition operation that requires gradient computation
 *
 ** @details
 * For y = T(x), derivative is:
 * dL/dx = T(dL/dy)
 *
 * This implementation uses the same O(1) view logic as the forward pass.
 * Mathematically this is not a “true” transpose — it only swaps
 * shape and stride metadata, keeping memory layout identical.
 *
 * This backward variant is extremely fast but may yield numerically
 * inconsistent results for some elementwise operations.
 *
 * @warning
 * Both forward and backward are mathematically “incorrect” but symmetric,
 * ensuring consistency of the AD graph.
 *
 ** @exception NNL2Error
 * If tensor or its components are NULL under MAX safety mode, function returns early.
 *
 ** @see nnl2_ad_transposition()
 ** @see nnl2_naive_transposition()
 **/
static void nnl2_ad_reverse_backward_transposition(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_transposition, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_transposition, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_transposition, root tensor is NULL");
	#endif

	// Compute derivative and propagate to root tensor
	nnl2_ad_reverse_derivative_transposition(tensor, tensor->roots[0]);

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief
 * Create an automatic differentiation tensor for O(1) transposition (view-based)
 *
 ** @param ad_tensor
 * Input AD tensor
 *
 ** @param ad_mode
 * Automatic differentiation mode (reverse, p1, p2, etc.)
 *
 ** @param track_graph
 * Whether to record this operation in computation graph
 *
 ** @return nnl2_ad_tensor*
 * New AD tensor representing transposed view of input, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if any input argument is invalid under the current safety mode.
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor or metadata arrays.
 *
 ** @exception NNL2Error
 * Returns NULL if the transposition view creation fails.
 *
 ** @warning
 * This transposition is O(1) — no memory is copied.
 * The result shares the same data buffer as the input.
 *
 ** @see nnl2_naive_transposition()
 ** @see nnl2_ad_reverse_derivative_transposition()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_transposition(nnl2_ad_tensor* ad_tensor, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	// Basic null check
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_transposition, passed AD tensor is NULL", NULL);
	#endif

	// Full structure validation
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data, "In function nnl2_ad_transposition, passed AD tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_transposition, passed AD tensor shape is NULL", NULL);
	#endif

	nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	if(!result) {
		NNL2_MALLOC_ERROR();
		return NULL;
	}
	result->magic_number = TENSOR_MAGIC_ALIVE;

	// Forward computation (view creation)
	result->data = nnl2_transposition(ad_tensor->data);
	if(!result->data) {
		NNL2_ERROR("In function nnl2_ad_transposition, failed to create transposed view using nnl2_naive_transposition()");
		free(result);
		return NULL;
	}

	// Gradient tensor (same shape as transposed result)
	result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
	if(!result->grad) {
		NNL2_ERROR("In function nnl2_ad_transposition, failed to allocate gradient tensor via nnl2_empty()");
		nnl2_free_tensor(result->data);
		free(result);
		return NULL;
	}

	// Computational graph
	if(track_graph) {
		result->num_roots = 1;
		result->roots = (nnl2_ad_tensor**)malloc(sizeof(*result->roots));
		if(!result->roots) {
			NNL2_MALLOC_ERROR();
			nnl2_free_tensor(result->data);
			nnl2_free_tensor(result->grad);
			free(result);
			return NULL;
		}

		result->roots[0] = ad_tensor;

		switch(ad_mode) {
			case nnl2_ad_reverse_mode:
				result->backward_fn = nnl2_ad_reverse_backward_transposition;
				break;

			default:
				NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
				nnl2_free_ad_tensor(result);
				return NULL;
		}
	} else {
		result->num_roots = 0;
		result->roots = NULL;
		result->backward_fn = NULL;
	}

	// Metadata
	result->requires_grad = ad_tensor->requires_grad;
	result->grad_initialized = false;
	result->is_leaf = false;
	result->name = NULL;
	result->ts_type = nnl2_type_ad;
	
	result -> extra_field = NULL;
	result -> extra_free = NULL;

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif

	return result;
}

#endif /** NNL2_AD_TRANSPOSITION_H **/
