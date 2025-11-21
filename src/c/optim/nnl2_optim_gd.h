#ifndef NNL2_OPTIM_GD_H
#define NNL2_OPTIM_GD_H

// NNL2

/** @file nnl2_optim_gd.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Сontains a common gradient descent structure
 **/
 
typedef struct {
	nnl2_optim_object_type optim_type;		///< Type identifier for the optimizer 
	nnl2_optim data;						///< Optimizer data containing tensors
	nnl2_float32 lr;						///< Learning rate 
} nnl2_optim_gd;

/** @brief 
 * Free the memory allocated for a Gradient Descent optimizer
 *
 ** @note 
 * This does not free the tensors themselves, only the optimizer structure
 * 
 ** @param optim 
 * Pointer to the Gradient Descent optimizer to free
 */
static void nnl2_optim_gd_free(nnl2_optim_gd* optim) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!optim) {
			NNL2_ERROR("In function nnl2_optim_gd_free, nnl2_optim_gd* optim is NULL");
			return;
		}
	#endif
	
	free(optim);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
}

/** @brief 
 * Zero out gradients for all tensors in the Gradient Descent optimizer
 *
 ** @param optim 
 * Pointer to the Gradient Descent optimizer
 */
static void nnl2_optim_gd_zero_grad(nnl2_optim_gd* optim) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!optim) {
			NNL2_ERROR("In function nnl2_optim_gd_zero_grad, nnl2_optim_gd* optim is NULL");
			return;
		}
	#endif
	
	nnl2_optim_zero_grad(optim -> data);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
}

/** @brief 
 * Perform a single optimization step using Gradient Descent
 * Gradient Descent: θ = θ - η * ▽θ
 * 
 * @param gd_optim 
 * Pointer to the Gradient Descent optimizer
 */
static void nnl2_optim_gd_step(nnl2_optim_gd* gd_optim) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!gd_optim) {
			NNL2_ERROR("In function nnl2_optim_gd_step, nnl2_optim_gd* gd_optim is NULL");
			return;
		}
	#endif
	
	for(size_t it = 0; it < gd_optim -> data . num_tensors; it++) {
		nnl2_ad_step_inplace(gd_optim -> data . tensors [it], gd_optim -> lr);
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
}

/** @brief 
 * Create a new Gradient Descent optimizer
 * 
 ** @param tensors 
 * Array of pointers to tensors to be optimized
 *
 ** @param num_tensors 
 * Number of tensors in the array
 *
 ** @param learning_rate 
 * Learning rate for the optimizer
 * 
 ** @return nnl2_optim_gd*
 * Pointer to the newly created Gradient Descent optimizer
 */
nnl2_optim_gd* nnl2_optim_gd_create(nnl2_ad_tensor** tensors, size_t num_tensors, float learning_rate) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!tensors) {
			NNL2_ERROR("In function nnl2_optim_gd_create, nnl2_ad_tensor** tensors is NULL");
			return NULL;
		}
	#endif
	
	nnl2_optim_gd* optim = malloc(sizeof(nnl2_optim_gd));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!optim) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif
	
	// Initializing metadata
	optim -> lr = learning_rate;
	optim -> optim_type = nnl2_optim_type_gd;
	optim -> data . tensors = tensors;
	optim -> data . num_tensors = num_tensors;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif 
	
	return optim;
}

#endif /** NNL2_OPTIM_GD_H **/
