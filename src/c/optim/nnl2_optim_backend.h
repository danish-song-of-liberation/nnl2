#ifndef NNL2_OPTIM_BACKEND_H
#define NNL2_OPTIM_BACKEND_H

// NNL2

/** @file nnl2_optim_backend.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Сontains a common structure for other optimizers
 **/

/** @brief 
 * Macro to generate error message for unknown optimizer types
 *
 ** @param optim_type 
 * The unknown optimizer type that was received
 */  
#define NNL2_OPTIM_TYPE_ERROR(optim_type) NNL2_ERROR("Unknown optimizer type received (in enum: %d)", optim_type)

typedef enum {
	nnl2_optim_type_gd,			///< Gradient Descent optimizer
	nnl2_optim_type_unknown		///< Unknown optimizer type
} nnl2_optim_object_type;

typedef struct {
	nnl2_ad_tensor** tensors;	///< Array of pointers to tensors to be optimized
	size_t num_tensors;			///< Number of tensors in the array
} nnl2_optim;

/** @brief 
 * Zero out the gradients of all tensors in the optimizer
 *  
 ** @param optim 
 * The optimizer instance containing the tensors
 */
inline static void nnl2_optim_zero_grad(nnl2_optim optim) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	for(size_t it = 0; it < optim . num_tensors; it++) {
		nnl2_ad_zero_grad(optim . tensors [it]);
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
}

#endif /** NNL2_OPTIM_BACKEND_H **/
 