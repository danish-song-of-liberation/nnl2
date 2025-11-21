#ifndef NNL2_OPTIM_ZERO_GRAD_H
#define NNL2_OPTIM_ZERO_GRAD_H

/** @file nnl2_optim_zero_grad.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains a zero gradient function
 **/
 
/** @brief 
 * Zero out gradients for any optimizer type
 * 
 ** @param optim 
 * Generic pointer to the optimizer object
 */
void nnl2_optim_zero(void* optim) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (optim == NULL) return;
	#endif

	// Cast to optimizer type pointer to access the type identifier
	nnl2_optim_object_type* type_optim = ((nnl2_optim_object_type *) optim);
	
	switch(*type_optim) {
		case nnl2_optim_type_gd: {
			// Call Gradient Descent optimizer zero gradient function
			nnl2_optim_gd_zero_grad(optim);
			break;
		}
		
		case nnl2_optim_type_unknown:	
		
		default: {
			// Handle unknown optimizer types with error reporting
			NNL2_OPTIM_TYPE_ERROR(*type_optim);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_OPTIM_ZERO_GRAD_H **/
