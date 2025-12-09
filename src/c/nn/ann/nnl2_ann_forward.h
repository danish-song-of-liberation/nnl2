#ifndef NNL2_ANN_FORWARD_H
#define NNL2_ANN_FORWARD_H

#include <stddef.h>

/** @file ann_forward.h
 ** @brief Generic forward pass dispatcher for neural networks
 ** @date 2025
 ** @copyright MIT License
 **/

// Forward declarations
typedef struct nnl2_nn_fnn_struct nnl2_nn_fnn;
typedef struct nnl2_nn_sigmoid_struct nnl2_nn_sigmoid;
typedef struct nnl2_nn_tanh_struct nnl2_nn_tanh;
typedef struct nnl2_nn_sequential_struct nnl2_nn_sequential;

nnl2_ad_tensor* nnl2_nn_fnn_forward(nnl2_nn_fnn* nn, nnl2_ad_tensor* x);
nnl2_ad_tensor* nnl2_nn_sigmoid_forward(nnl2_nn_sigmoid* nn, nnl2_ad_tensor* x);
nnl2_ad_tensor* nnl2_nn_tanh_forward(nnl2_nn_tanh* nn, nnl2_ad_tensor* x);
nnl2_ad_tensor* nnl2_nn_sequential_forward(nnl2_nn_sequential* seq, nnl2_ad_tensor* x);

/** @brief 
 * Perform forward pass on any neural network model
 * 
 ** @param model 
 * Pointer to neural network model
 * 
 ** @param args 
 * Array of arguments (must have at least 1 element)
 *
 ** @return nnl2_ad_tensor* 
 * Output tensor, or NULL on error
 */
nnl2_ad_tensor* nnl2_ann_forward(void* model, void** args) { 
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(model == NULL) {
            NNL2_ERROR("Model pointer is NULL in ann_forward");
            return NULL;
        }
        
        if(args == NULL) {
            NNL2_ERROR("Arguments array is NULL in ann_forward");
            return NULL;
        }
        
        if(args[0] == NULL) {
            NNL2_ERROR("Input tensor (args[0]) is NULL in ann_forward");
            return NULL;
        }
    #endif
	
	nnl2_nn_ann* ann = (nnl2_nn_ann*)model;
	
	// Dispatch based on model type
	switch(ann -> nn_type) {
		case nnl2_nn_type_fnn: {
			nnl2_ad_tensor* input = (nnl2_ad_tensor*)args[0];
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to FNN layer");
            #endif
			
			return nnl2_nn_fnn_forward((nnl2_nn_fnn*)model, input);
		}
		
		case nnl2_nn_type_sigmoid: {
			nnl2_ad_tensor* input = (nnl2_ad_tensor*)args[0];
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to sigmoid layer");
            #endif
			
			return nnl2_nn_sigmoid_forward((nnl2_nn_sigmoid*)model, input);
		}
		
		case nnl2_nn_type_tanh: {
			nnl2_ad_tensor* input = (nnl2_ad_tensor*)args[0];
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to tanh layer");
            #endif
			
			return nnl2_nn_tanh_forward((nnl2_nn_tanh*)model, input);
		}
		
		case nnl2_nn_type_sequential: {
			nnl2_ad_tensor* input = (nnl2_ad_tensor*)args[0];
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to Sequential container");
            #endif
			
			return nnl2_nn_sequential_forward((nnl2_nn_sequential*)model, input);
		}
		
		default: {
            NNL2_ERROR("Unknown or unsupported neural network type: %d", ann -> nn_type);
			return NULL;
		}
	}
}

#endif /** NNL2_ANN_FORWARD_H **/
