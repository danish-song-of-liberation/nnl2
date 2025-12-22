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
typedef struct nnl2_nn_relu_struct nnl2_nn_relu;
typedef struct nnl2_nn_leaky_relu_struct nnl2_nn_leaky_relu;
typedef struct nnl2_nn_sequential_struct nnl2_nn_sequential;
typedef struct nnl2_nn_rnn_cell_struct nnl2_nn_rnn_cell;

nnl2_ad_tensor* nnl2_nn_sigmoid_forward(nnl2_nn_sigmoid* nn, nnl2_ad_tensor* x);
nnl2_ad_tensor* nnl2_nn_tanh_forward(nnl2_nn_tanh* nn, nnl2_ad_tensor* x);
nnl2_ad_tensor* nnl2_nn_relu_forward(nnl2_nn_relu* nn, nnl2_ad_tensor* x);
nnl2_ad_tensor* nnl2_nn_leaky_relu_forward(nnl2_nn_leaky_relu* nn, nnl2_ad_tensor* x);
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
    #endif
    
    nnl2_nn_ann* ann = (nnl2_nn_ann*)model;

    void* safe_args[2] = { args[0], args[1] };
    
    // Dispatch based on model type
    switch(ann -> nn_type) {
        case nnl2_nn_type_fnn: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(safe_args[0] == NULL) {
                    NNL2_ERROR("Input tensor is NULL for FNN layer");
                    return NULL;
                }
            #endif
            
            nnl2_ad_tensor* input = (nnl2_ad_tensor*)safe_args[0];
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to FNN layer");
            #endif
            
            nnl2_nn_fnn* fnn = (nnl2_nn_fnn*)model;
			return fnn->forward(fnn, input);
        }
        
        case nnl2_nn_type_rnn_cell: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if(safe_args[0] == NULL) {
					NNL2_ERROR("Input tensor is NULL for rnncell layer");
					return NULL;
				}
				
				if(safe_args[1] == NULL) {
					NNL2_ERROR("Hidden state tensor is NULL for rnncell layer");
					return NULL;
				}
			#endif
			
			nnl2_ad_tensor* input  = (nnl2_ad_tensor*)safe_args[0];
			nnl2_ad_tensor* hidden = (nnl2_ad_tensor*)safe_args[1];
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Dispatching forward to rnncell layer");
			#endif
			
			nnl2_nn_rnn_cell* cell = (nnl2_nn_rnn_cell*)model;
			return cell->forward(cell, input, hidden);
		}
		
		case nnl2_nn_type_rnn: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if(safe_args[0] == NULL) {
					NNL2_ERROR("Input tensor is NULL for RNN layer");
					return NULL;
				}
			#endif 
			
			nnl2_ad_tensor* input = (nnl2_ad_tensor*)safe_args[0];
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Dispatching forward to RNN layer");
			#endif
			
			nnl2_nn_rnn* rnn = (nnl2_nn_rnn*)model;
			return rnn->forward(rnn, input);
		}
        
        case nnl2_nn_type_sigmoid: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(safe_args[0] == NULL) {
                    NNL2_ERROR("Input tensor is NULL for sigmoid layer");
                    return NULL;
                }
            #endif
            
            nnl2_ad_tensor* input = (nnl2_ad_tensor*)safe_args[0];
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to sigmoid layer");
            #endif
            
            return nnl2_nn_sigmoid_forward((nnl2_nn_sigmoid*)model, input);
        }
        
        case nnl2_nn_type_tanh: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(safe_args[0] == NULL) {
                    NNL2_ERROR("Input tensor is NULL for tanh layer");
                    return NULL;
                }
            #endif
            
            nnl2_ad_tensor* input = (nnl2_ad_tensor*)safe_args[0];
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to tanh layer");
            #endif
            
            return nnl2_nn_tanh_forward((nnl2_nn_tanh*)model, input);
        }
        
        case nnl2_nn_type_relu: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(safe_args[0] == NULL) {
                    NNL2_ERROR("Input tensor is NULL for relu layer");
                    return NULL;
                }
            #endif
            
            nnl2_ad_tensor* input = (nnl2_ad_tensor*)safe_args[0];
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to relu layer");
            #endif
            
            return nnl2_nn_relu_forward((nnl2_nn_relu*)model, input);
        }
        
        case nnl2_nn_type_leaky_relu: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(safe_args[0] == NULL) {
                    NNL2_ERROR("Input tensor is NULL for leaky_relu layer");
                    return NULL;
                }
            #endif
            
            nnl2_ad_tensor* input = (nnl2_ad_tensor*)safe_args[0];
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("Dispatching forward to leaky_relu layer");
            #endif
            
            return nnl2_nn_leaky_relu_forward((nnl2_nn_leaky_relu*)model, input);
        }
        
        case nnl2_nn_type_sequential: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(safe_args[0] == NULL) {
                    NNL2_ERROR("Input tensor is NULL for Sequential container");
                    return NULL;
                }
            #endif
            
            nnl2_ad_tensor* input = (nnl2_ad_tensor*)safe_args[0];
            
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
