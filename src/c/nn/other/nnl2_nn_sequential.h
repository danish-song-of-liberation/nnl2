#ifndef NNL2_NN_SEQUENTIAL_H
#define NNL2_NN_SEQUENTIAL_H

#include <stdlib.h>
#include <stddef.h>

/** @file nn_sequential.h
 ** @brief Sequential neural network container implementation
 ** @date 2025
 ** @copyright MIT License
 **/



///@{ [nnl2_nn_sequential]

/** @struct nnl2_nn_sequential_struct **/
typedef struct nnl2_nn_sequential_struct {
	nnl2_nn_ann metadata;   ///< Base neural network metadata 
	void** layers;			///< Array of layer pointers
	size_t num_layers;		///< Number of layers 
} nnl2_nn_sequential;

///@} [nnl2_nn_sequential]



/** @brief Safely frees an ANN model of any supported type **/
void nnl2_ann_free(void* nn);

/** @brief Returns the number of parameters in an ANN model **/
size_t nnl2_ann_num_parameters(void* nn);

/** @brief Retrieves all parameters from an ANN model **/
nnl2_ad_tensor** nnl2_ann_parameters(void* nn);

/** @brief 
 * Creates a new sequential neural network container
 * 
 ** @param num_layers 
 * Number of layers in the sequential model
 *
 ** @param layers 
 * Array of layer pointers (each must be a valid neural network layer)
 * 
 ** @return nnl2_nn_sequential* 
 * Pointer to newly created sequential container
 */
nnl2_nn_sequential* nnl2_nn_sequential_create(size_t num_layers, void** layers) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(layers == NULL) {
			NNL2_ERROR("Layers array cannot be NULL in nnl2_nn_sequential_creates");
			return NULL;
		}
		
		if(num_layers == 0)  NNL2_WARN("Creating sequential model with 0 layers");
	#endif
	
	nnl2_nn_sequential* seq = malloc(sizeof(nnl2_nn_sequential));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!seq) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif
	
    // Initialize metadata
	seq -> metadata.nn_type = nnl2_nn_type_sequential;
	
    // Store layers
	seq -> layers = layers;
	seq -> num_layers = num_layers;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return seq;
}

/**
 * @brief Frees all resources associated with a sequential neural network
 * 
 * @param seq Pointer to sequential container to free
 * 
 * @note This function safely handles NULL pointers
 * @note Each layer is freed using the generic nnl2_ann_free function
 */
void nnl2_nn_sequential_free(nnl2_nn_sequential* seq) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	if (seq == NULL) return;
	
	for(size_t layer_idx = 0; layer_idx < seq->num_layers; layer_idx++) {
        if(seq->layers[layer_idx] != NULL) {
            nnl2_ann_free(seq->layers[layer_idx]);
        }
    }
	
	free(seq->layers);
	free(seq);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Performs forward pass through the sequential network
 * 
 ** @param seq 
 * Sequential container to process through
 *
 ** @param x 
 * Input tensor
 *
 ** @return nnl2_ad_tensor* 
 * Output tensor after passing through all layers
 *
 ** @warning 
 * caller is responsible for freeing the returned tensor
 */
nnl2_ad_tensor* nnl2_nn_sequential_forward(nnl2_nn_sequential* seq, nnl2_ad_tensor* x) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(seq == NULL) {
			NNL2_ERROR("Sequential container (seq) is NULL in forward pass");
			return NULL;
		}
		
		if(x == NULL) {
			NNL2_ERROR("Input tensor (x) is NULL in sequential forward pass");
			return NULL;
		}
	#endif
	
	nnl2_ad_tensor* current = x;
	
	// Through each layer sequentially
	for(size_t it = 0; it < seq -> num_layers; it++) {
		if(seq->layers[it] == NULL) {
            NNL2_ERROR("Layer %zu is NULL in sequential forward pass", it);
            return NULL;
        }
		
		void* args[1] = { (void*)current };
		nnl2_ad_tensor* next = nnl2_ann_forward(seq -> layers[it], args);
		
		if(next == NULL) {
            NNL2_ERROR("Sequential forward pass failed at layer %zu", it);
            return NULL;
        }
		
		// Update current tensor for next layer
		current = next;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return current;
}

/** @brief 
 * Calculates total number of parameters in the sequential network
 * 
 ** @param seq 
 * Sequential container to analyze
 *
 ** @return size_t 
 * Total number of trainable parameters across all layers
 * 
 ** @note 
 * Returns 0 if seq is NULL or has 0 layers
 */
size_t nnl2_nn_sequential_get_num_parameters(nnl2_nn_sequential* seq) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(seq == NULL) {
			NNL2_WARN("NULL sequential container in get_num_parameters. returning 0");
			return 0;
		}
	#endif
	
	size_t acc = 0;
	
	for(size_t it = 0; it < seq -> num_layers; it++) 
		acc += nnl2_ann_num_parameters(seq -> layers[it]);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return acc;
}

/** @brief 
 * Retrieves all parameters from all layers in the sequential network
 * 
 ** @param seq 
 * Input sequential container
 *
 ** @return nnl2_ad_tensor** 
 * Array of parameter tensors (dynamically allocated)
 * 
 ** @note 
 * The returned array must be freed by the caller
 */
nnl2_ad_tensor** nnl2_nn_sequential_get_parameters(nnl2_nn_sequential* seq) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	size_t total_params = nnl2_nn_sequential_get_num_parameters(seq);
	
	nnl2_ad_tensor** params = malloc(sizeof(nnl2_ad_tensor*) * total_params);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(params == NULL) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif
	
	size_t param_index = 0;
	
	// Collect parameters from each layer
    for(size_t layer_idx = 0; layer_idx < seq->num_layers; layer_idx++) {
		if(seq->layers[layer_idx] == NULL) {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
				NNL2_WARN("Layer %zu is NULL, skipping", layer_idx);
			#endif
			
            continue;  // Skip NULL 
        }
		
        nnl2_ad_tensor** layer_params = nnl2_ann_parameters(seq->layers[layer_idx]);
		if (layer_params == NULL) {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
				NNL2_WARN("Layer %zu has no parameters, skipping", layer_idx);
			#endif
			
            continue;  // Layer has no parameters
        }
        
        size_t layer_num_params = nnl2_ann_num_parameters(seq->layers[layer_idx]);
            
        for(size_t i = 0; i < layer_num_params; i++) {
            params[param_index++] = layer_params[i];
        }

        free(layer_params);
    }
		
	if(param_index != total_params) {
        NNL2_WARN("Parameter count mismatch: expected %zu, got %zu", total_params, param_index);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return params;
}

#endif /** NNL2_NN_SEQUENTIAL_H **/
