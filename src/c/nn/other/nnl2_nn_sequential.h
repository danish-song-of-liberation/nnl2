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

/** @brief 
 * Frees all resources associated with a sequential neural network
 * 
 ** @param seq 
 * Pointer to sequential container to free
 *
 ** @note 
 * Each layer is freed using the generic nnl2_ann_free function
 */
void nnl2_nn_sequential_free(nnl2_nn_sequential* seq) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
		NNL2_INFO("Freeing sequential network: %p", seq);
    #endif
	
	if (seq == NULL) return;
	
	for(size_t layer_idx = 0; layer_idx < seq->num_layers; layer_idx++) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_INFO("(Sequential) Freeing %zu layer: %p (nn_type: %d)", layer_idx, seq -> layers[layer_idx], *((nnl2_nn_type*)(seq -> layers[layer_idx]))); 
		#endif 
		
        if(seq->layers[layer_idx] != NULL) {
            nnl2_ann_free(seq->layers[layer_idx]);
        }
		
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_INFO("(Sequential) %zu layer succesfully freed"); 
		#endif 
    }
	
	free(seq->layers);
	free(seq);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_INFO("Succesfully freed sequential network");
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
            continue;  // Skip NULL 
        }
		
        nnl2_ad_tensor** layer_params = nnl2_ann_parameters(seq->layers[layer_idx]);
		if(layer_params == NULL) {
            continue;  // Layer has no parameters
        }
        
        size_t layer_num_params = nnl2_ann_num_parameters(seq->layers[layer_idx]);
            
        for(size_t i = 0; i < layer_num_params; i++) {
            params[param_index++] = layer_params[i];
        }

		if(layer_params) free(layer_params);
    }
		
	if(param_index != total_params) {
        NNL2_WARN("Parameter count mismatch: expected %zu, got %zu", total_params, param_index);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return params;
}

/** @brief Prints an ANN model of any supported type **/
void nnl2_ann_print(void* nn, bool terpri, int depth);

/** @brief 
 * Print sequential neural network in a formatted tree structure
 *
 ** @param nn 
 * Pointer to sequential network structure
 *
 ** @param depth 
 * Indentation depth for pretty printing
 */
void nnl2_nn_sequential_print(nnl2_nn_sequential* nn, int depth) {
    if(!nn) {
        printf("(sequential NULL)\n");
        return;
    }
    
    if(depth < 0) depth = 0;
    
    printf("(sequential\n");
    
    size_t last_idx = nn->num_layers - 1;
    
    for(size_t layer_idx = 0; layer_idx < nn->num_layers; layer_idx++) {
        // Indentation
        for(int i = 0; i < depth; i++) 
            printf("  ");
        
        // Layer index
        printf("%zu=", layer_idx);
        
        // Print layer (add newline for all but last)
        bool add_newline = (layer_idx != last_idx);
        nnl2_ann_print(nn->layers[layer_idx], add_newline, depth + 1);
    }
    
    // Final closing bracket with newlines
    printf(")\n\n");
}

nnl2_nnlrepr_template* nnl2_nn_sequential_nnlrepr_template(nnl2_nn_sequential* nn) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(nn == NULL) {
            NNL2_ERROR("In function nnl2_nn_sequential_nnlrepr_template, sequential pointer is NULL");
            return NULL;
        }
    #endif 
	
	nnl2_nnlrepr_template* result = malloc(sizeof(nnl2_nnlrepr_template));
	
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
	
	// Common metadata
    result->nn_type = nnl2_nn_type_sequential;
    result->num_shapes = 0;
    result->vector_size = 0;
    result->num_childrens = nn->num_layers;
    result->shapes = NULL;
    result->additional_data = NULL;
    result->childrens = NULL;
	
	if(result->num_childrens == 0) {
        result->childrens = NULL;
		
        #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
            NNL2_INFO("Created empty sequential nnlrepr template");
            NNL2_FUNC_EXIT();
        #endif
		
        return result;
    }
	
	result->childrens = malloc(sizeof(nnl2_nnlrepr_template*) * result->num_childrens);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result->childrens == NULL) {
            NNL2_MALLOC_ERROR();
            free(result);
            return NULL;
        }
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
            for(size_t i = 0; i < result->num_childrens; i++) {
                result->childrens[i] = NULL;
            }
        #endif
    #endif
	
	// Create templates for each layer
    bool allocation_failed = false;
    
    for(size_t it = 0; it < result->num_childrens; it++) {
        result->childrens[it] = nnl2_ann_nnlrepr_template(nn->layers[it]);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(result->childrens[it] == NULL) {
                NNL2_ERROR("Failed to create template for layer %zu", it);
                allocation_failed = true;
                break;
            }
        #endif
    }

    if(allocation_failed) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            for(size_t i = 0; i < result->num_childrens; i++) {
                if(result->childrens[i] != NULL) {
                    nnl2_nnlrepr_template_free(result->childrens[i]);
                }
            }
			
            free(result->childrens);
            free(result);
        #endif
		
        return NULL;
    } 
	
	result->vector_size = 0;
    for(size_t i = 0; i < result->num_childrens; i++) {
        result->vector_size += result->childrens[i]->vector_size;
    }
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("Created sequential nnlrepr template with %zu layers, total vector size: %zu", 
                  result->num_childrens, result->vector_size);
				  
        NNL2_FUNC_EXIT();
    #endif
	
	return result;
}

#endif /** NNL2_NN_SEQUENTIAL_H **/
