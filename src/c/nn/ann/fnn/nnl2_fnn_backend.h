#ifndef NNL2_FNN_BACKEND_H
#define NNL2_FNN_BACKEND_H

// NNL2

/** @file nnl2_fnn_backend.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Fully-Connected Neural Network (FNN) Backend
 *
 * Сontains the entire basic backend for fnn, including 
 * the structure, allocators, release functions, and 
 * other auxiliary functions
 *
 ** Filepath: nnl2/src/c/nn/ann/fnn/nnl2_fnn_backend.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

///@{ [nnl2_nn_fnn]

/** @brief Fully-Connected Neural Network (FNN) layer structure 
 ** @see nnl2_nn_ann 
 ** @extends nnl2_nn_ann **/

typedef struct nnl2_nn_fnn_struct {
	nnl2_nn_ann metadata;		///< Base neural network metadata and type information
	nnl2_ad_tensor* weights;    ///< Weight matrix of shape [in_features, out_features]
	nnl2_ad_tensor* bias;		///< Bias vector of shape [out_features]. May be NULL if use_bias is set to false in metadata
	
	nnl2_ad_tensor* (*forward)( 		///< Forward propagation function pointer (for with_bias/no_bias)
        struct nnl2_nn_fnn_struct* nn,
        nnl2_ad_tensor* x
    );   
} nnl2_nn_fnn;

///@} [nnl2_nn_fnn]



/** @brief Initializes FNN layer with zeros **/
static bool fnn_init_zeros(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype);

/** @brief Initializes FNN layer with uniform random values **/
static bool fnn_init_rand(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype);

/** @brief Initializes FNN layer with normal random values **/
static bool fnn_init_randn(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype);

/** @brief Initializes FNN layer with Xavier normal initialization **/
static bool fnn_init_xavier_normal(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype);

/** @brief Initializes FNN layer with Xavier uniform initialization **/
static bool fnn_init_xavier_uniform(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype);

/** @brief Initializes FNN layer with Kaiming normal initialization **/
static bool fnn_init_kaiming_normal(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype);

/** @brief Initializes FNN layer with Kaiming uniform initialization **/
static bool fnn_init_kaiming_uniform(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype);

/** @brief Initializes FNN layer with identity matrix for testing **/
static bool fnn_init_identity(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype, bool use_bias);

/** @brief Performs forward pass of FNN layer using weights and bias */
static nnl2_ad_tensor* fnn_forward_with_bias(nnl2_nn_fnn* nn, nnl2_ad_tensor* x);

/** @brief Performs forward pass of FNN layer using weights only (no bias) */
static nnl2_ad_tensor* fnn_forward_no_bias(nnl2_nn_fnn* nn, nnl2_ad_tensor* x);

/** @brief 
 * Creates a Fully-Connected Neural Network (FNN) layer
 *
 ** @param in_features 
 * The number of input units/features
 *
 ** @param out_features 
 * The number of output units/features
 *
 ** @param use_bias 
 * If true, a bias vector is created and initialized
 *
 ** @param dtype 
 * The data type for the layer's tensors
 *
 ** @return
 * A pointer to the newly created FNN layer
 *
 ** @retval 
 * NULL Returned if memory allocation or tensor initialization fails
 *
 ** @warning 
 * The caller is responsible for freeing the memory by calling
 * `void nnl2_ann_free(void* nn)` on the returned pointer
 *
 ** @see nnl2_ann_free
 ** @see nnl2_ad_xavier
 */
nnl2_nn_fnn* nnl2_nn_fnn_create(int in_features, int out_features, bool use_bias, nnl2_tensor_type dtype, nnl2_nn_init_type init_type) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	nnl2_nn_fnn* nn = malloc(sizeof(nnl2_nn_fnn));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!nn) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}	
	#endif
	
	nn -> metadata.nn_type = nnl2_nn_type_fnn;
	nn -> metadata.use_bias = use_bias;
	
	nn->forward = use_bias ? fnn_forward_with_bias : fnn_forward_no_bias;
	
	bool init_success = false;
	
    switch(init_type) {
        case nnl2_nn_init_zeros: 			init_success = fnn_init_zeros(nn, in_features, out_features, dtype);  				break;
        case nnl2_nn_init_rand: 			init_success = fnn_init_rand(nn, in_features, out_features, dtype); 			    break;            
        case nnl2_nn_init_randn: 			init_success = fnn_init_randn(nn, in_features, out_features, dtype);  				break;        
        case nnl2_nn_init_xavier_normal: 	init_success = fnn_init_xavier_normal(nn, in_features, out_features, dtype);  		break;            
        case nnl2_nn_init_xavier_uniform: 	init_success = fnn_init_xavier_uniform(nn, in_features, out_features, dtype);  		break;        
        case nnl2_nn_init_kaiming_normal:   init_success = fnn_init_kaiming_normal(nn, in_features, out_features, dtype); 	    break;
        case nnl2_nn_init_kaiming_uniform: 	init_success = fnn_init_kaiming_uniform(nn, in_features, out_features, dtype); 	    break;    
        case nnl2_nn_init_identity: 		init_success = fnn_init_identity(nn, in_features, out_features, dtype, use_bias);   break;
            
        case nnl2_nn_init_unknown:
		
        default: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                NNL2_ERROR("Unknown initialization type: %d", init_type);
            #endif
			
            free(nn);
            return NULL;
		}
    }
	
	if(!init_success) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            NNL2_ERROR("FNN initialization failed for type: %d", init_type);
        #endif
		
        free(nn);
        return NULL;
    }
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return nn;
}

/** @brief 
 * Creates a Fully-Connected Neural Network (FNN) layer with user-provided tensors
 *
 ** @param in_features 
 * The number of input units/features
 *
 ** @param out_features 
 * The number of output units/features
 *
 ** @param use_bias 
 * If true, a bias vector is used and must be provided
 *
 ** @param dtype 
 * The data type for the layer's tensors
 *
 ** @param w
 * Pointer to a user-provided weights tensor. Must have shape [in_features, out_features]
 *
 ** @param b
 * Pointer to a user-provided bias tensor. Must have shape [out_features] if use_bias is true
 *
 ** @param handle_as
 * Enum specifying how the layer handles the provided tensors
 * 
 * Details: 
 *     nnl2_nn_handle_as_copy: make a copy of the provided tensors (safe)
 *     nnl2_nn_handle_as_view: use the provided tensors directly (lifetime managed by the caller)
 *
 ** @return nnl2_nn_fnn*
 * A pointer to the newly created FNN layer
 *
 ** @retval NULL 
 * if memory allocation fails or if tensor shapes are incorrect
 *
 ** @see nnl2_nn_fnn_create
 ** @see nnl2_nn_handle_as
 ** @see nnl2_ann_free
 **/
nnl2_nn_fnn* nnl2_nn_fnn_manual_create(int in_features, int out_features, bool use_bias, 
									   nnl2_tensor_type dtype, nnl2_ad_tensor* w, nnl2_ad_tensor* b, 
									   nnl2_nn_handle_as handle_as) {
										   
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif			

    nnl2_nn_fnn* nn = malloc(sizeof(nnl2_nn_fnn));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!nn) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}	
	#endif	
	
	bool w_initialized_correctly;
	
	{ // w must be [in_features, out_features]
		if(w -> data -> shape[0] != in_features) 	    { w_initialized_correctly = false; }
		else if(w -> data -> shape[1] != out_features)  { w_initialized_correctly = false; }
		else 											{ w_initialized_correctly = true;  }
	}

	if(!w_initialized_correctly) {
		NNL2_ERROR("In function nnl2_nn_fnn_manual_create, transfered nnl2_ad_tensor* w shape is NOT CORRECT. Expected shape: [%d, %d] (Explicit: [in_features, out_features]), Got: [%d, %d]", 
				   in_features, out_features, w -> data -> shape[0], w -> data -> shape[1]);
		
		free(nn);
		return NULL;
	}
	
	if(use_bias) { // b must be [out_features]
		if(b -> data -> shape[0] != out_features) {
			NNL2_ERROR("In function nnl2_nn_fnn_manual_create, transfered nnl2_ad_tensor* b shape is NOT CORRECT. Expected shape: [%d] (Explicit: [out_features]), Got: [%d]",
					   out_features, b -> data -> shape[0]);
					   
			free(nn);
			return NULL;
		}
	}
	
	nn -> metadata.nn_type = nnl2_nn_type_fnn;
	nn -> metadata.use_bias = use_bias;
	
	nn->forward = use_bias ? fnn_forward_with_bias : fnn_forward_no_bias;
	
	switch(handle_as) {
		case nnl2_nn_handle_as_copy: {
			nn -> weights = nnl2_ad_copy(w, dtype);
			if(use_bias) nn -> bias = nnl2_ad_copy(b, dtype);
			break;
		}
		
		case nnl2_nn_handle_as_view: {
			nn -> weights = w;
			if(use_bias) nn -> bias = b;
			break;
		}
		
		default: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				NNL2_ERROR("Unknown handle method in function nnl2_nn_fnn_manual_create. Handle enum type numbering: %d", handle_as);
			#endif 
			
			free(nn);
			return NULL;
		}
	}
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return nn;
}

/** @brief 
 * Destroys a Fully-Connected Neural Network (FNN) layer and releases its memory
 *
 ** @param nn 
 * Pointer to the FNN layer to be destroyed
 *
 ** @see nnl2_nn_fnn_create
 **/
void nnl2_nn_fnn_free(nnl2_nn_fnn* nn) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(nn, "In function nnl2_nn_fnn_free, nnl2_nn_fnn* nn is NULL");
	#endif 
	
	nnl2_free_ad_tensor(nn -> weights);
	
	if(nn -> metadata.use_bias) {
		nnl2_free_ad_tensor(nn -> bias);
	}
	
	free(nn);
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Retrieves all trainable parameters from a Fully-Connected layer
 *
 ** @param nn 
 * Pointer to the FNN layer
 *
 ** @return 
 * Dynamically allocated array of parameter tensor pointers
 *
 ** @retval 
 * NULL if memory allocation fails
 *
 ** @note 
 * The caller is responsible for freeing the returned array with `void nnl2_ann_free_parameters(nnl2_ad_tensor** parameters)`
 *
 ** @see nnl2_ann_free_parameters
 **/
nnl2_ad_tensor** nnl2_nn_fnn_get_parameters(nnl2_nn_fnn* nn) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_fnn_get_parameters, nnl2_nn_fnn* nn is NULL", NULL);
	#endif 
	
	nnl2_ad_tensor** params = malloc(sizeof(nnl2_ad_tensor*) * (nn -> metadata.use_bias ? 2 : 1));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!params) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif
	
	params[0] = nn -> weights;
	
	if(nn -> metadata.use_bias) {
		params[1] = nn -> bias;
	}
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return params;
}

/** @brief 
 * Returns the number of trainable parameter tensors in a Fully-Connected layer
 *
 ** @param nn 
 * Pointer to the FNN layer
 *
 ** @return 
 * The number of parameter tensors in the layer
 *
 ** @retval 1 
 * If the layer does not use bias (weights only)
 *
 ** @retval 2 
 * If the layer uses bias (weights + bias)
 *
 ** @see nnl2_nn_fnn_get_parameters
 **/
size_t nnl2_nn_fnn_get_num_parameters(nnl2_nn_fnn* nn) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_fnn_get_num_parameters, nnl2_nn_fnn* nn is NULL", 0);
	#endif 
	
	size_t num_parameters = nn -> metadata.use_bias ? 2 : 1; 

	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return num_parameters;
}

/** @brief 
 * Get the number of input features for a fully-connected layer
 *
 ** @param nn 
 * Pointer to the fully-connected layer structure
 *
 ** @return int
 * Number of input features (first dimension of weight matrix)
 */
int nnl2_nn_fnn_get_in_features(nnl2_nn_fnn* nn) {
	return nn -> weights -> data -> shape[0];
}

/** @brief 
 * Get the number of output features for a fully-connected layer
 *
 ** @param nn 
 * Pointer to the fully-connected layer structure
 *
 ** @return int 
 * Number of output features (second dimension of weight matrix)
 */
int nnl2_nn_fnn_get_out_features(nnl2_nn_fnn* nn) {
	return nn -> weights -> data -> shape[1];
}

/** @brief 
 * Print fully-connected layer information
 *
 ** @param nn 
 * Pointer to the fully-connected layer structure
 *
 ** @param terpri 
 * If true, print a newline after the output
 */
void nnl2_nn_fnn_print(nnl2_nn_fnn* nn, bool terpri) {
	if(!nn) {
        printf("(fnn NULL)%s", terpri ? "\n" : "");
        return;
    }
	
	printf("(fnn %d -> %d :bias %s)%s", nnl2_nn_fnn_get_in_features(nn), nnl2_nn_fnn_get_out_features(nn), nn -> metadata.use_bias ? "t" : "nil", terpri ? "\n" : "");
}

/** @brief 
 * Encodes FNN information in nnlrepr format
 *
 ** @param nn 
 * Input FNN
 */
static nnl2_nnlrepr_template* nnl2_nn_fnn_nnlrepr_template(nnl2_nn_fnn* nn) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
        if(!nn || !nn->weights || !nn->weights->data) {
            NNL2_ERROR("In function nnl2_nn_fnn_nnlrepr_template, failed assertion ```!nn || !nn->weights || !nn->weights->data```");
            return NULL;
        }
    #endif
	
    nnl2_nnlrepr_template* result = malloc(sizeof(nnl2_nnlrepr_template));
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
		if(!result) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif 
    
	// Common metadata
    result -> nn_type = nnl2_nn_type_fnn;
	result -> dtype = nn -> weights -> data -> dtype;
    result -> num_shapes = (nn -> metadata.use_bias ? 2 : 1);   // 2 - weights + bias : 1 - weights only
    result -> vector_size = product(nn -> weights -> data -> shape, 2); // 2 dimensions
    result -> num_childrens = 0;
    result -> childrens = NULL;
    result -> additional_data = NULL;

    result -> shapes = malloc(sizeof(int32_t*) * result -> num_shapes);
	
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
		if(!result->shapes) {
			NNL2_MALLOC_ERROR();
			free(result);
			return NULL;
		}
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
			for(size_t i = 0; i < result->num_shapes; i++) {
				result->shapes[i] = NULL;
			}
		#endif 
	#endif

    result -> shapes[0] = malloc(sizeof(int32_t) * 2); // 2 dimensions
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
        if(!result->shapes[0]) {
            NNL2_MALLOC_ERROR();
            nnl2_nnlrepr_template_free(result);
            return NULL;
        }
    #endif

    result -> shapes[0][0] = nn -> weights -> data -> shape[0];  
    result -> shapes[0][1] = nn -> weights -> data -> shape[1];
    
    if(nn->metadata.use_bias) {
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
            if(!nn->bias || !nn->bias->data) {
                NNL2_ERROR("In function nnl2_nn_fnn_nnlrepr_template, bias is enabled but bias data is invalid");				
                nnl2_nnlrepr_template_free(result);
                return NULL;
            }
        #endif
		
        result->shapes[1] = malloc(sizeof(int32_t)); // 1 dimensions so malloc(sizeof(int32_t) * 1) not needed
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
            if(!result->shapes[1]) {
                NNL2_MALLOC_ERROR();
                nnl2_nnlrepr_template_free(result);
                return NULL;
            }
        #endif
		
        result -> shapes[1][0] = nn -> bias -> data -> shape[0];
		result -> vector_size += nn -> bias -> data -> shape[0];
    }
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
    
    return result;
}

/** @brief 
 * Decodes FNN information in nnlrepr format
 *
 ** @param vector 
 * Encoded 1D nnlrepr vector 
 *
 ** @param offset
 * Encoded vector shift to nnl2_ad_vector_as_parameter(..., ..., offset, ...);
 * 
 ** @param num_shapes
 * Number of shapes of all parameters 
 *
 ** @param shape_w 
 * fnn -> weights Parameter
 *
 ** @param shape_b 
 * fnn -> bias Parameter (If bias needed)
 *
 ** @param dtype 
 * Encoder data type. Needs for nnl2_nn_fnn_manual_create(..., ......, dtype, ..., ......)
 *
 ** @return nnl2_nn_fnn* 
 * Pointer to created FNN or NULL on error
 */
static nnl2_nn_fnn* nnl2_nn_fnn_nnlrepr_decode(nnl2_ad_tensor* vector, size_t offset, int num_shapes, int32_t* shape_w, int32_t* shape_b, nnl2_tensor_type dtype) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(vector, "In function nnl2_nn_fnn_nnlrepr_decode, nnl2_ad_tensor* vector is NULL. returning NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape_w, "In function nnl2_nn_fnn_nnlrepr_decode, int32_t* shape_w is NULL. returning NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape_b, "In function nnl2_nn_fnn_nnlrepr_decode, int32_t* shape_b is NULL. returning NULL", NULL);
	#endif
	
	nnl2_ad_tensor* w_view = nnl2_ad_vector_as_parameter(shape_w, 2, offset, vector); // 2 is a two dimensions
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(w_view == NULL) {
            NNL2_ERROR("In function nnl2_nn_fnn_nnlrepr_decode, failed to create weights view. returning NULL");
            return NULL;
        }
    #endif
	
	offset += product(shape_w, 2); // 2 dimensions
	
	nnl2_ad_tensor* b_view = NULL;
	if(num_shapes == 2) {
        b_view = nnl2_ad_vector_as_parameter(shape_b, 1, offset, vector); // 1 is a one dimension
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(b_view == NULL) {
                NNL2_ERROR("In function nnl2_nn_fnn_nnlrepr_decode, failed to create bias view. returning NULL");
                nnl2_free_ad_tensor(w_view);
                return NULL;
            }
        #endif

        offset += shape_b[0];
    }
	
	nnl2_nn_fnn* result = nnl2_nn_fnn_manual_create(
        shape_w[0],                // input_size
        shape_w[1],                // output_size
        (num_shapes == 2),         // use_bias
        dtype,                     // data type
        w_view,                    // weights tensor
        b_view,                    // bias tensor (may be NULLL)
        nnl2_nn_handle_as_view     // handle as view
    );
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to create FNN from decoded parameters. returning NULL");
            nnl2_free_ad_tensor(w_view);
            if (b_view != NULL) nnl2_free_ad_tensor(b_view);
            return NULL;
        }
    #endif
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
		NNL2_INFO("Successfully decoded FNN: input_size=%d, output_size=%d, use_bias=%s",
                  shape_w[0], shape_w[1], (num_shapes == 2) ? "true" : "false");
				  
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

#endif /** NNL2_FNN_BACKEND_H **/
