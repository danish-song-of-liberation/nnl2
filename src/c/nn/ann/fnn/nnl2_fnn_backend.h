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

typedef struct {
	nnl2_nn_ann metadata;		///< Base neural network metadata and type information
	nnl2_ad_tensor* weights;    ///< Weight matrix of shape [in_features, out_features]
	nnl2_ad_tensor* bias;		///< Bias vector of shape [out_features]. May be NULL if use_bias is set to false in metadata
} nnl2_nn_fnn;

///@} [nnl2_nn_fnn]

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
nnl2_nn_fnn* nnl2_nn_fnn_create(int in_features, int out_features, bool use_bias, nnl2_tensor_type dtype) {
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
	
	// Matrix with [in_features, out_features] shape
	nn -> weights = nnl2_ad_xavier((int[]){ in_features, out_features }, 2, dtype, true, "fnn:weights", in_features, out_features, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!nn -> weights) {
			NNL2_TENSOR_ERROR("fnn:weights:xavier");
			free(nn);
			return NULL;
		}
	#endif
	
	if(use_bias) {
		// Vector with [out_features] shape
		nn -> bias = nnl2_ad_xavier((int[]){ out_features }, 1, dtype, true, "fnn:bias", in_features, out_features, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST); 
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if(!nn->bias) {
				NNL2_TENSOR_ERROR("fnn:bias:xavier");
				nnl2_free_ad_tensor(nn -> weights);
				free(nn);
				return NULL;
			}
		#endif
	} else {
		nn -> bias = NULL;
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

#endif /** NNL2_FNN_BACKEND_H **/
