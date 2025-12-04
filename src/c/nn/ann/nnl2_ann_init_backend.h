#ifndef NNL2_ANN_INIT_BACKEND_H
#define NNL2_ANN_INIT_BACKEND_H

/** @file nnl2_ann_init_backend.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains nnl2_nn_init_type enum definition
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_init_backend.h
 **           
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2	
 **/    

///@{ [nnl2_nn_init_type]

/** @enum nnl2_nn_init_type
 ** @brief
 * Supported neural network initialization methods for weight tensors
 */
typedef enum {
	nnl2_nn_init_identity = 0, 			 ///< Do not perform any initialization. The weights remain untouched. Used when a user-supplied custom initializer function
	nnl2_nn_init_zeros    = 1,			 ///< Fill the weight tensor with zeros
	nnl2_nn_init_rand     = 2,			 ///< Fill tensor with values sampled from a uniform distribution in [0, 1]
	nnl2_nn_init_randn    = 3,			 ///< Fill tensor with values sampled from a standard normal distribution (mean=0, std=1)
	nnl2_nn_init_xavier_normal   = 4,	 ///< Xavier (Glorot) initialization using a normal distribution
	nnl2_nn_init_xavier_uniform  = 5,	 ///< Xavier (Glorot) initialization using a uniform distribution
	nnl2_nn_init_kaiming_normal  = 6,    ///< Kaiming (He) initialization using a normal distribution
	nnl2_nn_init_kaiming_uniform = 7,    ///< Kaiming (He) initialization using a uniform distribution
	nnl2_nn_init_unknown = 8			 ///< Undefined or unsupported initialization type
} nnl2_nn_init_type;

///@} [nnl2_nn_init_type]

#endif /** NNL2_ANN_INIT_BACKEND_H **/
