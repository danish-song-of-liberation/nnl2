#ifndef NNL2_NN_EMBEDDING_H
#define NNL2_NN_EMBEDDING_H

/** @file nnl2_nn_softmax.h
 ** @date 2026
 ** @copyright MIT License
 ** @brief Contains embedding for nnl2.hli.nn 
 **/



///@{ [nnl2_nn_embedding]

typedef struct nnl2_nn_embedding_struct {
    nnl2_nn_ann metadata;       ///< Base neural network metadata
	nnl2_ad_tensor* embedding;  ///< Embedding matrix
} nnl2_nn_embedding;

///@} [nnl2_nn_embedding]



nnl2_nn_embedding* nnl2_nn_embedding_create(int vocab_size, int embed_dim, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_nn_embedding* nn = malloc(sizeof(nnl2_nn_embedding));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }   
    #endif
    
    nn -> metadata.nn_type = nnl2_nn_type_embedding;
    nn -> metadata.use_bias = false; 
    nn -> metadata.nn_magic = NNL2_NN_MAGIC;
	
	nn -> embedding = nnl2_ad_xavier((int[]){ vocab_size, embed_dim }, 2, dtype, true, "embedding", vocab_size, embed_dim, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST);
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nn;
}

void nnl2_nn_embedding_free(nnl2_nn_embedding* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            return;
        }
    #endif 
    
	nnl2_free_ad_tensor(nn -> embedding);
	
    free(nn);
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

nnl2_ad_tensor** nnl2_nn_embedding_get_parameters(nnl2_nn_embedding* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER(); 
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
        if (nn == NULL) { 
            return NULL;
        }
    #endif 
    
    nnl2_ad_tensor** params = malloc(sizeof(nnl2_ad_tensor*));
	
	params[0] = nn -> embedding;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return params; 
}

size_t nnl2_nn_embedding_get_num_parameters(nnl2_nn_embedding* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            return 0;
        }
    #endif 
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return 1;  
}

nnl2_ad_tensor* nnl2_nn_embedding_forward_index(nnl2_nn_embedding* nn, nnl2_ad_tensor* index) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_embedding_forward_index, nnl2_nn_softmax* nn is NULL", NULL);
    #endif
    
    nnl2_ad_tensor* forward_pass = nnl2_ad_tref_getter(nn -> embedding, (int32_t*)(index -> data -> data), 1, nnl2_ad_reverse_mode, true, false);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(forward_pass, "In function nnl2_nn_embedding_forward_index, failed to exrract row from embedding matrix", NULL);
    #endif
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif

    return forward_pass;
}

#endif /** NNL2_NN_EMBEDDING_H **/
