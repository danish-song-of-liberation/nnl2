#ifndef NNL2_FNN_INIT_H
#define NNL2_FNN_INIT_H

/** @file nnl2_fnn_init.h
 ** @date 2025
 ** @copyright MIT License
 **/

/** @brief Initializes FNN layer with zeros **/
static bool fnn_init_zeros(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing FNN with zeros");
    #endif
    
    nn->weights = nnl2_ad_zeros((int[]){ in_features, out_features }, 2, dtype, true, "fnn:weights");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn->weights) {
            NNL2_TENSOR_ERROR("fnn:weights:zeros");
            return false;
        }
    #endif
    
    if(nn->metadata.use_bias) {
        nn->bias = nnl2_ad_zeros((int[]){ out_features }, 1, dtype, true, "fnn:bias");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!nn->bias) {
                NNL2_TENSOR_ERROR("fnn:bias:zeros");
                nnl2_free_ad_tensor(nn->weights);
                return false;
            }
        #endif
    } else {
        nn->bias = NULL;
    }
    
    return true;
}

/** @brief Initializes FNN layer with uniform random values **/
static bool fnn_init_rand(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing FNN with uniform random");
    #endif
    
    nn->weights = nnl2_ad_rand((int[]){ in_features, out_features }, 2, dtype, true, "fnn:weights");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn->weights) {
            NNL2_TENSOR_ERROR("fnn:weights:rand");
            return false;
        }
    #endif
    
    if(nn->metadata.use_bias) {
        nn->bias = nnl2_ad_rand((int[]){ out_features }, 1, dtype, true, "fnn:bias");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!nn->bias) {
                NNL2_TENSOR_ERROR("fnn:bias:rand");
                nnl2_free_ad_tensor(nn->weights);
                return false;
            }
        #endif
    } else {
        nn->bias = NULL;
    }
    
    return true;
}

/** @brief Initializes FNN layer with normal random values **/
static bool fnn_init_randn(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing FNN with normal random");
    #endif
    
    nn->weights = nnl2_ad_randn((int[]){ in_features, out_features }, 2, dtype, true, 
                               "fnn:weights", 0.0, 1.0);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn->weights) {
            NNL2_TENSOR_ERROR("fnn:weights:randn");
            return false;
        }
    #endif
    
    if(nn->metadata.use_bias) {
        nn->bias = nnl2_ad_randn((int[]){ out_features }, 1, dtype, true, 
                                "fnn:bias", 0.0, 1.0);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!nn->bias) {
                NNL2_TENSOR_ERROR("fnn:bias:randn");
                nnl2_free_ad_tensor(nn->weights);
                return false;
            }
        #endif
    } else {
        nn->bias = NULL;
    }
    
    return true;
}

/** @brief Initializes FNN layer with Xavier normal initialization **/
static bool fnn_init_xavier_normal(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing FNN with Xavier normal");
    #endif
    
    nn->weights = nnl2_ad_xavier((int[]){ in_features, out_features }, 2, dtype, true, "fnn:weights", in_features, out_features, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn->weights) {
            NNL2_TENSOR_ERROR("fnn:weights:xavier_normal");
            return false;
        }
    #endif
    
    if(nn->metadata.use_bias) {
        nn->bias = nnl2_ad_xavier((int[]){ out_features }, 1, dtype, true, "fnn:bias", in_features, out_features, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!nn->bias) {
                NNL2_TENSOR_ERROR("fnn:bias:xavier_normal");
                nnl2_free_ad_tensor(nn->weights);
                return false;
            }
        #endif
    } else {
        nn->bias = NULL;
    }
    
    return true;
}

/** @brief Initializes FNN layer with Xavier uniform initialization **/
static bool fnn_init_xavier_uniform(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing FNN with Xavier uniform");
    #endif
    
    nn->weights = nnl2_ad_xavier((int[]){ in_features, out_features }, 2, dtype, true, "fnn:weights", in_features, out_features, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_UNIFORM_DIST);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn->weights) {
            NNL2_TENSOR_ERROR("fnn:weights:xavier_uniform");
            return false;
        }
    #endif
    
    if(nn->metadata.use_bias) {
        nn->bias = nnl2_ad_xavier((int[]){ out_features }, 1, dtype, true, "fnn:bias", in_features, out_features, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_UNIFORM_DIST);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!nn->bias) {
                NNL2_TENSOR_ERROR("fnn:bias:xavier_uniform");
                nnl2_free_ad_tensor(nn->weights);
                return false;
            }
        #endif
    } else {
        nn->bias = NULL;
    }
    
    return true;
}

/** @brief Initializes FNN layer with Kaiming normal initialization **/
static bool fnn_init_kaiming_normal(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing FNN with Kaiming normal");
    #endif
    
    nn->weights = nnl2_ad_kaiming((int[]){ in_features, out_features }, 2, dtype, true, "fnn:weights", in_features, out_features, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_NORMAL_DIST, NNL2_KAIMING_MODE_FAN_IN);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn->weights) {
            NNL2_TENSOR_ERROR("fnn:weights:kaiming_normal");
            return false;
        }
    #endif
    
    if(nn->metadata.use_bias) {
        nn->bias = nnl2_ad_kaiming((int[]){ out_features }, 1, dtype, true, "fnn:bias", in_features, out_features, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_NORMAL_DIST, NNL2_KAIMING_MODE_FAN_IN);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!nn->bias) {
                NNL2_TENSOR_ERROR("fnn:bias:kaiming_normal");
                nnl2_free_ad_tensor(nn->weights);
                return false;
            }
        #endif
    } else {
        nn->bias = NULL;
    }
    
    return true;
}

/** @brief Initializes FNN layer with Kaiming uniform initialization **/
static bool fnn_init_kaiming_uniform(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing FNN with Kaiming uniform");
    #endif
    
    nn->weights = nnl2_ad_kaiming((int[]){ in_features, out_features }, 2, dtype, true, "fnn:weights", in_features, out_features, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_UNIFORM_DIST, NNL2_KAIMING_MODE_FAN_IN);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn->weights) {
            NNL2_TENSOR_ERROR("fnn:weights:kaiming_uniform");
            return false;
        }
    #endif
    
    if(nn->metadata.use_bias) {
        nn->bias = nnl2_ad_kaiming((int[]){ out_features }, 1, dtype, true, "fnn:bias", in_features, out_features, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_UNIFORM_DIST, NNL2_KAIMING_MODE_FAN_IN);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!nn->bias) {
                NNL2_TENSOR_ERROR("fnn:bias:kaiming_uniform");
                nnl2_free_ad_tensor(nn->weights);
                return false;
            }
        #endif
    } else {
        nn->bias = NULL;
    }
    
    return true;
}

/** @brief Initializes FNN layer with identity matrix for testing **/
static bool fnn_init_identity(nnl2_nn_fnn* nn, int in_features, int out_features, nnl2_tensor_type dtype, bool use_bias) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing FNN with identity matrix (for testing)");
    #endif
    
    nn->weights = nnl2_ad_empty((int[]){ in_features, out_features }, 2, dtype, true, "fnn:weights");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn->weights) {
            NNL2_TENSOR_ERROR("fnn:weights:identity");
            return false;
        }
    #endif
    
    if(use_bias) {
        nn->bias = nnl2_ad_empty((int[]){ out_features }, 1, dtype, true, "fnn:bias");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!nn->bias) {
                NNL2_TENSOR_ERROR("fnn:bias:identity");
                nnl2_free_ad_tensor(nn->weights);
                return false;
            }
        #endif
    } else {
        nn->bias = NULL;
    }
    
    return true;
}

#endif /** NNL2_FNN_INIT_H **/
