#ifndef NNL2_RNNCELL_INIT_H
#define NNL2_RNNCELL_INIT_H

/** @brief Initializes RNNCell with zeros **/
static bool rnncell_init_zeros(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing RNNCell with zeros");
    #endif
    
    cell->wxh = nnl2_ad_zeros((int[]){ input_size, hidden_size }, 2, dtype, true, "rnncell:wxh");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->wxh) {
            NNL2_TENSOR_ERROR("rnncell:wxh:zeros");
            return false;
        }
    #endif
    
    cell->whh = nnl2_ad_zeros((int[]){ hidden_size, hidden_size }, 2, dtype, true, "rnncell:whh");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->whh) {
            NNL2_TENSOR_ERROR("rnncell:whh:zeros");
            nnl2_free_ad_tensor(cell->wxh);
            return false;
        }
    #endif
    
    if (cell->metadata.use_bias) {
        cell->bxh = nnl2_ad_zeros((int[]){ hidden_size }, 1, dtype, true, "rnncell:bxh");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bxh) {
                NNL2_TENSOR_ERROR("rnncell:bxh:zeros");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                return false;
            }
        #endif
        
        cell->bhh = nnl2_ad_zeros((int[]){ hidden_size }, 1, dtype, true, "rnncell:bhh");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bhh) {
                NNL2_TENSOR_ERROR("rnncell:bhh:zeros");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                nnl2_free_ad_tensor(cell->bxh);
                return false;
            }
        #endif
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }
    
    cell->hidden_size = hidden_size;
    
    return true;
}

/** @brief Initializes RNNCell with uniform random values **/
static bool rnncell_init_rand(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing RNNCell with uniform random");
    #endif
    
    cell->wxh = nnl2_ad_rand((int[]){ input_size, hidden_size }, 2, dtype, true, "rnncell:wxh");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->wxh) {
            NNL2_TENSOR_ERROR("rnncell:wxh:rand");
            return false;
        }
    #endif
    
    cell->whh = nnl2_ad_rand((int[]){ hidden_size, hidden_size }, 2, dtype, true, "rnncell:whh");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->whh) {
            NNL2_TENSOR_ERROR("rnncell:whh:rand");
            nnl2_free_ad_tensor(cell->wxh);
            return false;
        }
    #endif
    
    if (cell->metadata.use_bias) {
        cell->bxh = nnl2_ad_rand((int[]){ hidden_size }, 1, dtype, true, "rnncell:bxh");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bxh) {
                NNL2_TENSOR_ERROR("rnncell:bxh:rand");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                return false;
            }
        #endif
        
        cell->bhh = nnl2_ad_rand((int[]){ hidden_size }, 1, dtype, true, "rnncell:bhh");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bhh) {
                NNL2_TENSOR_ERROR("rnncell:bhh:rand");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                nnl2_free_ad_tensor(cell->bxh);
                return false;
            }
        #endif
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }
    
    cell->hidden_size = hidden_size;
    
    return true;
}

/** @brief Initializes RNNCell with normal random values **/
static bool rnncell_init_randn(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing RNNCell with normal random");
    #endif
    
    cell->wxh = nnl2_ad_randn((int[]){ input_size, hidden_size }, 2, dtype, true, "rnncell:wxh", 0.0, 1.0);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->wxh) {
            NNL2_TENSOR_ERROR("rnncell:wxh:randn");
            return false;
        }
    #endif
    
    cell->whh = nnl2_ad_randn((int[]){ hidden_size, hidden_size }, 2, dtype, true, "rnncell:whh", 0.0, 1.0);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->whh) {
            NNL2_TENSOR_ERROR("rnncell:whh:randn");
            nnl2_free_ad_tensor(cell->wxh);
            return false;
        }
    #endif
    
    if (cell->metadata.use_bias) {
        cell->bxh = nnl2_ad_randn((int[]){ hidden_size }, 1, dtype, true, "rnncell:bxh", 0.0, 1.0);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bxh) {
                NNL2_TENSOR_ERROR("rnncell:bxh:randn");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                return false;
            }
        #endif
        
        cell->bhh = nnl2_ad_randn((int[]){ hidden_size }, 1, dtype, true, "rnncell:bhh", 0.0, 1.0);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bhh) {
                NNL2_TENSOR_ERROR("rnncell:bhh:randn");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                nnl2_free_ad_tensor(cell->bxh);
                return false;
            }
        #endif
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }
    
    cell->hidden_size = hidden_size;
    
    return true;
}

/** @brief Initializes RNNCell with Xavier normal initialization **/
static bool rnncell_init_xavier_normal(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing RNNCell with Xavier normal");
    #endif
    
    // Wxh: [input_size, hidden_size] - учитываем fan_in = input_size
    cell->wxh = nnl2_ad_xavier((int[]){ input_size, hidden_size }, 2, dtype, true, "rnncell:wxh", 
                              input_size, hidden_size, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->wxh) {
            NNL2_TENSOR_ERROR("rnncell:wxh:xavier_normal");
            return false;
        }
    #endif
    
    // Whh: [hidden_size, hidden_size] - учитываем fan_in = hidden_size
    cell->whh = nnl2_ad_xavier((int[]){ hidden_size, hidden_size }, 2, dtype, true, "rnncell:whh", 
                              hidden_size, hidden_size, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->whh) {
            NNL2_TENSOR_ERROR("rnncell:whh:xavier_normal");
            nnl2_free_ad_tensor(cell->wxh);
            return false;
        }
    #endif
    
    if (cell->metadata.use_bias) {
        // Bias: учитываем размеры как для соответствующих весов
        cell->bxh = nnl2_ad_xavier((int[]){ hidden_size }, 1, dtype, true, "rnncell:bxh", 
                                  input_size, hidden_size, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bxh) {
                NNL2_TENSOR_ERROR("rnncell:bxh:xavier_normal");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                return false;
            }
        #endif
        
        cell->bhh = nnl2_ad_xavier((int[]){ hidden_size }, 1, dtype, true, "rnncell:bhh", 
                                  hidden_size, hidden_size, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_NORMAL_DIST);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bhh) {
                NNL2_TENSOR_ERROR("rnncell:bhh:xavier_normal");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                nnl2_free_ad_tensor(cell->bxh);
                return false;
            }
        #endif
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }
    
    cell->hidden_size = hidden_size;
    
    return true;
}

/** @brief Initializes RNNCell with Xavier uniform initialization **/
static bool rnncell_init_xavier_uniform(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing RNNCell with Xavier uniform");
    #endif
    
    cell->wxh = nnl2_ad_xavier((int[]){ input_size, hidden_size }, 2, dtype, true, "rnncell:wxh", 
                              input_size, hidden_size, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_UNIFORM_DIST);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->wxh) {
            NNL2_TENSOR_ERROR("rnncell:wxh:xavier_uniform");
            return false;
        }
    #endif
    
    cell->whh = nnl2_ad_xavier((int[]){ hidden_size, hidden_size }, 2, dtype, true, "rnncell:whh", 
                              hidden_size, hidden_size, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_UNIFORM_DIST);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->whh) {
            NNL2_TENSOR_ERROR("rnncell:whh:xavier_uniform");
            nnl2_free_ad_tensor(cell->wxh);
            return false;
        }
    #endif
    
    if (cell->metadata.use_bias) {
        cell->bxh = nnl2_ad_xavier((int[]){ hidden_size }, 1, dtype, true, "rnncell:bxh", 
                                  input_size, hidden_size, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_UNIFORM_DIST);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bxh) {
                NNL2_TENSOR_ERROR("rnncell:bxh:xavier_uniform");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                return false;
            }
        #endif
        
        cell->bhh = nnl2_ad_xavier((int[]){ hidden_size }, 1, dtype, true, "rnncell:bhh", 
                                  hidden_size, hidden_size, NNL2_XAVIER_NO_GAIN, NNL2_XAVIER_UNIFORM_DIST);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bhh) {
                NNL2_TENSOR_ERROR("rnncell:bhh:xavier_uniform");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                nnl2_free_ad_tensor(cell->bxh);
                return false;
            }
        #endif
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }
    
    cell->hidden_size = hidden_size;
    
    return true;
}

/** @brief Initializes RNNCell with Kaiming normal initialization **/
static bool rnncell_init_kaiming_normal(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing RNNCell with Kaiming normal");
    #endif
    
    cell->wxh = nnl2_ad_kaiming((int[]){ input_size, hidden_size }, 2, dtype, true, "rnncell:wxh", 
                               input_size, hidden_size, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_NORMAL_DIST, NNL2_KAIMING_MODE_FAN_IN);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->wxh) {
            NNL2_TENSOR_ERROR("rnncell:wxh:kaiming_normal");
            return false;
        }
    #endif
    
    cell->whh = nnl2_ad_kaiming((int[]){ hidden_size, hidden_size }, 2, dtype, true, "rnncell:whh", 
                               hidden_size, hidden_size, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_NORMAL_DIST, NNL2_KAIMING_MODE_FAN_IN);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->whh) {
            NNL2_TENSOR_ERROR("rnncell:whh:kaiming_normal");
            nnl2_free_ad_tensor(cell->wxh);
            return false;
        }
    #endif
    
    if (cell->metadata.use_bias) {
        cell->bxh = nnl2_ad_kaiming((int[]){ hidden_size }, 1, dtype, true, "rnncell:bxh", 
                                   input_size, hidden_size, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_NORMAL_DIST, NNL2_KAIMING_MODE_FAN_IN);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bxh) {
                NNL2_TENSOR_ERROR("rnncell:bxh:kaiming_normal");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                return false;
            }
        #endif
        
        cell->bhh = nnl2_ad_kaiming((int[]){ hidden_size }, 1, dtype, true, "rnncell:bhh", 
                                   hidden_size, hidden_size, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_NORMAL_DIST, NNL2_KAIMING_MODE_FAN_IN);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bhh) {
                NNL2_TENSOR_ERROR("rnncell:bhh:kaiming_normal");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                nnl2_free_ad_tensor(cell->bxh);
                return false;
            }
        #endif
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }
    
    cell->hidden_size = hidden_size;
    
    return true;
}

/** @brief Initializes RNNCell with Kaiming uniform initialization **/
static bool rnncell_init_kaiming_uniform(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing RNNCell with Kaiming uniform");
    #endif
    
    cell->wxh = nnl2_ad_kaiming((int[]){ input_size, hidden_size }, 2, dtype, true, "rnncell:wxh", 
                               input_size, hidden_size, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_UNIFORM_DIST, NNL2_KAIMING_MODE_FAN_IN);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->wxh) {
            NNL2_TENSOR_ERROR("rnncell:wxh:kaiming_uniform");
            return false;
        }
    #endif
    
    cell->whh = nnl2_ad_kaiming((int[]){ hidden_size, hidden_size }, 2, dtype, true, "rnncell:whh", 
                               hidden_size, hidden_size, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_UNIFORM_DIST, NNL2_KAIMING_MODE_FAN_IN);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->whh) {
            NNL2_TENSOR_ERROR("rnncell:whh:kaiming_uniform");
            nnl2_free_ad_tensor(cell->wxh);
            return false;
        }
    #endif
    
    if (cell->metadata.use_bias) {
        cell->bxh = nnl2_ad_kaiming((int[]){ hidden_size }, 1, dtype, true, "rnncell:bxh", 
                                   input_size, hidden_size, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_UNIFORM_DIST, NNL2_KAIMING_MODE_FAN_IN);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bxh) {
                NNL2_TENSOR_ERROR("rnncell:bxh:kaiming_uniform");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                return false;
            }
        #endif
        
        cell->bhh = nnl2_ad_kaiming((int[]){ hidden_size }, 1, dtype, true, "rnncell:bhh", 
                                   hidden_size, hidden_size, NNL2_KAIMING_NO_GAIN, NNL2_KAIMING_UNIFORM_DIST, NNL2_KAIMING_MODE_FAN_IN);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bhh) {
                NNL2_TENSOR_ERROR("rnncell:bhh:kaiming_uniform");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                nnl2_free_ad_tensor(cell->bxh);
                return false;
            }
        #endif
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }
    
    cell->hidden_size = hidden_size;
    
    return true;
}

/** @brief Initializes RNNCell with identity matrix for testing **/
static bool rnncell_init_identity(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Initializing RNNCell with identity matrix (for testing)");
    #endif
    
    cell->wxh = nnl2_ad_empty((int[]){ input_size, hidden_size }, 2, dtype, true, "rnncell:wxh");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->wxh) {
            NNL2_TENSOR_ERROR("rnncell:wxh:identity");
            return false;
        }
    #endif
    
    cell->whh = nnl2_ad_empty((int[]){ hidden_size, hidden_size }, 2, dtype, true, "rnncell:whh");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell->whh) {
            NNL2_TENSOR_ERROR("rnncell:whh:identity");
            nnl2_free_ad_tensor(cell->wxh);
            return false;
        }
    #endif
    
    if (cell->metadata.use_bias) {
        cell->bxh = nnl2_ad_empty((int[]){ hidden_size }, 1, dtype, true, "rnncell:bxh");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bxh) {
                NNL2_TENSOR_ERROR("rnncell:bxh:identity");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                return false;
            }
        #endif
        
        cell->bhh = nnl2_ad_empty((int[]){ hidden_size }, 1, dtype, true, "rnncell:bhh");
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!cell->bhh) {
                NNL2_TENSOR_ERROR("rnncell:bhh:identity");
                nnl2_free_ad_tensor(cell->wxh);
                nnl2_free_ad_tensor(cell->whh);
                nnl2_free_ad_tensor(cell->bxh);
                return false;
            }
        #endif
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }
    
    cell->hidden_size = hidden_size;
    
    return true;
}

#endif /** NNL2_RNNCELL_INIT_H **/
