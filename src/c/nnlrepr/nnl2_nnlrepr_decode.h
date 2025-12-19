#ifndef NNL2_NNLREPR_DECODE_H
#define NNL2_NNLREPR_DECODE_H

typedef struct nnl2_nn_sequential_struct nnl2_nn_sequential;
typedef struct nnl2_nn_leaky_relu_struct nnl2_nn_leaky_relu;
typedef struct nnl2_nn_rnn_cell_struct nnl2_nn_rnn_cell;
typedef struct nnl2_nn_sigmoid_struct nnl2_nn_sigmoid;
typedef struct nnl2_nn_relu_struct nnl2_nn_relu;
typedef struct nnl2_nn_tanh_struct nnl2_nn_tanh;
typedef struct nnl2_nn_fnn_struct nnl2_nn_fnn;

/** @brief Decodes RNN cell from nnlrepr format **/	
static nnl2_nn_rnn_cell* nnl2_nn_rnn_cell_nnlrepr_decode(nnl2_ad_tensor* vector, size_t offset, int num_shapes, int32_t* shape_wxh, int32_t* shape_whh, 
														 int32_t* shape_bhh, int32_t* shape_bxh, nnl2_tensor_type dtype);

/** @brief Decodes FNN from nnlrepr format **/												  
static nnl2_nn_fnn* nnl2_nn_fnn_nnlrepr_decode(nnl2_ad_tensor* vector, size_t offset, int num_shapes, 
											   int32_t* shape_w, int32_t* shape_b, nnl2_tensor_type dtype);												  
												  
nnl2_nn_sequential* nnl2_nn_sequential_create(size_t num_layers, void** layers);
nnl2_nn_leaky_relu* nnl2_nn_leaky_relu_create(float alpha);
nnl2_nn_sigmoid* nnl2_nn_sigmoid_create(bool approx);
nnl2_nn_tanh* nnl2_nn_tanh_create(bool approx);
nnl2_nn_relu* nnl2_nn_relu_create(void);

/** @brief Internal recursive decode function **/
void* nnl2_nn_decode_internal(nnl2_nnlrepr_template* encoder, nnl2_ad_tensor* vector, size_t offset) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
        NNL2_INFO("Decoding layer type: %d at offset: %zu", encoder->nn_type, offset);
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (encoder == NULL) {
            NNL2_ERROR("In function nnl2_nn_decode_internal, encoder template is NULL. returning NULL");
            return NULL;
        }

        if (vector == NULL) {
            NNL2_ERROR("In function nnl2_nn_decode_internal, vector is NULL. returning NULL");
            return NULL;
        }
    #endif
	
	void* result = NULL;
	
	switch(encoder -> nn_type) {
		case nnl2_nn_type_fnn: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(encoder->num_shapes < 1 || encoder->shapes == NULL || encoder->shapes[0] == NULL) {
                    NNL2_ERROR("In function nnl2_nn_decode_internal, invalid FNN encoder. returning NULL");
                    return NULL;
                }
            #endif
			
			int32_t* shape_b = (encoder->num_shapes > 1) ? encoder->shapes[1] : NULL;
            result = nnl2_nn_fnn_nnlrepr_decode(vector, offset, encoder->num_shapes, encoder->shapes[0], shape_b, encoder->dtype);
			
            break;
		}
		
		case nnl2_nn_type_rnn_cell: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if (encoder->num_shapes < 2 || encoder->shapes == NULL || 
                    encoder->shapes[0] == NULL || encoder->shapes[1] == NULL) {
                    NNL2_ERROR("In function nnl2_nn_decode_internal, invalid RNN cell encoder. returning NULL");
                    return NULL;
                }
            #endif
            
            int32_t* shape_bxh = (encoder->num_shapes > 2 && encoder->shapes[2]) ? encoder->shapes[2] : NULL;
            int32_t* shape_bhh = (encoder->num_shapes > 3 && encoder->shapes[3]) ? encoder->shapes[3] : NULL;
            
            result = nnl2_nn_rnn_cell_nnlrepr_decode(vector, offset, encoder->num_shapes, 
                                                     encoder->shapes[0], encoder->shapes[1], 
                                                     shape_bhh, shape_bxh, encoder->dtype);
													 
            break;	
        }
		
		case nnl2_nn_type_sequential: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(encoder->num_childrens == 0 || encoder->childrens == NULL) {
                    NNL2_ERROR("In function nnl2_nn_decode_internal, sequential layer has no children. returning NULL");
                    return NULL;
                }
            #endif
			
			void** networks = malloc(sizeof(void*) * encoder -> num_childrens);
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if (networks == NULL) {
                    NNL2_MALLOC_ERROR();
                    return NULL;
                }
            #endif
			
			size_t current_offset = offset;
			
			for(size_t it = 0; it < encoder -> num_childrens; it++) {
				networks[it] = nnl2_nn_decode_internal(encoder -> childrens[it], vector, current_offset);
				current_offset += encoder->childrens[it]->vector_size;
			}
			
			result = nnl2_nn_sequential_create(encoder->num_childrens, networks);
			
			//free(networks);  Double free. UB. See nnl2_nnlrepr_free (spent 5 hours on that), nnl2_nn_sequential freeing
            
            break;
		}
		
		case nnl2_nn_type_relu: {
            result = nnl2_nn_relu_create();
            break;
        }
        
		case nnl2_nn_type_leaky_relu: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(encoder->additional_data == NULL) {
                    NNL2_ERROR("In function nnl2_nn_decode_internal, Leaky ReLU missing alpha parameter. returning NULL");
                    return NULL;
                }
            #endif
			
            float alpha = *((float*)encoder->additional_data);
            result = nnl2_nn_leaky_relu_create(alpha);
			
            break;
        }
		
		case nnl2_nn_type_sigmoid: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if(encoder->additional_data == NULL) {
                    NNL2_ERROR("In function nnl2_nn_decode_internal, Sigmoid missing approx parameter. returning NULL");
                    return NULL;
                }
            #endif
			
            bool approx = *((bool*)encoder->additional_data);
            result = nnl2_nn_sigmoid_create(approx);
			
            break;
        }
        
		case nnl2_nn_type_tanh: {
            if(encoder->additional_data == NULL) {
                NNL2_ERROR("In function nnl2_nn_decode_internal, Tanh missing approx parameter. returning NULL");
                return NULL;
            }
			
            bool approx = *((bool*)encoder->additional_data);
            result = nnl2_nn_tanh_create(approx);
				
            break;
        }
		
		default: {
			NNL2_ERROR("In function nnl2_nn_decode_internal, unknown nn type (numbering: %d). returning NULL", encoder -> nn_type);
			return NULL;
		}
	}
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("Successfully decoded layer type: %d", encoder->nn_type);
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief 
 * Main decode function 
 * Decodes complete neural network from nnlrepr format
 * 
 ** @param encoder 
 * Pointer to nnlrepr structure containing encoded network
 *
 ** @return void* 
 * Pointer to decoded neural network, NULL on error
 */
void* nnl2_nn_decode(nnl2_nnlrepr* encoder) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(encoder == NULL) {
            NNL2_ERROR("In function nnl2_nn_decode, encoder is NULL. returning NULL");
            return NULL;
        }
	#endif 
	
	void* result = nnl2_nn_decode_internal(encoder->template, encoder->vector, 0);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result == NULL) {
			NNL2_ERROR("In function nnl2_nn_decode, failed to decode neural network. returning NULL");
			return NULL;
		}
	#endif 
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return result;
}

#endif /** NNL2_NNLREPR_DECODE_H **/
