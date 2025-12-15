#ifndef NNL2_NNLREPR_ENCODE_H
#define NNL2_NNLREPR_ENCODE_H

/** @brief 
 * Encodes a neural network into nnlrepr format
 * 
 ** @param nn 
 * Pointer to the neural network to encode
 *
 ** @return nnl2_nnlrepr* 
 * Pointer to encoded representation, NULL on error
 */
nnl2_nnlrepr* nnl2_nn_encode(void* nn) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(nn == NULL) {
            NNL2_ERROR("In function nnl2_nn_encode, network pointer is NULL");
            return NULL;
        }
    #endif
	
	nnl2_nnlrepr* result = malloc(sizeof(nnl2_nnlrepr));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (result == NULL) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
	
	nnl2_ad_tensor** parameters = nnl2_ann_parameters(nn);
	size_t num_params = nnl2_ann_num_parameters(nn);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (parameters == NULL && num_params > 0) {
            NNL2_ERROR("In function nnl2_nn_encode, parameters array is NULL but num_params > 0");
            free(result);
            return NULL;
        }
		
        if (num_params > 0 && parameters[0] == NULL) {
            NNL2_ERROR("In function nnl2_nn_encode, first parameter is NULL");
            free(result);
            return NULL;
        }
    #endif
	
	result -> vector = nnl2_ad_vector_concat(parameters, num_params, parameters[0] -> data -> dtype);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result->vector == NULL) {
            NNL2_ERROR("In function nnl2_nn_encode, failed to create concatenated vector");
            free(result);
            return NULL;
        }
    #endif
	
	result -> template = nnl2_ann_nnlrepr_template(nn);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (result->template == NULL) {
            NNL2_ERROR("In function nnl2_nn_encode, failed to create network template");
            nnl2_free_ad_tensor(result->vector);
            free(result);
            return NULL;
        }
    #endif
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("Successfully encoded network with %zu parameters, vector size: %zu", num_params, result->template->vector_size);
        NNL2_FUNC_EXIT();
    #endif
	
	return result;
}

/** @brief 
 * Recursively prints encoder template structure
 * 
 ** @param encoder 
 * Template to print
 *
 ** @param terpri 
 * Whether to print newline after this element
 *
 ** @param depth 
 * Current recursion depth for indentation
 */
void nnl2_nn_print_encoder(nnl2_nnlrepr_template* encoder, bool terpri, int depth) {
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (encoder == NULL) {
            printf("(NULL-ENCODER)");
            if (terpri) printf("\n");
            return;
        }
    #endif
	
	switch(encoder -> nn_type) {
		case nnl2_nn_type_fnn: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
                if (encoder->shapes == NULL || encoder->shapes[0] == NULL) {
                    printf("(fnn INVALID)");
                    break;
                }
            #endif
			
			if(encoder -> num_shapes == 1) {
				printf("(fnn :vector-size %zu :w [%d, %d])%s", 
					   encoder -> vector_size, encoder -> shapes[0][0],
					   encoder -> shapes[0][1], (terpri ? "\n" : ""));
			} else if (encoder->num_shapes == 2) {
				#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
                    if (encoder->shapes[1] == NULL) {
                        printf("(fnn :vector-size %zu :w [%d, %d] :b [INVALID])%s", 
                               encoder->vector_size, encoder->shapes[0][0], 
                               encoder->shapes[0][1], (terpri ? "\n" : ""));
                        break;
                    }
                #endif
				
				printf("(fnn :vector-size %zu :w [%d, %d] :b [%d])%s", 
					   encoder -> vector_size, encoder -> shapes[0][0], 
					   encoder -> shapes[0][1], encoder -> shapes[1][0],
					   (terpri ? "\n" : ""));
			} else {
				printf("(fnn :invalid-num-shapes %zu)", encoder->num_shapes);
			}
			
			break;
		}
		
		case nnl2_nn_type_rnn_cell: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
                if (encoder->shapes == NULL || encoder->shapes[0] == NULL || encoder->shapes[1] == NULL) {
                    printf("(rnn-cell INVALID)");
                    break;
                }
            #endif
			
			if(encoder -> num_shapes == 4) {
				// wxh, whh, bxh, bhh
				printf("(rnn-cell :vector-size %zu :wxh [%d, %d] :whh [%d, %d] :bxh [%d] :bhh [%d])%s",
					   encoder -> vector_size,
					   encoder -> shapes[0][0], encoder -> shapes[0][1],  // wxh
					   encoder -> shapes[1][0], encoder -> shapes[1][1],  // whh
					   encoder -> shapes[2][0],                           // bxh
					   encoder -> shapes[3][0],                           // bhh
					   (terpri ? "\n" : ""));
			} else if(encoder -> num_shapes == 2) {
				printf("(rnn-cell :vector-size %zu :wxh [%d, %d] :whh [%d, %d])%s",
					   encoder -> vector_size,
					   encoder -> shapes[0][0], encoder -> shapes[0][1],  // wxh
					   encoder -> shapes[1][0], encoder -> shapes[1][1],  // whh
					   (terpri ? "\n" : ""));
			} else {
				printf("(rnn-cell :vector-size %zu", encoder -> vector_size);
				for(size_t i = 0; i < encoder->num_shapes; i++) {
					if(encoder->shapes[i][1] == 0) {
						printf(" :tensor%zu [%d]", i, encoder->shapes[i][0]);
					} else {
						printf(" :tensor%zu [%d, %d]", i, encoder->shapes[i][0], encoder->shapes[i][1]);
					}
				}
				
				printf(")%s", (terpri ? "\n" : ""));
			}
			
			break;
		}
		
		
		case nnl2_nn_type_sequential: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
                if (encoder->childrens == NULL) {
                    printf("(sequential :empty)\n");
                    break;
                }
            #endif
			
			printf("(sequential\n");
			for(size_t it = 0; it < encoder -> num_childrens; it++) {
				for(int i = 0; i <= depth; i++) printf("  ");
				
				if(it == (encoder -> num_childrens - 1)) {
					nnl2_nn_print_encoder(encoder -> childrens[it], false, depth + 1);
				} else {
					nnl2_nn_print_encoder(encoder -> childrens[it], true, depth + 1);
				}
			}
			
			printf(")\n\n");
			
			break;
		}
		
		case nnl2_nn_type_relu: {
		    printf("(.relu)%s", (terpri ? "\n" : ""));
			break;
		}
		
		case nnl2_nn_type_leaky_relu: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
                if(encoder->additional_data == NULL) {
                    printf("(.leaky-relu :alpha INVALID)%s", (terpri ? "\n" : ""));
                    break;
                }
            #endif
			
		    printf("(.leaky-relu :alpha %.2f)%s", *((float*)encoder -> additional_data), (terpri ? "\n" : ""));
			break;
		}
		
		case nnl2_nn_type_sigmoid: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
                if(encoder->additional_data == NULL) {
                    printf("(.sigmoid :approx INVALID)%s", (terpri ? "\n" : ""));
                    break;
                }
            #endif
			
		    printf("(.sigmoid :approx %s)%s", (*((bool*)encoder -> additional_data) ? "t" : "nil"), (terpri ? "\n" : ""));
			break;
		}
		
		case nnl2_nn_type_tanh: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
                if (encoder->additional_data == NULL) {
                    printf("(.tanh :approx INVALID)%s", (terpri ? "\n" : ""));
                    break;
                }
            #endif
			
		    printf("(.tanh :approx %s)%s", (*((bool*)encoder -> additional_data) ? "t" : "nil"), (terpri ? "\n" : ""));
			break;
		}
		
		default: {
			printf("(unknown-layer-type-%d)%s", encoder->nn_type, (terpri ? "\n" : ""));
			NNL2_ERROR("In function nnl2_nn_print_encoder, unknown nn type (numbering: %d)", encoder -> nn_type);
		}
	}
	
	return;
}	

#endif /** NNL2_NNLREPR_ENCODE_H **/
