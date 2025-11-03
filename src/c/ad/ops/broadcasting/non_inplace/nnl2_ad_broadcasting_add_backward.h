#ifndef NNL2_AD_ADD_BROADCASTING_BACKWARD_H
#define NNL2_AD_ADD_BROADCASTING_BACKWARD_H

void nnl2_ad_reverse_derivative_add_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* summand, nnl2_ad_tensor* sumend) {
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_sum = summand->grad;
    nnl2_tensor* grad_end = sumend->grad;
    nnl2_tensor* data_end = sumend->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_end = product(data_end->shape, data_end->rank);
    size_t blocks = numel_out / numel_end;
	
    if(summand->requires_grad) nnl2_add_inplace(grad_sum, grad_out);
	
	if(sumend->requires_grad) {
        switch(data_end->dtype) {
            case FLOAT64: {
                nnl2_float64* gout = (nnl2_float64*)grad_out->data;
                nnl2_float64* gend = (nnl2_float64*)grad_end->data;
				
                for (size_t i = 0; i < numel_end; i++) {
                    nnl2_float64 acc = 0.0;
                    for (size_t b = 0; b < blocks; b++) acc += gout[b * numel_end + i];
                    gend[i] += acc;
                }
				
                break;
            }
			
			case FLOAT32: {
                nnl2_float32* gout = (nnl2_float32*)grad_out->data;
                nnl2_float32* gend = (nnl2_float32*)grad_end->data;
				
                for (size_t i = 0; i < numel_end; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks; b++) acc += gout[b * numel_end + i];
                    gend[i] += acc;
                }
				
                break;
            }
			
			case INT32: {
                nnl2_int32* gout = (nnl2_int32*)grad_out->data;
                nnl2_int32* gend = (nnl2_int32*)grad_end->data;
				
                for (size_t i = 0; i < numel_end; i++) {
                    nnl2_int32 acc = 0.0f;
                    for(size_t b = 0; b < blocks; b++) acc += gout[b * numel_end + i];
                    gend[i] += acc;
                }
				
                break;
            }

            default: NNL2_TYPE_ERROR(data_end->dtype);
        }
    }
}

#endif /** NNL2_AD_ADD_BROADCASTING_BACKWARD_H **/
