#ifndef NNL2_AD_CORRESPONDENCE_SUB_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_SUB_BACKWARD_H

void nnl2_ad_reverse_derivative_sub_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* dec_tensor, void* dec) {
    (void)dec;
	(void)out_tensor;	
	
	if(dec_tensor->requires_grad) {
		switch(dec_tensor->data->dtype) {
			case FLOAT64: add_incf_inplace(dec_tensor->grad, &(nnl2_float64){1.0});  break;
			case FLOAT32: add_incf_inplace(dec_tensor->grad, &(nnl2_float32){1.0});  break;
			case INT32:   add_incf_inplace(dec_tensor->grad, &(nnl2_int32){1.0});    break;
			
			default: {
				NNL2_TYPE_ERROR(dec_tensor->data->dtype);
				return;
			}
		}
	}
}

#endif /** NNL2_AD_CORRESPONDENCE_SUB_BACKWARD_H **/
