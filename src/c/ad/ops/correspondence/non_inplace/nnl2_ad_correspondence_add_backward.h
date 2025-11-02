#ifndef NNL2_AD_CORRESPONDENCE_ADD_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_ADD_BACKWARD_H

void nnl2_ad_reverse_derivative_add_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* inc_tensor, void* inc) {
	(void)inc;
	(void)out_tensor;	
	
	if(inc_tensor->requires_grad) {
		switch(inc_tensor->data->dtype) {
			case FLOAT64: add_incf_inplace(inc_tensor->grad, &(nnl2_float64){1.0});  break;
			case FLOAT32: add_incf_inplace(inc_tensor->grad, &(nnl2_float32){1.0});  break;
			case INT32:   add_incf_inplace(inc_tensor->grad, &(nnl2_int32){1.0});    break;
			
			default: {
				NNL2_TYPE_ERROR(inc_tensor->data->dtype);
				return;
			}
		}
	}
}

#endif /** NNL2_AD_CORRESPONDENCE_ADD_BACKWARD_H **/
