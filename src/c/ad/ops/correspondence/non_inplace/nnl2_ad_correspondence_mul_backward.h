#ifndef NNL2_AD_CORRESPONDENCE_MUL_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_MUL_BACKWARD_H

void nnl2_ad_reverse_derivative_mul_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* multiplier_tensor, void* multiplier) {
	Tensor* grad_out = out_tensor->grad;
    Tensor* grad_in  = multiplier_tensor->grad;

    switch (multiplier_tensor->data->dtype) {
		case FLOAT64: axpy_inplace(grad_in, grad_out, *((nnl2_float64*)multiplier));    break;
		case FLOAT32: axpy_inplace(grad_in, grad_out, *((nnl2_float32*)multiplier));    break;
		case INT32:   axpy_inplace(grad_in, grad_out, *((nnl2_int32*)multiplier));      break; 
		
		default: NNL2_TYPE_ERROR(multiplier_tensor->data->dtype);
	}
}

#endif /** NNL2_AD_CORRESPONDENCE_MUL_BACKWARD_H **/
