#ifndef NNL2_AD_CORRESPONDENCE_DIV_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_DIV_BACKWARD_H

void nnl2_ad_reverse_derivative_div_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* divisor_tensor, void* divisor) {
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_in  = divisor_tensor->grad;

    switch (divisor_tensor->data->dtype) {
        case FLOAT64: {
            nnl2_float64 c = *((nnl2_float64*)divisor);
            axpy_inplace(grad_in, grad_out, 1.0 / c);
            break;
        }
		
        case FLOAT32: {
            nnl2_float32 c = *((nnl2_float32*)divisor);
            axpy_inplace(grad_in, grad_out, 1.0f / c);
            break;
        }
		
        case INT32: {
            nnl2_int32 c = *((nnl2_int32*)divisor);
            axpy_inplace(grad_in, grad_out, 1.0f / (float)c);
            break;
        }
		
        default: NNL2_TYPE_ERROR(divisor_tensor->data->dtype);
    }
}

#endif /** NNL2_AD_CORRESPONDENCE_DIV_BACKWARD_H **/
