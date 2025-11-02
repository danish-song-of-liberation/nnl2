#ifndef NNL2_AD_TANH_BACKWARD_H
#define NNL2_AD_TANH_BACKWARD_H

void nnl2_ad_reverse_derivative_tanh(nnl2_ad_tensor* ad_tensor, nnl2_ad_tensor* tensor_root, bool approx) {
	(void)approx;
	
    nnl2_tensor* output = ad_tensor->data;
    nnl2_tensor* grad_output = ad_tensor->grad;
    nnl2_tensor* grad_input = tensor_root->grad;

    size_t total_elems = product(output->shape, output->rank);

    switch (output->dtype) {
        case FLOAT64: {
            double* y = (double*)output->data;
            double* grad_out = (double*)grad_output->data;
            double* grad_in = (double*)grad_input->data;
		
            for(size_t i = 0; i < total_elems; i++) {
                double t = y[i];
                grad_in[i] += grad_out[i] * (1.0 - t * t);
            }
			
            break;
        }

        case FLOAT32: {
            float* y = (float*)output->data;
            float* grad_out = (float*)grad_output->data;
            float* grad_in = (float*)grad_input->data;
			
            for(size_t i = 0; i < total_elems; i++) {
                float t = y[i];
                grad_in[i] += grad_out[i] * (1.0f - t * t);
            }
			
            break;
        }

        case INT32: {
            double* y = (double*)output->data;
            double* grad_out = (double*)grad_output->data;
            double* grad_in = (double*)grad_input->data;
			
            for(size_t i = 0; i < total_elems; i++) {
                double t = y[i];
                grad_in[i] += grad_out[i] * (1.0 - t * t);
            }
			
            break;
        }

        default: {
            NNL2_TYPE_ERROR(output->dtype);
            return;
        }
    }
}

#endif /** NNL2_AD_TANH_BACKWARD_H **/
