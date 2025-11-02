#ifndef NNL2_AD_MIN_BACKWARD_DERIVATIVE_H
#define NNL2_AD_MIN_BACKWARD_DERIVATIVE_H

void nnl2_ad_reverse_derivative_min(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a, nnl2_ad_tensor* b) {
    //todo fix crutch

    size_t numel = product(a->data->shape, a->data->rank);

    nnl2_tensor_type dtype = a->data->dtype;

    switch (dtype) {
		case FLOAT64: {
			nnl2_float64* ad = (nnl2_float64*)a->data->data;
			nnl2_float64* bd = (nnl2_float64*)b->data->data;
			nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data;

			if (a->requires_grad) {
				nnl2_float64* ag = (nnl2_float64*)a->grad->data;
				for (size_t i = 0; i < numel; i++) if (ad[i] < bd[i]) ag[i] += gd[i];
			}

			if (b->requires_grad) {
				nnl2_float64* bg = (nnl2_float64*)b->grad->data;
				for (size_t i = 0; i < numel; i++) if (bd[i] < ad[i]) bg[i] += gd[i];
			}
			
			break;
		}

		case FLOAT32: {
			nnl2_float32* ad = (nnl2_float32*)a->data->data;
			nnl2_float32* bd = (nnl2_float32*)b->data->data;
			nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;

			if (a->requires_grad) {
				nnl2_float32* ag = (nnl2_float32*)a->grad->data;
				for (size_t i = 0; i < numel; i++) if (ad[i] < bd[i]) ag[i] += gd[i];
			}

			if (b->requires_grad) {
				nnl2_float32* bg = (nnl2_float32*)b->grad->data;
				for (size_t i = 0; i < numel; i++) if (bd[i] < ad[i]) bg[i] += gd[i];
			}
			
			break;
		}

		case INT32: {
			nnl2_int32* ad = (nnl2_int32*)a->data->data;
			nnl2_int32* bd = (nnl2_int32*)b->data->data;
			nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data; 

			if (a->requires_grad) {
				nnl2_float32* ag = (nnl2_float32*)a->grad->data;
				for (size_t i = 0; i < numel; i++) if (ad[i] < bd[i]) ag[i] += gd[i];
			}

			if (b->requires_grad) {
				nnl2_float32* bg = (nnl2_float32*)b->grad->data;
				for (size_t i = 0; i < numel; i++) if (bd[i] < ad[i]) bg[i] += gd[i];
			}
			
			break;
		}

		default: {
			NNL2_TYPE_ERROR(dtype);
			break;
		}
	}
}

#endif /** NNL2_AD_MIN_BACKWARD_DERIVATIVE_H **/
