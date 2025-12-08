#ifndef NNL2_ANN_FORWARD_H
#define NNL2_ANN_FORWARD_H

typedef struct nnl2_nn_fnn_struct nnl2_nn_fnn;
typedef struct nnl2_nn_sequential_struct nnl2_nn_sequential;

nnl2_ad_tensor* nnl2_nn_fnn_forward(nnl2_nn_fnn* nn, nnl2_ad_tensor* x);
nnl2_ad_tensor* nnl2_nn_sequential_forward(nnl2_nn_sequential* seq, nnl2_ad_tensor* x);

nnl2_ad_tensor* nnl2_ann_forward(void* model, void** args) { // This is a war crime
	nnl2_nn_ann* ann = (nnl2_nn_ann*)model;
	
	switch(ann -> nn_type) {
		case nnl2_nn_type_fnn: {
			nnl2_ad_tensor* input = (nnl2_ad_tensor*)args[0];
			return nnl2_nn_fnn_forward((nnl2_nn_fnn*)model, input);
		}
		
		case nnl2_nn_type_sequential: {
			nnl2_ad_tensor* input = (nnl2_ad_tensor*)args[0];
			return nnl2_nn_sequential_forward((nnl2_nn_sequential*)model, input);
		}
		
		default: {
			return NULL;
		}
	}
}

#endif /** NNL2_ANN_FORWARD_H **/
