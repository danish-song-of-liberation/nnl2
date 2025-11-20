#ifndef NNL2_OPTIM_GD_H
#define NNL2_OPTIM_GD_H

typedef struct {
	nnl2_optim_object_type optim_type;
	nnl2_optim data;
	nnl2_float32 lr;
} nnl2_optim_gd;

void nnl2_optim_gd_free(nnl2_optim_gd* optim) {
	free(optim);
}

void nnl2_optim_gd_zero_grad(nnl2_optim_gd* optim) {
	nnl2_optim_zero_grad(optim -> data);
}

void nnl2_optim_gd_step(nnl2_optim_gd* gd_optim) {
	for(size_t it = 0; it < gd_optim -> data . num_tensors; it++) {
		nnl2_ad_step_inplace(gd_optim -> data . tensors [it], gd_optim -> lr);
	}
}

nnl2_optim_gd* nnl2_optim_gd_create(nnl2_ad_tensor** tensors, size_t num_tensors, float learning_rate) {
	nnl2_optim_gd* optim = malloc(sizeof(nnl2_optim_gd));
	optim -> lr = learning_rate;
	optim -> optim_type = nnl2_optim_type_gd;
	optim -> data . tensors = tensors;
	optim -> data . num_tensors = num_tensors;
	
	return optim;
}

#endif /** NNL2_OPTIM_GD_H **/
