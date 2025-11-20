#ifndef NNL2_OPTIM_BACKEND_H
#define NNL2_OPTIM_BACKEND_H

#define NNL2_OPTIM_TYPE_ERROR(optim_type) NNL2_ERROR("Unknown optimizer type received (in enum: %d)", optim_type)

typedef enum {
	nnl2_optim_type_gd,
	nnl2_optim_type_unknown
} nnl2_optim_object_type;

typedef struct {
	nnl2_ad_tensor** tensors;
	size_t num_tensors;
} nnl2_optim;

void nnl2_optim_zero_grad(nnl2_optim optim) {
	for(size_t it = 0; it < optim . num_tensors; it++) {
		nnl2_ad_zero_grad(optim . tensors [it]);
	}
}

#endif /** NNL2_OPTIM_BACKEND_H **/
 