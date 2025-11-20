#ifndef NNL2_OPTIM_FREE_H
#define NNL2_OPTIM_FREE_H

void nnl2_optim_free(void* optim) {
	if (optim == NULL) return;

	nnl2_optim_object_type* type_optim = ((nnl2_optim_object_type *) optim);
	
	switch(*type_optim) {
		case nnl2_optim_type_gd: {
			nnl2_optim_gd_free(optim);
			break;
		}
		
		case nnl2_optim_type_unknown:	
		
		default: {
			NNL2_OPTIM_TYPE_ERROR(*type_optim);
			return;
		}
	}
}

#endif /** NNL2_OPTIM_FREE_H **/ 
