#ifndef NNL2_TIMESTEP_VIEW_H
#define NNL2_TIMESTEP_VIEW_H

nnl2_tensor* nnl2_tensor_timestep_view(nnl2_tensor* src, int t) {
    //NNL2_DEBUG("[TIMESTEP_VIEW] Creating tensor timestep view, src=%p, t=%d", (void*)src, t);
    
    if(!src) {
        NNL2_ERROR("[TIMESTEP_VIEW] src is NULL");
        return NULL;
    }
    
    //NNL2_DEBUG("[TIMESTEP_VIEW] Source tensor: rank=%d, dtype=%d, is_view=%d", 
    //           src->rank, src->dtype, src->is_view);
    
    if(src->rank != 3) {
        NNL2_ERROR("[TIMESTEP_VIEW] timestep_view expects rank-3 tensor [T, B, F], got rank=%d", src->rank);
        return NULL;
    }
    
    if(t < 0 || t >= src->shape[0]) {
        NNL2_ERROR("[TIMESTEP_VIEW] timestep index out of bounds: t=%d, shape[0]=%d", 
                   t, src->shape[0]);
        return NULL;
    }
    
   // NNL2_DEBUG("[TIMESTEP_VIEW] Source shape: [%d x %d x %d], strides: [%zu x %zu x %zu]",
   //            src->shape[0], src->shape[1], src->shape[2],
   //            src->strides[0], src->strides[1], src->strides[2]);

    nnl2_tensor* view = malloc(sizeof(nnl2_tensor));
    if(!view) {
        NNL2_ERROR("[TIMESTEP_VIEW] Failed to allocate tensor struct");
        return NULL;
    }
    
    //NNL2_DEBUG("[TIMESTEP_VIEW] Tensor struct allocated: %p", (void*)view);

    memset(view, 0, sizeof(nnl2_tensor));
    
    view->rank = 2;
    
    // shape [batch, features]
    view->shape = malloc(2 * sizeof(int));
    //if(!view->shape) {
    //    NNL2_ERROR("[TIMESTEP_VIEW] Failed to allocate shape array");
    //    free(view);
    //    return NULL;
    //}
    
    view->shape[0] = src->shape[1];
    view->shape[1] = src->shape[2];
    
    // strides [features, 1]
    view->strides = malloc(2 * sizeof(size_t));
    if(!view->strides) {
        //NNL2_ERROR("[TIMESTEP_VIEW] Failed to allocate strides array");
        free(view->shape);
        free(view);
        return NULL;
    }
    
    view->strides[0] = src->strides[1];  
    view->strides[1] = src->strides[2];  
    
    view->dtype = src->dtype;
    
    size_t elem_size = get_dtype_size(src->dtype);
    //NNL2_DEBUG("[TIMESTEP_VIEW] Element size: %zu bytes", elem_size);
    
    view->shape[0] = src->shape[1]; 
	view->shape[1] = src->shape[2]; 

	view->strides[1] = 1;  
	view->strides[0] = view->shape[1];  

	view->dtype = src->dtype;
	view->magic_number = TENSOR_MAGIC_ALIVE;

	size_t bytes_per_timestep = view->shape[0] * view->shape[1] * elem_size;  // batch * features * elem_size
	view->data = (char*)src->data + t * bytes_per_timestep;

	//NNL2_DEBUG("[TIMESTEP_VIEW] View data pointer: %p (offset: %zu bytes)", 
	//		   (void*)view->data, t * bytes_per_timestep);
    
    //NNL2_DEBUG("[TIMESTEP_VIEW] View data pointer: %p (offset from src: %p + %zu bytes)", 
     //          (void*)view->data, (void*)src->data, t * src->strides[0] * elem_size);
    
    view->is_view = true;
    
    //NNL2_DEBUG("[TIMESTEP_VIEW] View created successfully: %p, shape: [%d x %d], strides: [%zu x %zu]",
    //           (void*)view, view->shape[0], view->shape[1], view->strides[0], view->strides[1]);
    
    return view;
}

#endif /** NNL2_TIMESTEP_VIEW_H **/
