#ifndef NNL2_AD_TIMESTEP_VIEW_H
#define NNL2_AD_TIMESTEP_VIEW_H

nnl2_ad_tensor* nnl2_ad_tensor_timestep_view(nnl2_ad_tensor* src, int t) {
    //NNL2_DEBUG("[TIMESTEP_VIEW_AD] Creating AD timestep view, src=%p, t=%d", (void*)src, t);
    
    //if(!src) {
    //    NNL2_ERROR("[TIMESTEP_VIEW_AD] Source AD tensor is NULL");
    //    return NULL;
    //}
    
    //NNL2_DEBUG("[TIMESTEP_VIEW_AD] Source tensor: requires_grad=%d, is_leaf=%d, name=%s",
    //           src->requires_grad, src->is_leaf, src->name ? src->name : "NULL");
    
    //if(src->data) {
    //    NNL2_DEBUG("[TIMESTEP_VIEW_AD] Source data shape: [%d x %d x %d]", 
    //               src->data->shape[0], src->data->shape[1], src->data->shape[2]);
    //} else {
    //    NNL2_ERROR("[TIMESTEP_VIEW_AD] Source tensor data is NULL!");
    //    return NULL;
    //}
    
    nnl2_tensor* base_view = nnl2_tensor_timestep_view(src->data, t);
   // if(!base_view) {
    //    NNL2_ERROR("[TIMESTEP_VIEW_AD] Failed to create base tensor view");
     //   return NULL;
    //}
    
    //NNL2_DEBUG("[TIMESTEP_VIEW_AD] Base view created: %p, shape: [%d x %d]", 
             //  (void*)base_view, base_view->shape[0], base_view->shape[1]);

    nnl2_ad_tensor* view = malloc(sizeof(nnl2_ad_tensor));
   // if(!view) {
    //    NNL2_ERROR("[TIMESTEP_VIEW_AD] Failed to allocate AD tensor struct");
    //    nnl2_free_tensor(base_view); 
    //    return NULL;
   // }
    
    //NNL2_DEBUG("[TIMESTEP_VIEW_AD] AD tensor struct allocated: %p", (void*)view);

    memset(view, 0, sizeof(nnl2_ad_tensor));
    
    view->data = base_view;
    
    view->roots = malloc(sizeof(nnl2_ad_tensor*));
    //if(!view->roots) {
    //    NNL2_ERROR("[TIMESTEP_VIEW_AD] Failed to allocate roots array");
    //    free(view);
    //    nnl2_free_tensor(base_view);
    //    return NULL;
    //}
    
    view->roots[0] = src;
    view->num_roots = 1;
    
    view->requires_grad = src->requires_grad;
    view->grad_initialized = false;
    view->is_leaf = false;        
    view->name = NULL;
    view->ts_type = nnl2_type_ad;
	view->magic_number = TENSOR_MAGIC_ALIVE;
    
    view->extra_field = NULL;
    view->extra_free  = NULL;
    
    view->backward_fn = NULL;   
    
    //NNL2_DEBUG("[TIMESTEP_VIEW_AD] AD view created successfully: %p, requires_grad=%d", 
    //           (void*)view, view->requires_grad);
    
    return view;
}

#endif /** NNL2_AD_TIMESTEP_VIEW_H **/
