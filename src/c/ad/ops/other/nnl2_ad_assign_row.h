#ifndef NNL2_AD_ASSIGN_ROW_H
#define NNL2_AD_ASSIGN_ROW_H

// NNL2

/** @file nnl2_ad_assign_row.h
 ** @date 2025
 ** @copyright MIT
 ** @brief Row assignment operation for AD tensors
 **/
 
static void nnl2_ad_reverse_backward_assign_row(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if(!tensor) {
            NNL2_ERROR("In function assign_row backward, tensor is NULL");
        }
		
        if(!tensor->roots || !tensor->roots[0]) {
            NNL2_ERROR("In function assign_row backward, src root tensor is NULL");
        }
		
        if(!tensor->extra_field) {
            NNL2_ERROR("In function assign_row backward, seq_index context is missing");
        }
		
        if(!tensor->grad) {
            NNL2_ERROR("In function assign_row backward, dst gradient tensor is NULL");
        }
    #endif
	
	nnl2_ad_tensor* src = tensor->roots[0];
    int seq_index = *(int*)tensor->extra_field;

    if(!src->grad) {
        src->grad = nnl2_zeros(src->data->shape, src->data->rank, src->data->dtype);
    }

    // src->grad += dst->grad[:, seq_index, :]
    nnl2_assign_row_add(src->grad, seq_index, tensor->grad);

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Applies row assignment operation to AD tensors
 *
 ** @param dst
 * Pointer to the destination AD tensor (3D: [batch, seq, features])
 * The tensor will be modified in-place
 *
 ** @param seq_index
 * Index of the sequence position to assign to (0..seq_length-1)
 *
 ** @param src
 * Pointer to the source AD tensor (2D: [batch, features])
 *
 ** @param retain_graph
 * If true, indicates the computation graph should be
 * preserved for backward passes. In-place operations
 * are not allowed when this is true and the tensor
 * requires gradients
 *
 ** @exception NNL2_AD_INPLACE_FATAL
 * Terminates program if destination tensor requires gradients 
 * and retain_graph is true
 *
 ** @exception NNL2_ERROR
 * NULL tensor pointer
 *
 ** @exception NNL2_ERROR
 * Dimension mismatch between tensors
 *
 ** @see nnl2_ad_tensor
 ** @see nnl2_assign_row()
 **
 ** @return void
 **
 ** @code
 * nnl2_ad_tensor* output = nnl2_ad_zeros(..., 3);  // [batch, seq, hidden]
 * nnl2_ad_tensor* hidden = nnl2_ad_tensor(...);    // [batch, hidden]
 * // RNN step
 * nnl2_ad_assign_row(output, t, hidden, false);    // Assign hidden state to time step t
 ** @endcode
 **
 ** @see NNL2_AD_INPLACE_FATAL
 ** @see NNL2_ERROR
 **/
void nnl2_ad_assign_row(nnl2_ad_tensor* dst, int seq_index, nnl2_ad_tensor* src, bool track_graph) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!dst) {
            NNL2_ERROR("dst ad_tensor is NULL (in function nnl2_ad_assign_row)");
        }
        
        if(!src) {
            NNL2_ERROR("src ad_tensor is NULL (in function nnl2_ad_assign_row)");
        }
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
            if(!dst->data) {
                NNL2_ERROR("dst->data is NULL (in function nnl2_ad_assign_row)");
            }
            
            if(!src->data) {
                NNL2_ERROR("src->data is NULL (in function nnl2_ad_assign_row)");
            }
        #endif
    #endif

	// forward
    nnl2_assign_row(dst->data, seq_index, src->data);
	
	if(track_graph && src->requires_grad) {
        dst->num_roots = 1;
        dst->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
        if(!dst->roots) {
            NNL2_MALLOC_ERROR();
            return;
        }

        dst->roots[0] = src;
        dst->backward_fn = nnl2_ad_reverse_backward_assign_row;

        int* idx = malloc(sizeof(int));
        if(!idx) {
            NNL2_MALLOC_ERROR();
            return;
        }
		
        *idx = seq_index;

        dst->extra_field = idx;
        dst->extra_free = free;

        dst->is_leaf = false;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_ASSIGN_ROW_H **/
