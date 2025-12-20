#ifndef NNL2_AD_ASSIGN_ROW_H
#define NNL2_AD_ASSIGN_ROW_H

// NNL2

/** @file nnl2_ad_assign_row.h
 ** @date 2025
 ** @copyright MIT
 ** @brief Row assignment operation for AD tensors
 **/

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
void nnl2_ad_assign_row(nnl2_ad_tensor* dst, int seq_index, nnl2_ad_tensor* src, bool retain_graph) {
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

    if(dst->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".assign_row! (row assignment in-place)", dst);
    }
    
    nnl2_assign_row(dst->data, seq_index, src->data);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_ASSIGN_ROW_H **/
