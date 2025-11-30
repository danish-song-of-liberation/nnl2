#ifndef NNL2_OPTIM_ACCESSORS_H
#define NNL2_OPTIM_ACCESSORS_H

/** @brief
 * Gets the optimizer type from GD optimizer
 *
 ** @note
 * Lisp wrapper for accessing nnl2_optim_gd.optim_type
 */
nnl2_optim_object_type nnl2_optim_gd_optim_type_getter(nnl2_optim_gd* optim) {
    return optim->optim_type;
}

/** @brief
 * Sets the optimizer type in GD optimizer
 *
 ** @note
 * Lisp wrapper for modifying nnl2_optim_gd.optim_type
 */
void nnl2_optim_gd_optim_type_setter(nnl2_optim_gd* optim, nnl2_optim_object_type new_optim_type) {
    optim->optim_type = new_optim_type;
}

/** @brief
 * Gets the optimizer data from GD optimizer
 *
 ** @note
 * Lisp wrapper for accessing nnl2_optim_gd.data
 */
nnl2_optim nnl2_optim_gd_data_getter(nnl2_optim_gd* optim) {
    return optim->data;
}

/** @brief
 * Sets the optimizer data in GD optimizer
 *
 ** @note
 * Lisp wrapper for modifying nnl2_optim_gd.data
 */
void nnl2_optim_gd_data_setter(nnl2_optim_gd* optim, nnl2_optim new_data) {
    optim->data = new_data;
}

/** @brief
 * Gets the learning rate from GD optimizer
 *
 ** @note
 * Lisp wrapper for accessing nnl2_optim_gd.lr
 */
nnl2_float32 nnl2_optim_gd_lr_getter(nnl2_optim_gd* optim) {
    return optim->lr;
}

/** @brief
 * Sets the learning rate in GD optimizer
 *
 ** @note
 * Lisp wrapper for modifying nnl2_optim_gd.lr
 */
void nnl2_optim_gd_lr_setter(nnl2_optim_gd* optim, nnl2_float32 new_lr) {
    optim->lr = new_lr;
}

/** @brief
 * Gets the tensors array from optimizer
 *
 ** @note
 * Lisp wrapper for accessing nnl2_optim.tensors
 */
nnl2_ad_tensor** nnl2_optim_tensors_getter(nnl2_optim* optim) {
    return optim->tensors;
}

/** @brief
 * Sets the tensors array in optimizer
 *
 ** @note
 * Lisp wrapper for modifying nnl2_optim.tensors
 */
void nnl2_optim_tensors_setter(nnl2_optim* optim, nnl2_ad_tensor** new_tensors) {
    optim->tensors = new_tensors;
}

/** @brief
 * Gets the number of tensors from optimizer
 *
 ** @note
 * Lisp wrapper for accessing nnl2_optim.num_tensors
 */
size_t nnl2_optim_num_tensors_getter(nnl2_optim* optim) {
    return optim->num_tensors;
}

/** @brief
 * Sets the number of tensors in optimizer
 *
 ** @note
 * Lisp wrapper for modifying nnl2_optim.num_tensors
 */
void nnl2_optim_num_tensors_setter(nnl2_optim* optim, size_t new_num_tensors) {
    optim->num_tensors = new_num_tensors;
}

#endif /** NNL2_OPTIM_ACCESSORS_H **/
