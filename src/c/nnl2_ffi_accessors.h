#ifndef NNL2_FFI_ACCESSORS_H
#define NNL2_FFI_ACCESSORS_H

#include <stddef.h>

/** @brief 
 * Fast 32-bit float array element retrieval (for lisp)
 * 
 ** @param data 
 * Pointer to float array
 *
 ** @param index 
 * Element index 
 *
 ** @return 
 * Value at specified index
 */
nnl2_float32 nnl2_fast_float32_get(nnl2_float32* data, size_t index) {
    return data[index];
}

/** @brief 
 * Fast 32-bit float array element assignment (for lisp)
 * 
 ** @param data 
 * Pointer to float array
 *
 ** @param index 
 * Element index 
 *
 ** @param value 
 * Value to assign
 */
void nnl2_fast_float32_set(nnl2_float32* data, size_t index, float value) {
    data[index] = value;
}

/** @brief 
 * Fast 64-bit double array element retrieval (for lisp)
 * 
 ** @param data 
 * Pointer to double array
 *
 ** @param index 
 * Element index 
 *
 ** @return 
 * Value at specified index
 */
nnl2_float64 nnl2_fast_float64_get(nnl2_float64* data, size_t index) {
    return data[index];
}

/** @brief 
 * Fast 64-bit double array element assignment (for lisp)
 * 
 ** @param data 
 * Pointer to double array
 *
 ** @param index 
 * Element index 
 *
 ** @param value 
 * Value to assign
 */
void nnl2_fast_float64_set(nnl2_float64* data, size_t index, double value) {
    data[index] = value;
}

/** @brief 
 * Fast 32-bit integer array element retrieval (for lisp)
 * 
 ** @param data 
 * Pointer to int array
 *
 ** @param index 
 * Element index 
 *
 ** @return 
 * Value at specified index
 */
nnl2_int32 nnl2_fast_int32_get(nnl2_int32* data, size_t index) {
    return data[index];
}

/** @brief 
 * Fast 32-bit integer array element assignment (for lisp)
 * 
 ** @param data 
 * Pointer to int array
 *
 ** @param index 
 * Element index 
 *
 ** @param value 
 * Value to assign
 */
void nnl2_fast_int32_set(nnl2_int32* data, size_t index, int value) {
    data[index] = value;
}

#endif /** NNL2_FFI_ACCESSORS_H **/
