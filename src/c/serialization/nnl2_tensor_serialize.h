#ifndef NNL2_TENSOR_SERIALIZE_H
#define NNL2_TENSOR_SERIALIZE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <limits.h>

// NNL2 

/** @file nnl2_tensor_serialize
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains function for .nnlt tensor serialization
 **/

/** @brief Macro for file operation validation **/ 
#define NNL2_VALIDATE_FILE_WRITE(f, data, size, count, error_label) \
    if(fwrite((data), (size), (size_t)(count), (f)) != (size_t)(count)) { \
        NNL2_ERROR("File write error at %s: %s\n", #error_label, strerror(errno)); \
        errno = EIO; \
        goto error_label; \
    }

/** @brief 
 * Serializes a tensor to a binary file
 * 
 ** @param tensor 
 * Pointer to tensor to serialize
 *
 ** @param path 
 * Path to the binary file for serialized tensor
 *
 ** @return bool 
 * true on success, false on error
 * 
 ** @warning 
 * Serialized files can be loaded using nnl2_tensor_deserialize()
 *
 ** @warning 
 * Overwrites existing files at the specified path
 * 
 ** @example
 ** @code
 * nnl2_tensor* tensor = nnl2_tensor_randn((int32_t[]){2, 3, 4}, 3, nnl2_dtype_f32);
 * if (nnl2_tensor_serialize(tensor, "my_tensor.nnlt")) {
 *     // Serialization successful
 *     ...
 * }
 * nnl2_free_tensor(tensor);
 ** @endcode
 **/
bool nnl2_tensor_serialize(nnl2_tensor* tensor, char* path) {
	FILE* file = NULL;
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if(tensor == NULL) {
            NNL2_ERROR("In function nnl2_tensor_serialize, tensor is NULL");
            return false;
        }
        
        if(path == NULL) {
            NNL2_ERROR("In function nnl2_tensor_serialize, path is NULL");
            return false;
        }
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(strlen(path) == 0) {
            NNL2_ERROR("In function nnl2_tensor_serialize, empty file path provided");
            errno = EINVAL;
            return false;
        }
    
        if(strstr(path, "..") != NULL) {
            NNL2_ERROR("In function nnl2_tensor_serialize, path traversal attempt detected: %s", path);
            errno = EACCES;
            return false;
        }
        
        if(tensor->ts_type != nnl2_type_ts) {
            NNL2_ERROR("In function nnl2_tensor_serialize, invalid tensor object type: %d", tensor->ts_type);
            return false;
        }
        
        if(tensor->magic_number != TENSOR_MAGIC_ALIVE) {
            NNL2_ERROR("In function nnl2_tensor_serialize, invalid tensor magic number: %d", tensor->magic_number);
            return false;
        }
        
        if(tensor->rank < 0) {
            NNL2_ERROR("In function nnl2_tensor_serialize, invalid tensor rank: %d", tensor->rank);
            return false;
        }
    #endif
	
	file = fopen(path, "wb");
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(file == NULL) {
            NNL2_ERROR("In function nnl2_tensor_serialize, failed to open file '%s': %s", path, strerror(errno));
            return false;
        }
    #endif
	
	NNL2_VALIDATE_FILE_WRITE(file, &(tensor->ts_type), sizeof(nnl2_object_type), 1, file_error);
    NNL2_VALIDATE_FILE_WRITE(file, &(tensor->dtype), sizeof(nnl2_tensor_type), 1, file_error);
    NNL2_VALIDATE_FILE_WRITE(file, &(tensor->is_view), sizeof(bool), 1, file_error);
    NNL2_VALIDATE_FILE_WRITE(file, &(tensor->magic_number), sizeof(int8_t), 1, file_error);
    NNL2_VALIDATE_FILE_WRITE(file, &(tensor->rank), sizeof(int32_t), 1, file_error);

	if(tensor->rank > 0) {
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(tensor->shape == NULL) {
                NNL2_ERROR("In function nnl2_tensor_serialize, tensor shape is NULL for rank %d", tensor->rank);
                goto file_error;
            }
            
            if(tensor->strides == NULL) {
                NNL2_ERROR("In function nnl2_tensor_serialize, tensor strides is NULL for rank %d", tensor->rank);
                goto file_error;
            }
        #endif
		
		NNL2_VALIDATE_FILE_WRITE(file, tensor->shape, sizeof(int32_t), tensor->rank, file_error);
        NNL2_VALIDATE_FILE_WRITE(file, tensor->strides, sizeof(int32_t), tensor->rank, file_error);
	}

	size_t total_elements = product(tensor->shape, tensor->rank);
	size_t dtype_size = get_dtype_size(tensor->dtype);
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(total_elements == 0 && tensor->rank > 0) {
            NNL2_ERROR("In function nnl2_tensor_serialize, failed to calculate total elements");
            goto file_error;
        }
        
        if(dtype_size == 0) {
            NNL2_ERROR("In function nnl2_tensor_serialize, invalid data type size");
            goto file_error;
        }
        
        if(total_elements > SIZE_MAX / dtype_size) {
            NNL2_ERROR("In function nnl2_tensor_serialize, tensor size overflow (elements: %zu, dtype size: %zu)", total_elements, dtype_size);
            errno = EOVERFLOW;
            goto file_error;
        }
        
        if(tensor->data == NULL && total_elements > 0) {
            NNL2_ERROR("In function nnl2_tensor_serialize, tensor data is NULL for non-empty tensor");
            goto file_error;
        }
    #endif
	
	if(total_elements > 0 && tensor->data != NULL) {
        NNL2_VALIDATE_FILE_WRITE(file, tensor->data, dtype_size, total_elements, file_error);
    }
    
    if(fflush(file) != 0) {
        NNL2_ERROR("In function nnl2_tensor_serialize, failed to flush file: %s", strerror(errno));
        goto file_error;
    }
	
	fclose(file);
	
	return true;
	
	file_error: {
		if(file != NULL) {
			fclose(file);
			remove(path);
		}
		
		return false;
	}
}

#endif /** NNL2_TENSOR_SERIALIZE_H **/
