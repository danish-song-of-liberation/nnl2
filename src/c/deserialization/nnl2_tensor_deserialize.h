#ifndef NNL2_TENSOR_DESERIALIZE_H
#define NNL2_TENSOR_DESERIALIZE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <limits.h>

// NNL2 

/** @file nnl2_tensor_deserialize
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains function for .nnlt tensor deserialization
 **/
 
/** @brief Macro for file operation validation **/ 
#define NNL2_VALIDATE_FILE_READ(f, data, size, count, error_label) \
    if(fread((data), (size), (count), (f)) != (count)) { \
        if(feof(f)) { \
            NNL2_ERROR("Unexpected end of file at %s\n", #error_label); \
        } else if(ferror(f)) { \
            NNL2_ERROR("File read error at %s: %s\n", #error_label, strerror(errno)); \
        } \
        errno = EIO; \
        goto error_label; \
    }

/** @brief Macro for file operation validation with size_t **/
#define NNL2_VALIDATE_FILE_READ_SIZED(f, data, size, count, error_label) \
    if(fread((data), (size), (count), (f)) != (size_t)(count)) { \
        NNL2_ERROR("File read size mismatch at %s", #error_label); \
        errno = EIO; \
        goto error_label; \
    }

/** @brief 
 * Validates tensor header fields
 * 
 ** @param ts_type 
 * Object type
 *
 ** @param dtype 
 * Data type
 *
 ** @param magic_number 
 * Magic number for format validation
 *
 ** @param rank 
 * Tensor rank
 *
 ** @return bool 
 * true if valid, false otherwise
 */
static bool validate_tensor_header(nnl2_object_type ts_type, nnl2_tensor_type dtype, int8_t magic_number, int32_t rank) {
	// Validate object type
    if(ts_type != nnl2_type_ts) {
        NNL2_ERROR("In function validate_tensor_header, invalid object type: %d", ts_type);
        return false;
    }
 
    // Validate data type
    if(get_dtype_size(dtype) == 0) {
        NNL2_ERROR("Invalid data type: %d", dtype);
        return false;
    }
    
	// Validate magic number (format signature)
    if(magic_number != TENSOR_MAGIC_ALIVE) {
        NNL2_ERROR("Invalid magic number: %d (expected %d)", magic_number, TENSOR_MAGIC_ALIVE);
        return false;
    }

	// Validate rank
    if(rank < 0) {
        NNL2_ERROR("Invalid tensor rank: %d", rank);
        return false;
    }
    
    return true;
}

/** @brief 
 * Deserializes a tensor from a binary file
 * 
 ** @param path 
 * Path to the binary file containing serialized tensor
 *
 ** @return nnl2_tensor* 
 * Pointer to deserialized tensor, NULL on error
 * 
 ** @warning 
 * Files must be created by nnl2_tensor_serialize()
 *
 ** @warning 
 * Caller must free the tensor using nnl2_tensor_free()
 * 
 ** @example
 ** @code
 * nnl2_tensor* tensor = nnl2_tensor_deserialize("my_awesome_tensor.nnlt");
 * if (tensor != NULL) {
 *     // Use the tensor
 *     ...
 *     nnl2_free_tensor(tensor);
 * }
 ** @endcode
 **/
nnl2_tensor* nnl2_tensor_deserialize(const char* path) {
	FILE* file = NULL;
	nnl2_tensor* tensor = NULL;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if(path == NULL) {
			NNL2_ERROR("In function nnl2_tensor_deserialize, path is NULL");
			return NULL;
		}
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(strlen(path) == 0) {
			NNL2_ERROR("In function nnl2_tensor_deserialize, empty file path provided");
			errno = EINVAL;
			return NULL;
		}
	
		if(strstr(path, "..") != NULL) {
			NNL2_ERROR("In function nnl2_tensor_deserialize, path traversal attempt detected: %s", path);
			errno = EACCES;
			return NULL;
		}
	#endif
	
	file = fopen(path, "rb");
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if(file == NULL) {
			NNL2_ERROR("In function nnl2_tensor_deserialize, failed to open file '%s': %s", path, strerror(errno));
			return NULL;
		}
	#endif
	
	nnl2_object_type ts_type;
    nnl2_tensor_type dtype;
    bool is_view;
    int8_t magic_number;
    int32_t rank;
	
	NNL2_VALIDATE_FILE_READ(file, &ts_type, sizeof(nnl2_object_type), 1, file_error);
    NNL2_VALIDATE_FILE_READ(file, &dtype, sizeof(nnl2_tensor_type), 1, file_error);
    NNL2_VALIDATE_FILE_READ(file, &is_view, sizeof(bool), 1, file_error);
    NNL2_VALIDATE_FILE_READ(file, &magic_number, sizeof(int8_t), 1, file_error);
    NNL2_VALIDATE_FILE_READ(file, &rank, sizeof(int32_t), 1, file_error);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!validate_tensor_header(ts_type, dtype, magic_number, rank)) {
			goto file_error;
		}
	#endif 
	
	tensor = (nnl2_tensor*)malloc(sizeof(nnl2_tensor));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!tensor) {
			NNL2_MALLOC_ERROR();
			goto file_error;
		}
	#endif 
	
	tensor -> ts_type = ts_type;
    tensor -> dtype = dtype;
    tensor -> is_view = is_view;
    tensor -> magic_number = magic_number;
    tensor -> rank = rank;
	
	if(rank > 0) {
        tensor -> shape = (int32_t*)malloc(rank * sizeof(int32_t));
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
			if(tensor -> shape == NULL) {
				NNL2_MALLOC_ERROR();
				goto cleanup_error;
			}
		#endif 
		
        tensor -> strides = (int32_t*)malloc(rank * sizeof(int32_t));
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
			if(tensor -> strides == NULL) {
				NNL2_MALLOC_ERROR();
				free(tensor -> shape);
				tensor -> shape = NULL;
				goto cleanup_error;
			}
		#endif
        
        NNL2_VALIDATE_FILE_READ_SIZED(file, tensor->shape, sizeof(int32_t), rank, cleanup_error);
        NNL2_VALIDATE_FILE_READ_SIZED(file, tensor->strides, sizeof(int32_t), rank, cleanup_error);
		
		for(int32_t i = 0; i < rank; i++) {
            if(tensor -> shape[i] <= 0) {
                NNL2_ERROR("Invalid shape value %d at dimension %d", tensor -> shape[i], i);
                errno = EINVAL;
                goto cleanup_error;
            }
        }
    } 

	size_t total_elements;

	if(tensor -> rank > 0) {
		total_elements = nnl2_product(tensor->shape, rank);
	} else {
		total_elements = 0;
	}
	
    size_t dtype_size = get_dtype_size(dtype);
    
    if(total_elements == 0 || dtype_size == 0) {
        tensor -> data = NULL;
        fclose(file);
        return tensor;
    }
    
	if(total_elements == 0 && rank > 0) {
        NNL2_ERROR("In function nnl2_tensor_deserialize, failed to calculate total elements");
        goto cleanup_error;
    }
     
    if(dtype_size == 0) {
        NNL2_ERROR("In function nnl2_tensor_deserialize, invalid data type size");
        goto cleanup_error;
    }	 
	
	if(total_elements > SIZE_MAX / dtype_size) {
        NNL2_ERROR("In function nnl2_tensor_deserialize, tensor size overflow (elements: %zu, dtype size: %zu)", total_elements, dtype_size);
        errno = EOVERFLOW;
        goto cleanup_error;
    }
		
	size_t total_size = total_elements * dtype_size;
	
	ALLOC_ALIGNED(tensor->data, (size_t)TENSOR_MEM_ALIGNMENT, total_size);	
	if(tensor -> data == NULL) {
        NNL2_ERROR("In function nnl2_tensor_deserialize, Failed to allocate aligned memory for tensor data");
        errno = ENOMEM;
        goto cleanup_error;
    }	
    
	size_t elements_read = fread(tensor->data, dtype_size, total_elements, file);
    if(elements_read != total_elements) {
        NNL2_ERROR("In function nnl2_tensor_deserialize, Data read incomplete. Expected %zu elements, got %zu", total_elements, elements_read);
        FREE_ALIGNED(tensor -> data);
        tensor -> data = NULL;
        goto cleanup_error;
    }
	
	int extra_byte = fgetc(file);
    if(extra_byte != EOF) {
        NNL2_WARN("In function nnl2_tensor_deserialize, File contains extra data beyond tensor structure");
    }
    
    fclose(file);
    return tensor;
	
	cleanup_error: {
		if(tensor) {
			if(tensor -> shape) {
				free(tensor->shape);
			}
			
			if(tensor -> strides) {
				free(tensor->strides);
			}
			
			if(tensor -> data) {
				FREE_ALIGNED(tensor->data);  
			}
			
			free(tensor);
		}
	}
	
	file_error: {
		if(file != NULL) {
			fclose(file);
		}
    }
	
	return NULL;
}

#endif /** NNL2_TENSOR_DESERIALIZE_H **/
