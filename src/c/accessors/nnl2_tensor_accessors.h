#ifndef NNL2_TENSOR_ACCESSORS_H
#define NNL2_TENSOR_ACCESSORS_H

#include "../nnl2_core.h"
#include "../nnl2_log.h"
#include "../nnl2_tensor_backend.h"
#include "../nnl2_backend_system_docs.h"

#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

/// NNL2

/** @file nnl2_tensor_accessors.h
 ** @brief Contains all operations for accessing tensors of type tref
 ** @copyright MIT License
 ** @date 2025
 *
 * The file fully includes tensor accessors, 
 * i.e. tref getters and setters, as well as 
 * some additional functions
 *
 ** Filepath: nnl2/src/c/accessors/nnl2_tensor_accessors.h
 ** File: nnl2_tensor_accessors.h
 **
 ** The file contains tensor accessories
 **
 ** Note:
 **   You can find many backend declarations in 
 **   the code, and you can view their full 
 **   documentation in the nnl2_backend_system_docs.h 
 **   file (in the same directory)
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2		
 **/

/** @brief
 * Gets an element or a subtensor from a tensor using the specified indices
 * Uses precomputed strides for optimal performance
 *
 ** @param tensor
 * Pointer to the source tensor from which to extract elements or subtensors
 * Must be a valid tensor with properly initialized strides and shape arrays
 *
 ** @param indices
 * An array of indices for accessing tensor elements along each dimension
 * Indices are applied in row-major order (first index for outermost dimension)
 * Partial indexing is supported for creating subtensors
 *
 ** @param num_indices
 * The number of indices in the indices array
 * Can range from 0 (return full tensor view) to tensor->rank (return single element)
 *
 ** @return
 * If num_indices == tensor->rank returns pointer to the specific element in tensor data
 * If num indices < tensor->rank returns pointer to a subtensor
 * NULL in case of any error or invalid parameters
 *
 ** @note
 * When returning a subtensor, it creates a view that shares data with the original tensor
 * Modifications to the subtensor will affect the original tensor and vice versa
 *
 ** @note
 * The function performs index boundary checks based on the safety level
 *
 ** @note
 * Uses tensor->strides for efficient offset calculation
 *
 ** @details
 * The function firstly:
 *
 *** Сhecks the parameters for correctness
 ** Then
 *** Calculates the shift using steps
 ** Finally
 *** Returns a result based on the received 
 *** data, namely a scalar or a subtensor
 *
 ** @code
 * // Example 1: Access single element from 3D tensor
 * int indices[] = {1, 2, 3};
 * float* element = (float*)nnl2_naive_view(tensor3d, indices, 3);
 *
 * // Example 2: Create 2D slice from 3D tensor
 * int slice_indices[] = {1};
 * Tensor* slice = (Tensor*)nnl2_naive_view(tensor3d, slice_indices, 1);
 ** @endcode
 **
 ** @warning
 * The returned subtensor is a view that shares data with the original tensor
 * Freeing the original tensor while subtensors exist will cause dangling pointers
 *
 ** @see nnl2_free_tensor()
 ** @see get_dtype_size()
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 **
 ** @exception[invalid_argument]
 * If tensor is NULL, indices is NULL, or tensor structure is invalid
 *
 ** @exception[out_of_range]
 * If num_indices exceeds tensor rank or any index is out of bounds
 *
 ** @exception[out_of_memory]
 * If memory allocation fails for subtensor structure or arrays
 *
 **/
void* nnl2_naive_view(Tensor* tensor, const int32_t* indices, uint8_t num_indices) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Parameter validation and safety checks
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (tensor == NULL) {
			NNL2_ERROR("Null tensor pointer in view");
			return NULL;
		}
    
		if (indices == NULL) {
			NNL2_ERROR("Null indices pointer in view");
			return NULL;
		}
	#endif
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (tensor->rank <= 0 || tensor->shape == NULL || tensor->data == NULL) {
			NNL2_ERROR("Invalid tensor structure in view");
			return NULL;
		}
	#endif
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (num_indices > tensor->rank) {
			NNL2_ERROR("Too many indices (%u > %d) in view", num_indices, tensor->rank);
			return NULL;
		}
		
		// Validate each index against corresponding dimension bounds	
		for (uint8_t i = 0; i < num_indices; i++) {
			if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
				NNL2_ERROR("Index %u (%d) out of bounds for dimension %u (size %d) in view",
							i, indices[i], i, tensor->shape[i]);
						
				return NULL;
			}
		}
	#endif
	
	// Offset calculation using precomputed strides
    size_t offset = 0;

    for (uint8_t i = 0; i < num_indices; i++) {
        offset += indices[i] * tensor->strides[i];	
    }
	
	// Result construction
    if (num_indices == tensor->rank) {
        const size_t element_size = get_dtype_size(tensor->dtype);
		return (char*)tensor->data + offset * element_size;
    }

    Tensor* subtensor = (Tensor*)malloc(sizeof(Tensor));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!subtensor) {
			NNL2_ERROR("Failed to allocate subtensor in view");
			return NULL;
		}
	#endif

    subtensor->dtype = tensor->dtype;
    subtensor->rank = tensor->rank - num_indices;
	subtensor->is_view = true;
        
	 // Allocate and copy remaining shape dimensions	
    subtensor->shape = (int32_t*)malloc(subtensor->rank * sizeof(int32_t));
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!subtensor->shape) {
			NNL2_ERROR("Failed to allocate subtensor shape in view");
			free(subtensor);
			return NULL;
		}
	#endif
    
	// Copy shape information for non-indexed dimensions
	memcpy(subtensor->shape, tensor->shape + num_indices, subtensor->rank * sizeof(int32_t));

	// Allocate and copy corresponding strides
	// Strides for remaining dimensions remain valid since they're precomputed
	subtensor->strides = (int32_t*)malloc(subtensor->rank * sizeof(int32_t));
	
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!subtensor->strides) {
            NNL2_ERROR("Failed to allocate subtensor strides in view");
            free(subtensor->shape);
            free(subtensor);
            return NULL;
        }
    #endif
	
	// Copy stride information for non-indexed dimensions
	memcpy(subtensor->strides, tensor->strides + num_indices, subtensor->rank * sizeof(int32_t));

	// Set data pointer to shared memory with calculated offset
    const size_t element_size = get_dtype_size(tensor->dtype);
    subtensor->data = (char*)tensor->data + offset * element_size;

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif

    return subtensor;
}

/** @ingroup backend_system
 ** @brief Backend implementations for view
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_tref_getter: Basic reference implementation
 *
 ** @see REGISTER_BACKEND
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_naive_view
 ** @see nnl2_naive
 **/
Implementation nnl2_view_backends[] = {
	REGISTER_BACKEND(nnl2_naive_view, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for the active view backend 
 * @ingroup backend_system 
 */
viewfn nnl2_view;

/** 
 * @brief Sets the backend for view
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void nnl2_set_view_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nnl2_view_backends, nnl2_view, backend_name);
}

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(nnl2_view);

/** 
 * @brief Gets the name of the active backend for view
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* nnl2_get_view_backend() {
	return CURRENT_BACKEND(nnl2_view);
}

/** 
 * @brief Function declaration for getting all `view` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(nnl2_view);

/**
 * @brief Function declaration for getting the number of all `view` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(nnl2_view);

/** @brief
 * Gets an independent copy of an element or a subtensor from a tensor using the specified indices
 * Creates deep copies instead of views for complete data independence
 *
 ** @param tensor
 * Pointer to the source tensor from which to extract elements or subtensors
 * Must be a valid tensor with properly initialized strides and shape arrays
 *
 ** @param indices
 * An array of indices for accessing tensor elements along each dimension
 * Indices are applied in row-major order (first index for outermost dimension)
 * Partial indexing is supported for creating subtensors
 *
 ** @param num_indices
 * The number of indices in the indices array
 * Can range from 0 (return full tensor copy) to tensor->rank (return single element copy)
 *
 ** @return
 * If num_indices == tensor->rank returns pointer to a copy of the specific element
 * If num_indices < tensor->rank returns pointer to an independent subtensor copy
 * NULL in case of any error or invalid parameters
 *
 ** @note
 * When returning a subtensor, it creates an independent copy that does not share data
 * with the original tensor. Modifications to the copy will not affect the original tensor
 *
 ** @note
 * The function performs index boundary checks based on the safety level
 *
 ** @note
 * Uses tensor->strides for efficient offset calculation
 *
 ** @details
 * The function:
 * 1. Validates parameters and performs safety checks
 * 2. Calculates the memory offset using strides
 * 3. Creates independent copies of data instead of views
 *
 ** @code
 * // Example 1: Copy single element from 3D tensor
 * int indices[] = {1, 2, 3};
 * float* element_copy = (float*)nnl2_naive_copy(tensor3d, indices, 3);
 *
 * // Example 2: Create independent 2D slice from 3D tensor
 * int slice_indices[] = {1};
 * Tensor* slice_copy = (Tensor*)nnl2_naive_copy(tensor3d, slice_indices, 1);
 ** @endcode
 **
 ** @warning
 * The returned data must be manually freed when no longer needed
 * For subtensors use nnl2_free_tensor()
 * For single elements use FREE_ALIGNED()
 *
 ** @see nnl2_free_tensor()
 ** @see get_dtype_size()
 ** @see ALLOC_ALIGNED
 ** @see FREE_ALIGNED
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 **
 ** @exception[invalid_argument]
 * If tensor is NULL, indices is NULL, or tensor structure is invalid
 *
 ** @exception[out_of_range]
 * If num_indices exceeds tensor rank or any index is out of bounds
 *
 ** @exception[out_of_memory]
 * If memory allocation fails for data copying
 *
 **/
void* nnl2_naive_tref_getter(Tensor* tensor, const int32_t* indices, uint8_t num_indices) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Parameter validation and safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (tensor == NULL) {
            NNL2_ERROR("Null tensor pointer in copy");
            return NULL;
        }
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (indices == NULL) {
            NNL2_ERROR("Null indices pointer in copy");
            return NULL;
        }
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if (tensor->rank <= 0 || tensor->shape == NULL || tensor->data == NULL) {
            NNL2_ERROR("Invalid tensor structure in copy");
            return NULL;
        }
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (num_indices > tensor->rank) {
            NNL2_ERROR("Too many indices (%u > %d) in copy", num_indices, tensor->rank);
            return NULL;
        }

        for (uint8_t i = 0; i < num_indices; i++) {
            if (indices[i] < 0 || indices[i] >= tensor->shape[i]) {
                NNL2_ERROR("Index %u (%d) out of bounds for dimension %u (size %d) in copy",
                            i, indices[i], i, tensor->shape[i]);
                return NULL;
            }
        }
    #endif
    
    // Offset calculation using precomputed strides
    size_t offset = 0;
    for (uint8_t i = 0; i < num_indices; i++) {
        offset += indices[i] * tensor->strides[i];    
    }
    
    const size_t element_size = get_dtype_size(tensor->dtype);

    // Result construction with independent copies
    if (num_indices == tensor->rank) {
        // Copy single element using aligned allocation
        void* element_copy = NULL;
        ALLOC_ALIGNED(element_copy, TENSOR_MEM_ALIGNMENT, element_size);
		
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if (!element_copy) {
                NNL2_ERROR("Failed to allocate element copy");
				free(element_copy);
                return NULL;
            }
        #endif
        
        memcpy(element_copy, (char*)tensor->data + offset * element_size, element_size);
        return element_copy;
    }

    // Create independent subtensor copy
    Tensor* subtensor = (Tensor*)malloc(sizeof(Tensor));
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!subtensor) {
            NNL2_ERROR("Failed to allocate subtensor in copy");
            return NULL;
        }
    #endif

    subtensor->dtype = tensor->dtype;
    subtensor->rank = tensor->rank - num_indices;
    
    // Allocate and copy shape dimensions
    subtensor->shape = (int32_t*)malloc(subtensor->rank * sizeof(int32_t));
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!subtensor->shape) {
            NNL2_ERROR("Failed to allocate subtensor shape in copy");
            free(subtensor);
            return NULL;
        }
    #endif
    
    memcpy(subtensor->shape, tensor->shape + num_indices, subtensor->rank * sizeof(int32_t));

    // Allocate and copy strides
    subtensor->strides = (int32_t*)malloc(subtensor->rank * sizeof(int32_t));
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!subtensor->strides) {
            NNL2_ERROR("Failed to allocate subtensor strides in copy");
            free(subtensor->shape);
            free(subtensor);
            return NULL;
        }
    #endif
    
    memcpy(subtensor->strides, tensor->strides + num_indices, subtensor->rank * sizeof(int32_t));

    // Calculate total elements in subtensor
    size_t numel = 1;
    for (int i = 0; i < subtensor->rank; i++) {
        numel *= subtensor->shape[i];
    }

    // Allocate independent aligned memory for data and copy from source
    size_t data_size = numel * element_size;
    void* data = NULL;
    ALLOC_ALIGNED(data, TENSOR_MEM_ALIGNMENT, data_size);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!data) {
            NNL2_ERROR("Failed to allocate subtensor data in copy");
            free(subtensor->strides);
            free(subtensor->shape);
            free(subtensor);
            return NULL;
        }
    #endif
    
    subtensor->data = data;
    
    // Copy the contiguous block of data for the subtensor
    char* source_ptr = (char*)tensor->data + offset * element_size;
    memcpy(subtensor->data, source_ptr, data_size);

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif

    return subtensor;
}

/** @ingroup backend_system
 ** @brief Backend implementations for tref (getter)
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_tref_getter: Basic reference implementation
 *
 ** @see REGISTER_BACKEND
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_naive_tref_getter
 ** @see nnl2_naive
 **/
Implementation nnl2_tref_getter_backends[] = {
	REGISTER_BACKEND(nnl2_naive_tref_getter, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for the active tref (getter) 
 * @ingroup backend_system 
 */
trefgetterfn nnl2_tref_getter;

/** 
 * @brief Sets the backend for tref (getter)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void nnl2_set_tref_getter_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nnl2_tref_getter_backends, nnl2_tref_getter, backend_name);
}

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(nnl2_tref_getter);

/** 
 * @brief Gets the name of the active backend for tref (getter)
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* nnl2_get_tref_getter_backend() {
	return CURRENT_BACKEND(nnl2_tref_getter);
}

/** 
 * @brief Function declaration for getting all `tref (getter)` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(nnl2_tref_getter);

/**
 * @brief Function declaration for getting the number of all `tref (getter)` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(nnl2_tref_getter);

/** @brief
 * Sets a subtensor or single element in the destination tensor
 *
 ** @param dest
 * Destination tensor to modify
 *
 ** @param dest_shape
 * Array specifying the starting indices for each dimension
 *
 ** @param dest_rank
 * Number of dimensions specified in dest_shape
 *
 ** @param src
 * Source tensor to copy from
 *
 ** @param src_shape
 * Array specifying which indices to copy from source
 *
 ** @param src_rank
 * Number of dimensions specified in src_shape
 */
void nnl2_tensor_set_subtensor(Tensor* dest, int* dest_shape, int dest_rank, Tensor* src, int* src_shape, int src_rank);

/** @brief
 * Naive tensor reference setter for assignment operations
 *
 * Handles assignment of both scalar values and subtensors to a target tensor
 * Supports wildcard indexing (-1) for iterating through dimensions
 * In lisp wrapper, all '* are automatically converted to -1
 *
 ** @param tensor
 * Target tensor to modify
 *
 ** @param shape
 * Array of indices specifying the target location
 *
 ** @param rank
 * Number of dimensions specified in shape array
 *
 ** @param change_with
 * Pointer to the data to assign (scalar or tensor)
 *
 ** @param is_tensor
 * Flag indicating if change_with is a tensor (true) or scalar (false)
 */
void nnl2_naive_tref_setter(Tensor* tensor, int* shape, int rank, void* change_with, bool is_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    TensorType tensor_dtype = tensor->dtype;

	// Handle tensor assignment (subtensor to subtensor)
    if (is_tensor) {
        Tensor* sub_tensor = (Tensor*)change_with;

		// Additional checks depending on the security level
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if (tensor->rank - rank != sub_tensor->rank) {
				NNL2_ERROR("Rank mismatch in tensor assignment");
				return;
			}
			
			if (tensor->dtype != sub_tensor->dtype) {
				NNL2_ERROR("Data type mismatch in tensor assignment");
				return;
			}
		#endif
        
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			// Validate shape compatibility for all dimensionss
			for (int i = 0; i < sub_tensor->rank; i++) {
				if (tensor->shape[rank + i] != sub_tensor->shape[i]) {
					NNL2_ERROR("Shape mismatch in dimension %d", i);
					return;
				}
			}
		#endif
        
        int* full_dest_shape = malloc(sizeof(int) * tensor->rank);
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if (!full_dest_shape) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
		#endif
        
        memcpy(full_dest_shape, shape, sizeof(int) * rank);
        
        for (int i = rank; i < tensor->rank; i++) {
            full_dest_shape[i] = -1;
        }
        
        int* src_iter_shape = malloc(sizeof(int) * sub_tensor->rank);
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if (!src_iter_shape) {
				free(full_dest_shape);
				NNL2_ERROR("Memory allocation failed");
				return;
			}
		#endif
        
        for (int i = 0; i < sub_tensor->rank; i++) {
            src_iter_shape[i] = -1;
        }
        
        nnl2_tensor_set_subtensor(tensor, full_dest_shape, rank, sub_tensor, src_iter_shape, 0);
        
        free(src_iter_shape);
        free(full_dest_shape);
		
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_FUNC_EXIT();
		#endif
		
        return;
    }
    
	// Handle wildcard indices by recursively iterating through those dimensions
    for(int i = 0; i < rank; i++) {
        if(shape[i] == -1) {
            for(int shape_i = 0; shape_i < tensor->shape[i]; shape_i++) {
                shape[i] = shape_i;
                nnl2_naive_tref_setter(tensor, shape, rank, change_with, is_tensor);
            }
            
            shape[i] = -1;
            return;
        }
    }
    
    if(rank == tensor->rank) {
		// Handling case when reached target element for scalar assignment
        switch(tensor_dtype) {
            case FLOAT64: {
                double* change_elem = (double*)change_with;
                double* elem = (double*)nnl2_view(tensor, shape, rank);
                *elem = *change_elem;
                break;
            }
			
            case FLOAT32: {
                float* change_elem = (float*)change_with;
                float* elem = (float*)nnl2_view(tensor, shape, rank);
                *elem = *change_elem;
                break;
            }
			
            case INT32: {
                int32_t* change_elem = (int32_t*)change_with;
                int32_t* elem = (int32_t*)nnl2_view(tensor, shape, rank);
                *elem = *change_elem;
                break;
            }
			
            default: {
                NNL2_TYPE_ERROR(tensor_dtype);			
                return;
            }
        }
    } else {
		// Handling case when need to go deeper into the tensor hierarchy
        int new_rank = rank + 1;
        int* subshape = malloc(sizeof(int) * new_rank);
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if (!subshape) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
		#endif

        memcpy(subshape, shape, sizeof(int) * rank);

        for (int i = 0; i < tensor->shape[rank]; i++) {
            subshape[rank] = i;
            nnl2_naive_tref_setter(tensor, subshape, new_rank, change_with, is_tensor);
        }

        free(subshape);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief
 * Recursively copies data from source tensor to destination subtensor
 * See docs at declaration 
 *
 ** @see nnl2_tensor_set_subtensor
 **/
void nnl2_tensor_set_subtensor(Tensor* dest, int* dest_shape, int dest_rank, Tensor* src, int* src_shape, int src_rank) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
	
    if (src_rank == src->rank) {
        void* dest_ptr = nnl2_view(dest, dest_shape, dest_rank);
        void* src_ptr = nnl2_view(src, src_shape, src_rank);
        
		// Determine data type size for memcpy
        size_t type_size;
        switch (dest->dtype) {
            case FLOAT64: type_size = sizeof(double); break;
            case FLOAT32: type_size = sizeof(float); break;
            case INT32: type_size = sizeof(int32_t); break;
            default: NNL2_TYPE_ERROR(dest->dtype); return;
        }
        
		// Perform the actual data copy
        memcpy(dest_ptr, src_ptr, type_size);
        return;
    }

    // Handle wildcard indices in source by iterating through that dimension
    if (src_shape[src_rank] == -1) {
        for (int i = 0; i < src->shape[src_rank]; i++) {
            src_shape[src_rank] = i;
            dest_shape[dest_rank] = i;
            nnl2_tensor_set_subtensor(dest, dest_shape, dest_rank + 1, src, src_shape, src_rank + 1);
        }
		
        src_shape[src_rank] = -1;
        dest_shape[dest_rank] = -1;
    } else {
        dest_shape[dest_rank] = src_shape[src_rank];
        nnl2_tensor_set_subtensor(dest, dest_shape, dest_rank + 1, src, src_shape, src_rank + 1);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for tensor reference setter operation
 * @details
 * Array follows the common backend registration pattern for tensor reference setting operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive_tref_setter
 * @see nnl2_naive
 */
Implementation tref_setter_backends[] = {
	REGISTER_BACKEND(nnl2_naive_tref_setter, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for tensor reference setter operation
 * @ingroup backend_system 
 */
trefsetterfn tref_setter;

/** 
 * @brief Sets the backend for tensor reference setter operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 */
void set_tref_setter_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(tref_setter_backends, tref_setter, backend_name);
}

/** @brief
 * Performs element-wise addition with broadcasting (in place)
 *
 ** @details
 * Nick land - Keep the war going. It's pointless.
 *
 ** @param summand
 * Pointer to summand tensor 
 *
 ** @param sumend
 * Pointer to sumend tensor
 */
void naive_add_broadcasting_inplace(Tensor* summand, Tensor* sumend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks for max safety level 
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand->shape, "Summand shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->shape, "Sumend shape is NULL");
	#endif
	
	// Calculate the total number of elements in each tensor
	size_t numel_summand = product(summand->shape, summand->rank);
	size_t numel_sumend = product(sumend->shape, sumend->rank);
	
	// Getting the tensor data types
	TensorType summand_dtype = summand->dtype;
	TensorType sumend_dtype = sumend->dtype;
	
	// Checking the possibility of broadcasting (numel_summand must be a multiple of numel_sumend)
	if((numel_summand % numel_sumend) == 0) {
		// Handling the case where the data types match (more efficiently)
		if(summand_dtype == sumend_dtype) {
			switch(summand_dtype) {
				case FLOAT64: {
					double* cast_summand_data = (double*)summand->data;
					double* cast_sumend_data = (double*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_summand_data = (float*)summand->data;
					float* cast_sumend_data = (float*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_summand_data = (int32_t*)summand->data;
					int32_t* cast_sumend_data = (int32_t*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(summand_dtype);
					return;
				}
			}	
		} else {
			// Handling a case with different data types (conversion required)
			size_t sumend_step = get_dtype_size(sumend_dtype); // The size of the element in bytes
			char* sumend_data = (char*)sumend->data; // Byte pointer for accessing data
			
			switch(summand_dtype) {
				case FLOAT64: {
					double* data_minuend = (double*)summand->data;
                
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							// Get a pointer to the sumend element and convert its type
							void* sumend_elem = sumend_data + i * sumend_step;
							data_minuend[i * numel_sumend + j] += nnl2_convert_to_float64(sumend_elem, sumend_dtype);
						}
					}
                
					break; 
				}
				
				case FLOAT32: {
					float* data_minuend = (float*)summand->data;
                
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							void* sumend_elem = sumend_data + i * sumend_step;
							data_minuend[i * numel_sumend + j] += nnl2_convert_to_float32(sumend_elem, sumend_dtype);
						}
					}
                
					break; 
				}
				
				case INT32: {
					int32_t* data_minuend = (int32_t*)summand->data;
                
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							void* sumend_elem = sumend_data + i * sumend_step;
							data_minuend[i * numel_sumend + j] += nnl2_convert_to_int32(sumend_elem, sumend_dtype);
						}
					}
                
					break; 
				}
				
				default: {
					NNL2_TYPE_ERROR(summand_dtype);
					return;
				}
			}
		}
	} 
	
	else {
		NNL2_ERROR("Cannot broadcast sumend tensor");
		return;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place addition with broadcasting
 * @details
 * Array follows the common backend registration pattern for in-place addition
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place addition with broadcasting
 * 
 * @see nnl2_naive
 * @see naive_add_broadcasting_inplace
 */
Implementation add_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_add_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place addition with broadcasting operation
 * @ingroup backend_system
 */
addbroadcastinginplacefn add_broadcasting_inplace;

/**
 * @brief Sets the backend for in-place addition with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for in-place addition with broadcasting
 * @see SET_BACKEND_BY_NAME
 * @see add_broadcasting_inplace_backends
 */
void set_add_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_broadcasting_inplace_backends, add_broadcasting_inplace, backend_name);
}

/** @brief
 * Performs element-wise addition with broadcasting support
 *
 ** @param summand
 * First tensor to add
 *
 ** @param sumend
 * Second tensor to add
 * 
 ** @return
 * New tensor containing the result of addition
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_add_broadcasting(Tensor* summand, Tensor* sumend) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks for max safety level 
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->shape, "Summand shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->shape, "Sumend shape is NULL", NULL);
	#endif
 
	// Calculate the total number of elements in each tensor
	size_t numel_summand = product(summand->shape, summand->rank);
	size_t numel_sumend = product(sumend->shape, sumend->rank);
	
	// Getting the tensor data types
	TensorType summand_dtype = summand->dtype;
	TensorType sumend_dtype = sumend->dtype;
	
	TensorType winner_in_the_type_hierarchy = MAX(summand_dtype, sumend_dtype);
	
	// Сreating a resultant tensor
	Tensor* result = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
	
	// Checking the possibility of broadcasting (numel_summand must be a multiple of numel_sumend)
	if((numel_summand % numel_sumend) == 0) {
		if(summand_dtype == sumend_dtype) {
			// Handling the case where the data types match (more efficiently)
			switch(summand_dtype) {
				case FLOAT64: {
					double* cast_summand_data = (double*)summand->data;
					double* cast_sumend_data = (double*)sumend->data;
					double* cast_result_data = (double*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_summand_data = (float*)summand->data;
					float* cast_sumend_data = (float*)sumend->data;
					float* cast_result_data = (float*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_summand_data = (int32_t*)summand->data;
					int32_t* cast_sumend_data = (int32_t*)sumend->data;
					int32_t* cast_result_data = (int32_t*)result->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + cast_sumend_data[j];
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(summand_dtype);
					return NULL;
				}
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		} 
		
		else {
			// Handling a case with different data types (conversion required)
			size_t summand_step = get_dtype_size(summand_dtype);
			size_t sumend_step = get_dtype_size(sumend_dtype);
			
			switch(winner_in_the_type_hierarchy) {
				case FLOAT64: {
					double* cast_data_result = (double*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							// Get a pointer to summand element, sumend element and convert its type
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend = cast_sumend_data + j * sumend_step; 
							
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_float64(elem_summand, summand_dtype) + nnl2_convert_to_float64(elem_sumend, sumend_dtype);
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_data_result = (float*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
						for(size_t j = 0; j < numel_sumend; j++) {
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend =  cast_sumend_data + j * sumend_step;
							
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_float32(elem_summand, summand_dtype) + nnl2_convert_to_float32(elem_sumend, sumend_dtype);
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_data_result = (int32_t*)result->data;
					
					char* cast_summand_data = (char*)summand->data;
					char* cast_sumend_data =  (char*)sumend->data;
					
					for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {					
						for(size_t j = 0; j < numel_sumend; j++) {
							void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
							void* elem_sumend =  cast_sumend_data + j * sumend_step;
						
							cast_data_result[i * numel_sumend + j] = nnl2_convert_to_int32(elem_summand, summand_dtype) + nnl2_convert_to_int32(elem_sumend, sumend_dtype);
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
					nnl2_free_tensor(result);
					return NULL;
				}
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		}
	} 
	
	else {
		NNL2_ERROR("Cannot broadcast sumend tensor");
		return NULL;
	}
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for addition with broadcasting
 * @details
 * Array follows the common backend registration pattern for addition
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for addition with broadcasting
 * 
 * @see nnl2_naive
 * @see naive_add_broadcasting
 */
Implementation add_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_add_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for addition with broadcasting operation
 * @ingroup backend_system
 */
addbroadcastingfn add_broadcasting;

/**
 * @brief Sets the backend for addition with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for addition with broadcasting
 * @see SET_BACKEND_BY_NAME
 * @see add_broadcasting_backends
 */
void set_add_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_broadcasting_backends, add_broadcasting, backend_name);
}

/** @brief
 * Performs element-wise subtraction with broadcasting (in place)
 *
 ** @details
 * Subtracts subtrahend tensor from minuend tensor with broadcasting support
 *
 ** @param minuend
 * Pointer to minuend tensor (will be modified in place)
 *
 ** @param subtrahend
 * Pointer to subtrahend tensor
 */
void naive_sub_broadcasting_inplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX     
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->shape, "Minuend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->shape, "Subtrahend shape is NULL");
    #endif
    
    // Calculate the total number of elements in each tensor
    size_t numel_minuend = product(minuend->shape, minuend->rank);
    size_t numel_subtrahend = product(subtrahend->shape, subtrahend->rank);
    
    // Getting the tensor data types
    TensorType minuend_dtype = minuend->dtype;
    TensorType subtrahend_dtype = subtrahend->dtype;
    
    // Checking the possibility of broadcasting (numel_minuend must be a multiple of numel_subtrahend)
    if((numel_minuend % numel_subtrahend) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(minuend_dtype == subtrahend_dtype) {
            switch(minuend_dtype) {
                case FLOAT64: {
                    double* cast_minuend_data = (double*)minuend->data;
                    double* cast_subtrahend_data = (double*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_minuend_data = (float*)minuend->data;
                    float* cast_subtrahend_data = (float*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_minuend_data = (int32_t*)minuend->data;
                    int32_t* cast_subtrahend_data = (int32_t*)subtrahend->data;
                    
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            cast_minuend_data[i * numel_subtrahend + j] -= cast_subtrahend_data[j];
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(minuend_dtype);
                    return;
                }
            }    
        } else {
            // Handling a case with different data types (conversion required)
            size_t subtrahend_step = get_dtype_size(subtrahend_dtype); // The size of the element in bytes
            char* subtrahend_data = (char*)subtrahend->data; // Byte pointer for accessing data
            
            switch(minuend_dtype) {
                case FLOAT64: {
                    double* data_minuend = (double*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            // Get a pointer to the subtrahend element and convert its type
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_float64(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_minuend = (float*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_float32(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_minuend = (int32_t*)minuend->data;
                
                    for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
                        for(size_t j = 0; j < numel_subtrahend; j++) {
                            void* subtrahend_elem = subtrahend_data + j * subtrahend_step;
                            data_minuend[i * numel_subtrahend + j] -= nnl2_convert_to_int32(subtrahend_elem, subtrahend_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(minuend_dtype);
                    return;
                }
            }
        }
    } 
    
    else {
        NNL2_ERROR("Cannot broadcast subtrahend tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for subtraction with broadcasting (in place)
 * @details
 * Array follows the common backend registration pattern for subtraction
 * with broadcasting operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for subtraction with broadcasting
 * 
 * @see nnl2_naive
 * @see naive_sub_broadcasting_inplace
 */
Implementation sub_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_sub_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for subtraction with broadcasting operation (in place)
 * @ingroup backend_system
 */
subbroadcastinginplacefn sub_broadcasting_inplace;

/**
 * @brief Sets the backend for subtraction with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for subtraction with broadcasting
 * @see SET_BACKEND_BY_NAME
 * @see sub_broadcasting_inplace_backends
 */
void set_sub_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_broadcasting_inplace_backends, sub_broadcasting_inplace, backend_name);
}

/** @brief
 * Performs element-wise subtraction with broadcasting support
 *
 ** @param minuend
 * First tensor to subtract from
 *
 ** @param subtrahend
 * Second tensor to subtract
 * 
 ** @return
 * New tensor containing the result of subtraction
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_sub_broadcasting(Tensor* minuend, Tensor* subtrahend) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks for max safety level 
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX	
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend, "Minuend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend, "Subtrahend tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(minuend->shape, "Minuend shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(subtrahend->shape, "Subtrahend shape is NULL", NULL);
	#endif
 
	// Calculate the total number of elements in each tensor
	size_t numel_minuend = product(minuend->shape, minuend->rank);
	size_t numel_subtrahend = product(subtrahend->shape, subtrahend->rank);
	
	// Getting the tensor data types
	TensorType minuend_dtype = minuend->dtype;
	TensorType subtrahend_dtype = subtrahend->dtype;
	
	TensorType winner_in_the_type_hierarchy = MAX(minuend_dtype, subtrahend_dtype);
	
	// Сreating a resultant tensor
	Tensor* result = nnl2_empty(minuend->shape, minuend->rank, winner_in_the_type_hierarchy);
	
	// Checking the possibility of broadcasting (numel_minuend must be a multiple of numel_subtrahend)
	if((numel_minuend % numel_subtrahend) == 0) {
		if(minuend_dtype == subtrahend_dtype) {
			// Handling the case where the data types match (more efficiently)
			switch(minuend_dtype) {
				case FLOAT64: {
					double* cast_minuend_data = (double*)minuend->data;
					double* cast_subtrahend_data = (double*)subtrahend->data;
					double* cast_result_data = (double*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_minuend_data = (float*)minuend->data;
					float* cast_subtrahend_data = (float*)subtrahend->data;
					float* cast_result_data = (float*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_minuend_data = (int32_t*)minuend->data;
					int32_t* cast_subtrahend_data = (int32_t*)subtrahend->data;
					int32_t* cast_result_data = (int32_t*)result->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							cast_result_data[i * numel_subtrahend + j] = cast_minuend_data[i * numel_subtrahend + j] - cast_subtrahend_data[j];
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(minuend_dtype);
					return NULL;
				}
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		} 
		
		else {
			// Handling a case with different data types (conversion required)
			size_t minuend_step = get_dtype_size(minuend_dtype);
			size_t subtrahend_step = get_dtype_size(subtrahend_dtype);
			
			switch(winner_in_the_type_hierarchy) {
				case FLOAT64: {
					double* cast_data_result = (double*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							// Get a pointer to minuend element, subtrahend element and convert its type
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend = cast_subtrahend_data + j * subtrahend_step; 
							
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_float64(elem_minuend, minuend_dtype) - nnl2_convert_to_float64(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				case FLOAT32: {
					float* cast_data_result = (float*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {
						for(size_t j = 0; j < numel_subtrahend; j++) {
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend =  cast_subtrahend_data + j * subtrahend_step;
							
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_float32(elem_minuend, minuend_dtype) - nnl2_convert_to_float32(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				case INT32: {
					int32_t* cast_data_result = (int32_t*)result->data;
					
					char* cast_minuend_data = (char*)minuend->data;
					char* cast_subtrahend_data =  (char*)subtrahend->data;
					
					for(size_t i = 0; i < (numel_minuend / numel_subtrahend); i++) {					
						for(size_t j = 0; j < numel_subtrahend; j++) {
							void* elem_minuend = cast_minuend_data + (i * numel_subtrahend + j) * minuend_step;
							void* elem_subtrahend =  cast_subtrahend_data + j * subtrahend_step;
						
							cast_data_result[i * numel_subtrahend + j] = nnl2_convert_to_int32(elem_minuend, minuend_dtype) - nnl2_convert_to_int32(elem_subtrahend, subtrahend_dtype);
						}
					}
					
					break;
				}
				
				default: {
					NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
					nnl2_free_tensor(result);
					return NULL;
				}
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		}
	} 
	
	else {
		NNL2_ERROR("Cannot broadcast subtrahend tensor");
		return NULL;
	}
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for subtraction with broadcasting
 */
Implementation sub_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_sub_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for subtraction with broadcasting operation
 * @ingroup backend_system
 */
subbroadcastingfn sub_broadcasting;

/**
 * @brief Sets the backend for subtraction with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for subtraction with broadcasting
 */
void set_sub_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_broadcasting_backends, sub_broadcasting, backend_name);
}

/** @brief
 * Performs element-wise multiplication with broadcasting (in place)
 *
 ** @param multiplicand
 * Pointer to multiplicand tensor (will be modified in place)
 *
 ** @param multiplier
 * Pointer to multiplier tensor
 */
void naive_mul_broadcasting_inplace(Tensor* multiplicand, const Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "Multiplicand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->shape, "Multiplicand shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->shape, "Multiplier shape is NULL");
    #endif
    
    size_t numel_multiplicand = product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = product(multiplier->shape, multiplier->rank);
    
    // Getting the tensor data types
    TensorType multiplicand_dtype = multiplicand->dtype;
    TensorType multiplier_dtype = multiplier->dtype;

    if((numel_multiplicand % numel_multiplier) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(multiplicand_dtype == multiplier_dtype) {
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* cast_multiplicand_data = (double*)multiplicand->data;
                    double* cast_multiplier_data = (double*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_multiplicand_data = (float*)multiplicand->data;
                    float* cast_multiplier_data = (float*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_multiplicand_data = (int32_t*)multiplicand->data;
                    int32_t* cast_multiplier_data = (int32_t*)multiplier->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_multiplicand_data[i * numel_multiplier + j] *= cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t multiplier_step = get_dtype_size(multiplier_dtype);
            char* multiplier_data = (char*)multiplier->data;
            
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* data_multiplicand = (double*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_float64(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_multiplicand = (float*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_float32(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* multiplier_elem = multiplier_data + j * multiplier_step;
                            data_multiplicand[i * numel_multiplier + j] *= nnl2_convert_to_int32(multiplier_elem, multiplier_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast multiplier tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for multiplication with broadcasting (in place)
 */
Implementation mul_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_mul_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for multiplication with broadcasting operation (in place)
 * @ingroup backend_system
 */
mulbroadcastinginplacefn mul_broadcasting_inplace;

/**
 * @brief Sets the backend for multiplication with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for multiplication with broadcasting
 */
void set_mul_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_broadcasting_inplace_backends, mul_broadcasting_inplace, backend_name);
}

/** @brief
 * Performs element-wise multiplication with broadcasting support
 *
 ** @param multiplicand
 * First tensor to multiply
 *
 ** @param multiplier
 * Second tensor to multiply
 * 
 ** @return
 * New tensor containing the result of multiplication
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_mul_broadcasting(Tensor* multiplicand, Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand, "Multiplicand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier, "Multiplier tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand->shape, "Multiplicand shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier->shape, "Multiplier shape is NULL", NULL);
    #endif
 
    size_t numel_multiplicand = product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = product(multiplier->shape, multiplier->rank);
    
    // Getting the tensor data types
    TensorType multiplicand_dtype = multiplicand->dtype;
    TensorType multiplier_dtype = multiplier->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(multiplicand_dtype, multiplier_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(multiplicand->shape, multiplicand->rank, winner_in_the_type_hierarchy);

    if((numel_multiplicand % numel_multiplier) == 0) {
        if(multiplicand_dtype == multiplier_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(multiplicand_dtype) {
                case FLOAT64: {
                    double* cast_multiplicand_data = (double*)multiplicand->data;
                    double* cast_multiplier_data = (double*)multiplier->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_multiplicand_data = (float*)multiplicand->data;
                    float* cast_multiplier_data = (float*)multiplier->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_multiplicand_data = (int32_t*)multiplicand->data;
                    int32_t* cast_multiplier_data = (int32_t*)multiplier->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            cast_result_data[i * numel_multiplier + j] = cast_multiplicand_data[i * numel_multiplier + j] * cast_multiplier_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(multiplicand_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t multiplicand_step = get_dtype_size(multiplicand_dtype);
            size_t multiplier_step = get_dtype_size(multiplier_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step; 
                            
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_float64(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_float64(elem_multiplier, multiplier_dtype);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step;
                            
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_float32(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_float32(elem_multiplier, multiplier_dtype);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_multiplicand_data = (char*)multiplicand->data;
                    char* cast_multiplier_data =  (char*)multiplier->data;
                    
                    for(size_t i = 0; i < (numel_multiplicand / numel_multiplier); i++) {                    
                        for(size_t j = 0; j < numel_multiplier; j++) {
                            void* elem_multiplicand = cast_multiplicand_data + (i * numel_multiplier + j) * multiplicand_step;
                            void* elem_multiplier = cast_multiplier_data + j * multiplier_step;
                        
                            cast_data_result[i * numel_multiplier + j] = nnl2_convert_to_int32(elem_multiplicand, multiplicand_dtype) * nnl2_convert_to_int32(elem_multiplier, multiplier_dtype);
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast multiplier tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for multiplication with broadcasting
 */
Implementation mul_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_mul_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for multiplication with broadcasting operation
 * @ingroup backend_system
 */
mulbroadcastingfn mul_broadcasting;

/**
 * @brief Sets the backend for multiplication with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for multiplication with broadcasting
 */
void set_mul_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_broadcasting_backends, mul_broadcasting, backend_name);
}

/** @brief
 * Performs element-wise division with broadcasting (in place)
 *
 ** @param dividend
 * Pointer to dividend tensor (will be modified in place)
 *
 ** @param divisor
 * Pointer to divisor tensor
 */
void naive_div_broadcasting_inplace(Tensor* dividend, const Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "Dividend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->shape, "Dividend shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->shape, "Divisor shape is NULL");
    #endif
    
    size_t numel_dividend = product(dividend->shape, dividend->rank);
    size_t numel_divisor = product(divisor->shape, divisor->rank);
    
    // Getting the tensor data types
    TensorType dividend_dtype = dividend->dtype;
    TensorType divisor_dtype = divisor->dtype;

    if((numel_dividend % numel_divisor) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(dividend_dtype == divisor_dtype) {
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* cast_dividend_data = (double*)dividend->data;
                    double* cast_divisor_data = (double*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_dividend_data = (float*)dividend->data;
                    float* cast_divisor_data = (float*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_dividend_data = (int32_t*)dividend->data;
                    int32_t* cast_divisor_data = (int32_t*)divisor->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_dividend_data[i * numel_divisor + j] /= cast_divisor_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t divisor_step = get_dtype_size(divisor_dtype);
            char* divisor_data = (char*)divisor->data;
            
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* data_dividend = (double*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_float64(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_dividend = (float*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_float32(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_dividend = (int32_t*)dividend->data;
                
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* divisor_elem = divisor_data + j * divisor_step;
                            data_dividend[i * numel_divisor + j] /= nnl2_convert_to_int32(divisor_elem, divisor_dtype);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast divisor tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for division with broadcasting (in place)
 */
Implementation div_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_div_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for division with broadcasting operation (in place)
 * @ingroup backend_system
 */
divbroadcastinginplacefn div_broadcasting_inplace;

/**
 * @brief Sets the backend for division with broadcasting operation (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for division with broadcasting
 */
void set_div_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_broadcasting_inplace_backends, div_broadcasting_inplace, backend_name);
}

/** @brief
 * Performs element-wise division with broadcasting support
 *
 ** @param dividend
 * First tensor to divide
 *
 ** @param divisor
 * Second tensor to divide by
 * 
 ** @return
 * New tensor containing the result of division
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_div_broadcasting(Tensor* dividend, Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend, "Dividend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor, "Divisor tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dividend->shape, "Dividend shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(divisor->shape, "Divisor shape is NULL", NULL);
    #endif
 
    size_t numel_dividend = product(dividend->shape, dividend->rank);
    size_t numel_divisor = product(divisor->shape, divisor->rank);
    
    // Getting the tensor data types
    TensorType dividend_dtype = dividend->dtype;
    TensorType divisor_dtype = divisor->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(dividend_dtype, divisor_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(dividend->shape, dividend->rank, winner_in_the_type_hierarchy);

    if((numel_dividend % numel_divisor) == 0) {
        if(dividend_dtype == divisor_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(dividend_dtype) {
                case FLOAT64: {
                    double* cast_dividend_data = (double*)dividend->data;
                    double* cast_divisor_data = (double*)divisor->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_dividend_data = (float*)dividend->data;
                    float* cast_divisor_data = (float*)divisor->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_dividend_data = (int32_t*)dividend->data;
                    int32_t* cast_divisor_data = (int32_t*)divisor->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            cast_result_data[i * numel_divisor + j] = cast_dividend_data[i * numel_divisor + j] / cast_divisor_data[j];
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(dividend_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t dividend_step = get_dtype_size(dividend_dtype);
            size_t divisor_step = get_dtype_size(divisor_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step; 
                            
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_float64(elem_dividend, dividend_dtype) / nnl2_convert_to_float64(elem_divisor, divisor_dtype);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step;
                            
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_float32(elem_dividend, dividend_dtype) / nnl2_convert_to_float32(elem_divisor, divisor_dtype);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_dividend_data = (char*)dividend->data;
                    char* cast_divisor_data =  (char*)divisor->data;
                    
                    for(size_t i = 0; i < (numel_dividend / numel_divisor); i++) {                    
                        for(size_t j = 0; j < numel_divisor; j++) {
                            void* elem_dividend = cast_dividend_data + (i * numel_divisor + j) * dividend_step;
                            void* elem_divisor = cast_divisor_data + j * divisor_step;
                        
                            cast_data_result[i * numel_divisor + j] = nnl2_convert_to_int32(elem_dividend, dividend_dtype) / nnl2_convert_to_int32(elem_divisor, divisor_dtype);
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast divisor tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for division with broadcasting
 */
Implementation div_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_div_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for division with broadcasting operation
 * @ingroup backend_system
 */
divbroadcastingfn div_broadcasting;

/**
 * @brief Sets the backend for division with broadcasting operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for division with broadcasting
 */
void set_div_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_broadcasting_backends, div_broadcasting, backend_name);
}

/** @brief
 * Performs element-wise power operation with broadcasting (in place)
 *
 ** @param base
 * Pointer to base tensor (will be modified in place)
 *
 ** @param exponent
 * Pointer to exponent tensor
 */
void naive_pow_broadcasting_inplace(Tensor* base, const Tensor* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(base, "Base tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(exponent, "Exponent tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(base->shape, "Base shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(exponent->shape, "Exponent shape is NULL");
    #endif
    
    size_t numel_base = product(base->shape, base->rank);
    size_t numel_exponent = product(exponent->shape, exponent->rank);
    
    // Getting the tensor data types
    TensorType base_dtype = base->dtype;
    TensorType exponent_dtype = exponent->dtype;

    if((numel_base % numel_exponent) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(base_dtype == exponent_dtype) {
            switch(base_dtype) {
                case FLOAT64: {
                    double* cast_base_data = (double*)base->data;
                    double* cast_exponent_data = (double*)exponent->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_base_data[i * numel_exponent + j] = pow(cast_base_data[i * numel_exponent + j], cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_base_data = (float*)base->data;
                    float* cast_exponent_data = (float*)exponent->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_base_data[i * numel_exponent + j] = powf(cast_base_data[i * numel_exponent + j], cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_base_data = (int32_t*)base->data;
                    int32_t* cast_exponent_data = (int32_t*)exponent->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_base_data[i * numel_exponent + j] = (int32_t)pow((double)cast_base_data[i * numel_exponent + j], (double)cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(base_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t exponent_step = get_dtype_size(exponent_dtype);
            char* exponent_data = (char*)exponent->data;
            
            switch(base_dtype) {
                case FLOAT64: {
                    double* data_base = (double*)base->data;
                
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* exponent_elem = exponent_data + j * exponent_step;
                            data_base[i * numel_exponent + j] = pow(data_base[i * numel_exponent + j], nnl2_convert_to_float64(exponent_elem, exponent_dtype));
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_base = (float*)base->data;
                
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* exponent_elem = exponent_data + j * exponent_step;
                            data_base[i * numel_exponent + j] = powf(data_base[i * numel_exponent + j], nnl2_convert_to_float32(exponent_elem, exponent_dtype));
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_base = (int32_t*)base->data;
                
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* exponent_elem = exponent_data + j * exponent_step;
                            data_base[i * numel_exponent + j] = (int32_t)pow((double)data_base[i * numel_exponent + j], (double)nnl2_convert_to_int32(exponent_elem, exponent_dtype));
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(base_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast exponent tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for power operation with broadcasting (in place)
 */
Implementation pow_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_pow_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for power operation with broadcasting (in place)
 * @ingroup backend_system
 */
powbroadcastinginplacefn pow_broadcasting_inplace;

/**
 * @brief Sets the backend for power operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for power operation with broadcasting
 */
void set_pow_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_broadcasting_inplace_backends, pow_broadcasting_inplace, backend_name);
}

/** @brief
 * Performs element-wise power operation with broadcasting support
 *
 ** @param base
 * Base tensor
 *
 ** @param exponent
 * Exponent tensor
 * 
 ** @return
 * New tensor containing the result of power operation
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_pow_broadcasting(Tensor* base, Tensor* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base, "Base tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(exponent, "Exponent tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base->shape, "Base shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(exponent->shape, "Exponent shape is NULL", NULL);
    #endif
 
    size_t numel_base = product(base->shape, base->rank);
    size_t numel_exponent = product(exponent->shape, exponent->rank);
    
    // Getting the tensor data types
    TensorType base_dtype = base->dtype;
    TensorType exponent_dtype = exponent->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(base_dtype, exponent_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(base->shape, base->rank, winner_in_the_type_hierarchy);

    if((numel_base % numel_exponent) == 0) {
        if(base_dtype == exponent_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(base_dtype) {
                case FLOAT64: {
                    double* cast_base_data = (double*)base->data;
                    double* cast_exponent_data = (double*)exponent->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_result_data[i * numel_exponent + j] = pow(cast_base_data[i * numel_exponent + j], cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_base_data = (float*)base->data;
                    float* cast_exponent_data = (float*)exponent->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_result_data[i * numel_exponent + j] = powf(cast_base_data[i * numel_exponent + j], cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_base_data = (int32_t*)base->data;
                    int32_t* cast_exponent_data = (int32_t*)exponent->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            cast_result_data[i * numel_exponent + j] = (int32_t)pow((double)cast_base_data[i * numel_exponent + j], (double)cast_exponent_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(base_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t base_step = get_dtype_size(base_dtype);
            size_t exponent_step = get_dtype_size(exponent_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_base_data = (char*)base->data;
                    char* cast_exponent_data =  (char*)exponent->data;
                    
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* elem_base = cast_base_data + (i * numel_exponent + j) * base_step;
                            void* elem_exponent = cast_exponent_data + j * exponent_step; 
                            
                            cast_data_result[i * numel_exponent + j] = pow(nnl2_convert_to_float64(elem_base, base_dtype), nnl2_convert_to_float64(elem_exponent, exponent_dtype));
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_base_data = (char*)base->data;
                    char* cast_exponent_data =  (char*)exponent->data;
                    
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* elem_base = cast_base_data + (i * numel_exponent + j) * base_step;
                            void* elem_exponent = cast_exponent_data + j * exponent_step;
                            
                            cast_data_result[i * numel_exponent + j] = powf(nnl2_convert_to_float32(elem_base, base_dtype), nnl2_convert_to_float32(elem_exponent, exponent_dtype));
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_base_data = (char*)base->data;
                    char* cast_exponent_data =  (char*)exponent->data;
                    
                    for(size_t i = 0; i < (numel_base / numel_exponent); i++) {                    
                        for(size_t j = 0; j < numel_exponent; j++) {
                            void* elem_base = cast_base_data + (i * numel_exponent + j) * base_step;
                            void* elem_exponent = cast_exponent_data + j * exponent_step;
                        
                            cast_data_result[i * numel_exponent + j] = (int32_t)pow((double)nnl2_convert_to_int32(elem_base, base_dtype), (double)nnl2_convert_to_int32(elem_exponent, exponent_dtype));
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast exponent tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for power operation with broadcasting
 */
Implementation pow_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_pow_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for power operation with broadcasting
 * @ingroup backend_system
 */
powbroadcastingfn pow_broadcasting;

/**
 * @brief Sets the backend for power operation with broadcasting
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for power operation with broadcasting
 */
void set_pow_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_broadcasting_backends, pow_broadcasting, backend_name);
}

/** @brief
 * Performs element-wise maximum with broadcasting (in place)
 *
 ** @param x
 * Pointer to first tensor (will be modified in place)
 *
 ** @param y
 * Pointer to second tensor
 */
void naive_max_broadcasting_inplace(Tensor* x, const Tensor* y) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(x, "X tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(y, "Y tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(x->shape, "X shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(y->shape, "Y shape is NULL");
    #endif
    
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);
    
    // Getting the tensor data types
    TensorType x_dtype = x->dtype;
    TensorType y_dtype = y->dtype;

    if((numel_x % numel_y) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(x_dtype == y_dtype) {
            switch(x_dtype) {
                case FLOAT64: {
                    double* cast_x_data = (double*)x->data;
                    double* cast_y_data = (double*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_x_data = (float*)x->data;
                    float* cast_y_data = (float*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_x_data = (int32_t*)x->data;
                    int32_t* cast_y_data = (int32_t*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t y_step = get_dtype_size(y_dtype);
            char* y_data = (char*)y->data;
            
            switch(x_dtype) {
                case FLOAT64: {
                    double* data_x = (double*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            double y_val = nnl2_convert_to_float64(y_elem, y_dtype);
                            data_x[i * numel_y + j] = MAX(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_x = (float*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            float y_val = nnl2_convert_to_float32(y_elem, y_dtype);
                            data_x[i * numel_y + j] = MAX(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_x = (int32_t*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            int32_t y_val = nnl2_convert_to_int32(y_elem, y_dtype);
                            data_x[i * numel_y + j] = MAX(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast y tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for maximum operation with broadcasting (in place)
 */
Implementation max_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_max_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for maximum operation with broadcasting (in place)
 * @ingroup backend_system
 */
maxbroadcastinginplacefn max_broadcasting_inplace;

/**
 * @brief Sets the backend for maximum operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for maximum operation with broadcasting
 */
void set_max_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_broadcasting_inplace_backends, max_broadcasting_inplace, backend_name);
}

/** @brief
 * Performs element-wise minimum with broadcasting (in place)
 *
 ** @param x
 * Pointer to first tensor (will be modified in place)
 *
 ** @param y
 * Pointer to second tensor
 */
void naive_min_broadcasting_inplace(Tensor* x, Tensor* y) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(x, "X tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(y, "Y tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(x->shape, "X shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(y->shape, "Y shape is NULL");
    #endif
    
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);
    
    // Getting the tensor data types
    TensorType x_dtype = x->dtype;
    TensorType y_dtype = y->dtype;

    if((numel_x % numel_y) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(x_dtype == y_dtype) {
            switch(x_dtype) {
                case FLOAT64: {
                    double* cast_x_data = (double*)x->data;
                    double* cast_y_data = (double*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_x_data = (float*)x->data;
                    float* cast_y_data = (float*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_x_data = (int32_t*)x->data;
                    int32_t* cast_y_data = (int32_t*)y->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_x_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t y_step = get_dtype_size(y_dtype);
            char* y_data = (char*)y->data;
            
            switch(x_dtype) {
                case FLOAT64: {
                    double* data_x = (double*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            double y_val = nnl2_convert_to_float64(y_elem, y_dtype); 
                            data_x[i * numel_y + j] = MIN(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_x = (float*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            float y_val = nnl2_convert_to_float32(y_elem, y_dtype);
                            data_x[i * numel_y + j] = MIN(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_x = (int32_t*)x->data;
                
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* y_elem = y_data + j * y_step;
                            int32_t y_val = nnl2_convert_to_int32(y_elem, y_dtype);
                            data_x[i * numel_y + j] = MIN(data_x[i * numel_y + j], y_val);
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast y tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for minimum operation with broadcasting (in place)
 */
Implementation min_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_min_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for minimum operation with broadcasting (in place)
 * @ingroup backend_system
 */
minbroadcastinginplacefn min_broadcasting_inplace;

/**
 * @brief Sets the backend for minimum operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for minimum operation with broadcasting
 */
void set_min_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_broadcasting_inplace_backends, min_broadcasting_inplace, backend_name);
}

/** @brief
 * Performs element-wise maximum with broadcasting support
 *
 ** @param x
 * First tensor
 *
 ** @param y
 * Second tensor
 * 
 ** @return
 * New tensor containing the result of maximum operation
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_max_broadcasting(Tensor* x, Tensor* y) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "X tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y, "Y tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x->shape, "X shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y->shape, "Y shape is NULL", NULL);
    #endif
 
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);
    
    // Getting the tensor data types
    TensorType x_dtype = x->dtype;
    TensorType y_dtype = y->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(x_dtype, y_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(x->shape, x->rank, winner_in_the_type_hierarchy);

    if((numel_x % numel_y) == 0) {
        if(x_dtype == y_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(x_dtype) {
                case FLOAT64: {
                    double* cast_x_data = (double*)x->data;
                    double* cast_y_data = (double*)y->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_x_data = (float*)x->data;
                    float* cast_y_data = (float*)y->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_x_data = (int32_t*)x->data;
                    int32_t* cast_y_data = (int32_t*)y->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MAX(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t x_step = get_dtype_size(x_dtype);
            size_t y_step = get_dtype_size(y_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step; 
                            
                            double x_val = nnl2_convert_to_float64(elem_x, x_dtype);
                            double y_val = nnl2_convert_to_float64(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MAX(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step;
                            
                            float x_val = nnl2_convert_to_float32(elem_x, x_dtype);
                            float y_val = nnl2_convert_to_float32(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MAX(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {                    
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step;
                        
                            int32_t x_val = nnl2_convert_to_int32(elem_x, x_dtype);
                            int32_t y_val = nnl2_convert_to_int32(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MAX(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast y tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for maximum operation with broadcasting
 */
Implementation max_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_max_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for maximum operation with broadcasting
 * @ingroup backend_system
 */
maxbroadcastingfn max_broadcasting;

/**
 * @brief Sets the backend for maximum operation with broadcasting
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for maximum operation with broadcasting
 */
void set_max_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_broadcasting_backends, max_broadcasting, backend_name);
}

/** @brief
 * Performs element-wise minimum with broadcasting support
 *
 ** @param x
 * First tensor
 *
 ** @param y
 * Second tensor
 * 
 ** @return
 * New tensor containing the result of minimum operation
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_min_broadcasting(Tensor* x, Tensor* y) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "X tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y, "Y tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x->shape, "X shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(y->shape, "Y shape is NULL", NULL);
    #endif
 
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);
    
    // Getting the tensor data types
    TensorType x_dtype = x->dtype;
    TensorType y_dtype = y->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(x_dtype, y_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(x->shape, x->rank, winner_in_the_type_hierarchy);

    if((numel_x % numel_y) == 0) {
        if(x_dtype == y_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(x_dtype) {
                case FLOAT64: {
                    double* cast_x_data = (double*)x->data;
                    double* cast_y_data = (double*)y->data;
                    double* cast_result_data = (double*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_x_data = (float*)x->data;
                    float* cast_y_data = (float*)y->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_x_data = (int32_t*)x->data;
                    int32_t* cast_y_data = (int32_t*)y->data;
                    int32_t* cast_result_data = (int32_t*)result->data;

                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            cast_result_data[i * numel_y + j] = MIN(cast_x_data[i * numel_y + j], cast_y_data[j]);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(x_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t x_step = get_dtype_size(x_dtype);
            size_t y_step = get_dtype_size(y_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step; 
                            
                            double x_val = nnl2_convert_to_float64(elem_x, x_dtype);
                            double y_val = nnl2_convert_to_float64(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MIN(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step;
                            
                            float x_val = nnl2_convert_to_float32(elem_x, x_dtype);
                            float y_val = nnl2_convert_to_float32(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MIN(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    
                    char* cast_x_data = (char*)x->data;
                    char* cast_y_data =  (char*)y->data;
                    
                    for(size_t i = 0; i < (numel_x / numel_y); i++) {                    
                        for(size_t j = 0; j < numel_y; j++) {
                            void* elem_x = cast_x_data + (i * numel_y + j) * x_step;
                            void* elem_y = cast_y_data + j * y_step;
                        
                            int32_t x_val = nnl2_convert_to_int32(elem_x, x_dtype);
                            int32_t y_val = nnl2_convert_to_int32(elem_y, y_dtype);
                            cast_data_result[i * numel_y + j] = MIN(x_val, y_val);
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast y tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for minimum operation with broadcasting
 */
Implementation min_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_min_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for minimum operation with broadcasting
 * @ingroup backend_system
 */
minbroadcastingfn min_broadcasting;

/**
 * @brief Sets the backend for minimum operation with broadcasting
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for minimum operation with broadcasting
 */
void set_min_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_broadcasting_backends, min_broadcasting, backend_name);
}

/** @brief 
 * Fills tensor with data from provided array 
 * 
 ** @param tensor
 * Tensor to fill with data
 *
 ** @param data 
 * Pointer to data array
 *
 ** @param num_elems 
 * Number of elements to copy
 */
inline static void naive_fill_tensor_with_data(Tensor* tensor, void* data, size_t num_elems) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "Passed tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(data, "Passed data pointer is NULL");
	#endif
	
	if(num_elems == 0) return;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* tensor_data = (double*)tensor->data;
			double* cast_data = (double*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];			
			break;
		}
		
		case FLOAT32: {
			float* tensor_data = (float*)tensor->data;
			float* cast_data = (float*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];			
			break;
		}
		
		case INT32: {
			int32_t* tensor_data = (int32_t*)tensor->data;
			int32_t* cast_data = (int32_t*)data;
			for(size_t it = 0; it < num_elems; it++) tensor_data[it] = cast_data[it];			
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for filling tensor with data
 */
Implementation fill_tensor_with_data_backends[] = {
    REGISTER_BACKEND(naive_fill_tensor_with_data, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for filling tensor with data operation
 * @ingroup backend_system
 */
filltensorwithdatafn fill_tensor_with_data;

/**
 * @brief Sets the backend for filling tensor with data operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for filling tensor with data
 */
void set_fill_tensor_with_data_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(fill_tensor_with_data_backends, fill_tensor_with_data, backend_name);
}

/** @brief
 * Creates a tensor from a flattened array
 *
 ** @param arr
 * Pointer to the flattened array data
 *
 ** @param num_elems_arr
 * Number of elements in the array
 *
 ** @param shape
 * Shape of the resulting tensor
 *
 ** @param rank
 * Rank of the resulting tensor
 *
 ** @param dtype
 * Data type of the resulting tensor
 *
 ** @return
 * New tensor with data copied from the array
 */
Tensor* make_tensor_from_flatten(void* arr, size_t num_elems_arr, int* shape, int rank, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t num_elems_tensor = product(shape, rank);
	
	if(num_elems_tensor != num_elems_arr) {
		NNL2_ERROR("The number of elements in the specified array does not match the specified shapes");
		return NULL;
	}
	
	Tensor* result = nnl2_empty(shape, rank, dtype);
	fill_tensor_with_data(result, arr, num_elems_tensor);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/** @brief 
 * Performs element-wise AXPY operation (naive implementation)
 * 
 * Computes: summand = summand + alpha * sumend
 * Performs the scaled vector addition operation on two tensors,
 * modifying the summand tensor in place
 *
 ** @param summand 
 * Pointer to the tensor that will be modified (receives the AXPY result)
 *
 ** @param sumend 
 * Pointer to the tensor whose values will be scaled and added to the summand
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor values
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The sumend elements are converted to the summand's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the summand tensor directly
 * Both tensors must have the same shape
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Compute a = a + 2.5 * b
 * naive_axpy_inplace(a, b, 2.5f);
 * 
 * // Now a contains 3.5 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_axpy_inplace(Tensor* summand, Tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "Sumend tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the summand tensor
    size_t total_elems = product(summand->shape, summand->rank);
    
    // If the tensor is empty, exit the function
    if(total_elems == 0) return;
    
    TensorType dtype_summand = summand->dtype;
    TensorType dtype_sumend = sumend->dtype;
    
    if(dtype_summand == dtype_sumend) {
        // Handling case when the tensors have the same type
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                volatile double* data_sumend = (double*)sumend->data;
                double alpha_double = (double)alpha;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha_double;
                }
				
                break;
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                volatile float* data_sumend = (float*)sumend->data;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha;
                }    
				
                break;
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                volatile int32_t* data_sumend = (int32_t*)sumend->data;
                int32_t alpha_int = (int32_t)alpha;
                
                // Element-wise AXPY operation
                for(size_t i = 0; i < total_elems; i++) {
                    data_summand[i] += data_sumend[i] * alpha_int;
                }        
				
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing sumend tensor elements
        size_t sumend_step = get_dtype_size(dtype_sumend);
        
        // Casting sumend data to char* for byte access
        char* sumend_data = (char*)sumend->data;
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                double alpha_double = (double)alpha;
                
                // For each element, convert the sumend element to FLOAT64 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_float64(sumend_elem, dtype_sumend) * alpha_double;
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                
                // For each element, convert the sumend element to FLOAT32 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_float32(sumend_elem, dtype_sumend) * alpha;
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                int32_t alpha_int = (int32_t)alpha;
                
                // For each element, convert the sumend element to INT32 and perform AXPY
                for(size_t i = 0; i < total_elems; i++) {
                    void* sumend_elem = sumend_data + i * sumend_step;
                    data_summand[i] += nnl2_convert_to_int32(sumend_elem, dtype_sumend) * alpha_int;
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPY in-place operation
 */
Implementation axpy_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpy_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPY in-place operation
 * @ingroup backend_system
 */
axpyinplacefn axpy_inplace;
make_current_backend(axpy_inplace);

/**
 * @brief Sets the backend for AXPY in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY in-place operation
 */
void set_axpy_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(axpy_inplace_backends, axpy_inplace, backend_name, current_backend(axpy_inplace));
}

/**
 * @brief Gets the name of the current backend for AXPY in-place operation
 * @ingroup backend_system
 * @return const char* Name of the current backend
 */
const char* get_axpy_inplace_backend() {
    return current_backend(axpy_inplace);
}

/**
 * @brief Gets the list of available backends for AXPY in-place operation
 * @ingroup backend_system
 * @return const char** Array of backend names
 */
DEFINE_GET_BACKENDS_FUNCTION(axpy_inplace);

/**
 * @brief Gets the number of available backends for AXPY in-place operation
 * @ingroup backend_system
 * @return size_t Number of available backends
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(axpy_inplace);

/** @brief
 * Performs element-wise AXPY operation (naive implementation)
 * Computes: result = summand + alpha * sumend
 *
 ** @details
 * The function creates a new tensor containing the result of the AXPY operation
 * on the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param summand
 * Pointer to the summand tensor
 *
 ** @param sumend
 * Pointer to the sumend tensor to be scaled and added
 *
 ** @param alpha
 * Scalar multiplier for the sumend tensor
 *
 ** @return 
 * Pointer to a new tensor with the AXPY operation result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_axpy(Tensor* summand, Tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->data, "Summand tensor's data is NULL", NULL);
        
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data, "Sumend tensor's data is NULL", NULL);
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(summand->shape, summand->rank);
    
    TensorType dtype_summand = summand->dtype;
    TensorType dtype_sumend = sumend->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_summand, dtype_sumend);

    // Create an output tensor with the same shape and winning data type
    Tensor* result = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
    
    if(len == 0) return result;
    
    if(dtype_summand == dtype_sumend) {
        // Handling the case if the data types match
        
        switch(dtype_summand) {
            case FLOAT64: {
                volatile double* data_summand = (double*)summand->data;
                volatile double* data_sumend = (double*)sumend->data;
                volatile double* data_result = (double*)result->data;
                double alpha_double = (double)alpha;
            
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha_double);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_summand = (float*)summand->data;
                volatile float* data_sumend = (float*)sumend->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_summand = (int32_t*)summand->data;
                volatile int32_t* data_sumend = (int32_t*)sumend->data;
                volatile int32_t* data_result = (int32_t*)result->data;
                int32_t alpha_int = (int32_t)alpha;
        
                // Element-wise AXPY operation
                for(size_t i = 0; i < len; i++) {
                    data_result[i] = data_summand[i] + (data_sumend[i] * alpha_int);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_summand);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch(winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_result = (double*)result->data;
                double alpha_double = (double)alpha;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_float64(elem_summand, dtype_summand) + (nnl2_convert_to_float64(elem_sumend, dtype_sumend) * alpha_double);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_float32(elem_summand, dtype_summand) + (nnl2_convert_to_float32(elem_sumend, dtype_sumend) * alpha);
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                int32_t alpha_int = (int32_t)alpha;
                
                for(size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_sumend = (char*)sumend->data + i * get_dtype_size(dtype_sumend);
                    
                    data_result[i] = nnl2_convert_to_int32(elem_summand, dtype_summand) + 
                                    (nnl2_convert_to_int32(elem_sumend, dtype_sumend) * alpha_int);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}
/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPY operation
 */
Implementation axpy_backends[] = {
    REGISTER_BACKEND(naive_axpy, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPY operation
 * @ingroup backend_system
 */
axpyfn axpy;
make_current_backend(axpy);

/**
 * @brief Sets the backend for AXPY operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY operation
 */
void set_axpy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(axpy_backends, axpy, backend_name, current_backend(axpy));
}

/**
 * @brief Gets the name of the current backend for AXPY operation
 * @ingroup backend_system
 * @return const char* Name of the current backend
 */
const char* get_axpy_backend() {
    return current_backend(axpy);
}

/**
 * @brief Gets the list of available backends for AXPY operation
 * @ingroup backend_system
 * @return const char** Array of backend names
 */
DEFINE_GET_BACKENDS_FUNCTION(axpy);

/**
 * @brief Gets the number of available backends for AXPY operation
 * @ingroup backend_system
 * @return size_t Number of available backends
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(axpy);

/** @brief 
 * Performs element-wise AXPF operation (scalar AXPY) in place
 * Computes: summand = summand + alpha * sumend (where sumend is a scalar)
 * 
 ** @param summand 
 * Pointer to the tensor that will be modified in place
 * 
 ** @param sumend 
 * Pointer to the scalar value to be scaled and added
 * 
 ** @param alpha
 * Scalar multiplier for the sumend value
 */
void naive_axpf_inplace(Tensor* summand, void* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(summand->shape, summand->rank);
    if(total_elems == 0) return; 
    
    switch(summand->dtype) {
        case FLOAT64: {
            double* cast_summand = (double*)summand->data;
            double cast_sumend = *((double*)sumend);
            double alpha_double = (double)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha_double;
            break;
        }
        
        case FLOAT32: {
            float* cast_summand = (float*)summand->data;
            float cast_sumend = *((float*)sumend);
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha;
            break;
        }
        
        case INT32: {
            int32_t* cast_summand = (int32_t*)summand->data;
            int32_t cast_sumend = *((int32_t*)sumend);
            int32_t alpha_int = (int32_t)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha_int;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(summand->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPF in-place operation
 */
Implementation axpf_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPF in-place operation
 * @ingroup backend_system
 */
axpfinplacefn axpf_inplace;

/**
 * @brief Sets the backend for AXPF in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPF in-place operation
 */
void set_axpf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpf_inplace_backends, axpf_inplace, backend_name);
}

/** @brief
 * Performs element-wise AXPF operation (scalar AXPY)
 * Computes: result = summand + alpha * sumend (where sumend is a scalar)
 *
 ** @param summand
 * Pointer to the input tensor
 *
 ** @param sumend
 * Pointer to the scalar value to be scaled and added
 *
 ** @param alpha
 * Scalar multiplier for the sumend value
 *
 ** @return
 * Pointer to a new tensor containing the result of the AXPF operation 
 * (or NULL in case of fail)
 */
Tensor* naive_axpf(Tensor* summand, void* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(summand->shape, summand->rank, summand->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
        }
    #endif
    
    size_t total_elems = product(summand->shape, summand->rank);
    if(total_elems == 0) return result;
    
    switch(summand->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)summand->data; 
            double* cast_data_result = (double*)result->data;
            double cast_sumend = *((double*)sumend);
            double alpha_double = (double)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha_double);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)summand->data;
            float* cast_data_result = (float*)result->data;
            float cast_sumend = *((float*)sumend);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)summand->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t cast_sumend = *((int32_t*)sumend);
            int32_t alpha_int = (int32_t)alpha;
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha_int);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(summand->dtype);
            nnl2_free_tensor(result);
            return NULL;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPF operation
 */
Implementation axpf_backends[] = {
    REGISTER_BACKEND(naive_axpf, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPF operation
 * @ingroup backend_system
 */
axpffn axpf;

/**
 * @brief Sets the backend for AXPF operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPF operation
 */
void set_axpf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpf_backends, axpf, backend_name);
}

/** @brief
 * Performs element-wise AXPY operation with broadcasting (in place)
 * Computes: summand = summand + alpha * sumend
 *
 ** @param summand
 * Pointer to summand tensor (will be modified in place)
 *
 ** @param sumend
 * Pointer to sumend tensor
 *
 ** @param alpha
 * Scalar multiplier
 */
void naive_axpy_broadcasting_inplace(Tensor* summand, const Tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->shape, "Summand shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->shape, "Sumend shape is NULL");
    #endif
    
    size_t numel_summand = product(summand->shape, summand->rank);
    size_t numel_sumend = product(sumend->shape, sumend->rank);
    
    // Getting the tensor data types
    TensorType summand_dtype = summand->dtype;
    TensorType sumend_dtype = sumend->dtype;

    if((numel_summand % numel_sumend) == 0) {
        // Handling the case where the data types match (more efficiently)
        if(summand_dtype == sumend_dtype) {
            switch(summand_dtype) {
                case FLOAT64: {
                    double* cast_summand_data = (double*)summand->data;
                    double* cast_sumend_data = (double*)sumend->data;
                    double alpha_double = (double)alpha;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha_double;
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_summand_data = (float*)summand->data;
                    float* cast_sumend_data = (float*)sumend->data;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha;
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_summand_data = (int32_t*)summand->data;
                    int32_t* cast_sumend_data = (int32_t*)sumend->data;
                    int32_t alpha_int = (int32_t)alpha;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha_int;
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(summand_dtype);
                    return;
                }
            }
        } else {
            // Handling a case with different data types (conversion required)
            size_t sumend_step = get_dtype_size(sumend_dtype);
            char* sumend_data = (char*)sumend->data;
            
            switch(summand_dtype) {
                case FLOAT64: {
                    double* data_summand = (double*)summand->data;
                    double alpha_double = (double)alpha;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_float64(sumend_elem, sumend_dtype) * alpha_double;
                        }
                    }
                
                    break; 
                }
                
                case FLOAT32: {
                    float* data_summand = (float*)summand->data;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_float32(sumend_elem, sumend_dtype) * alpha;
                        }
                    }
                
                    break; 
                }
                
                case INT32: {
                    int32_t* data_summand = (int32_t*)summand->data;
                    int32_t alpha_int = (int32_t)alpha;
                
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* sumend_elem = sumend_data + j * sumend_step;
                            data_summand[i * numel_sumend + j] += nnl2_convert_to_int32(sumend_elem, sumend_dtype) * alpha_int;
                        }
                    }
                
                    break; 
                }
                
                default: {
                    NNL2_TYPE_ERROR(summand_dtype);
                    return;
                }
            }
        }
    } else {
        NNL2_ERROR("Cannot broadcast sumend tensor");
        return;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPY operation with broadcasting (in place)
 */
Implementation axpy_broadcasting_inplace_backends[] = {
    REGISTER_BACKEND(naive_axpy_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPY operation with broadcasting (in place)
 * @ingroup backend_system
 */
axpybroadcastinginplacefn axpy_broadcasting_inplace;

/**
 * @brief Sets the backend for AXPY operation with broadcasting (in place)
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY operation with broadcasting
 */
void set_axpy_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpy_broadcasting_inplace_backends, axpy_broadcasting_inplace, backend_name);
}

/** @brief
 * Performs element-wise AXPY operation with broadcasting support
 * Computes: result = summand + alpha * sumend
 *
 ** @param summand
 * First tensor to add
 *
 ** @param sumend
 * Second tensor to multiply and add
 *
 ** @param alpha
 * Scalar multiplier
 * 
 ** @return
 * New tensor containing the result of AXPY operation
 *
 ** @note
 * Contains type conversion
 */
Tensor* naive_axpy_broadcasting(Tensor* summand, Tensor* sumend, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks for max safety level 
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand, "Summand tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "Sumend tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(summand->shape, "Summand shape is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->shape, "Sumend shape is NULL", NULL);
    #endif
 
    size_t numel_summand = product(summand->shape, summand->rank);
    size_t numel_sumend = product(sumend->shape, sumend->rank);
    
    // Getting the tensor data types
    TensorType summand_dtype = summand->dtype;
    TensorType sumend_dtype = sumend->dtype;
    
    TensorType winner_in_the_type_hierarchy = MAX(summand_dtype, sumend_dtype);
    
    // Сreating a resultant tensor
    Tensor* result = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);

    if((numel_summand % numel_sumend) == 0) {
        if(summand_dtype == sumend_dtype) {
            // Handling the case where the data types match (more efficiently)
            switch(summand_dtype) {
                case FLOAT64: {
                    double* cast_summand_data = (double*)summand->data;
                    double* cast_sumend_data = (double*)sumend->data;
                    double* cast_result_data = (double*)result->data;
                    double alpha_double = (double)alpha;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + (cast_sumend_data[j] * alpha_double);
                        }
                    }

                    break;
                }

                case FLOAT32: {
                    float* cast_summand_data = (float*)summand->data;
                    float* cast_sumend_data = (float*)sumend->data;
                    float* cast_result_data = (float*)result->data;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + (cast_sumend_data[j] * alpha);
                        }
                    }

                    break;
                }

                case INT32: {
                    int32_t* cast_summand_data = (int32_t*)summand->data;
                    int32_t* cast_sumend_data = (int32_t*)sumend->data;
                    int32_t* cast_result_data = (int32_t*)result->data;
                    int32_t alpha_int = (int32_t)alpha;

                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + (cast_sumend_data[j] * alpha_int);
                        }
                    }

                    break;
                }

                default: {
                    NNL2_TYPE_ERROR(summand_dtype);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        } else {
            // Handling a case with different data types (conversion required)
            size_t summand_step = get_dtype_size(summand_dtype);
            size_t sumend_step = get_dtype_size(sumend_dtype);
            
            switch(winner_in_the_type_hierarchy) {
                case FLOAT64: {
                    double* cast_data_result = (double*)result->data;
                    double alpha_double = (double)alpha;
                    
                    char* cast_summand_data = (char*)summand->data;
                    char* cast_sumend_data =  (char*)sumend->data;
                    
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
                            void* elem_sumend = cast_sumend_data + j * sumend_step; 
                            
                            cast_data_result[i * numel_sumend + j] = nnl2_convert_to_float64(elem_summand, summand_dtype) + (nnl2_convert_to_float64(elem_sumend, sumend_dtype) * alpha_double);
                        }
                    }
                    
                    break;
                }
                
                case FLOAT32: {
                    float* cast_data_result = (float*)result->data;
                    
                    char* cast_summand_data = (char*)summand->data;
                    char* cast_sumend_data =  (char*)sumend->data;
                    
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
                            void* elem_sumend = cast_sumend_data + j * sumend_step;
                            
                            cast_data_result[i * numel_sumend + j] = nnl2_convert_to_float32(elem_summand, summand_dtype) + (nnl2_convert_to_float32(elem_sumend, sumend_dtype) * alpha);
                        }
                    }
                    
                    break;
                }
                
                case INT32: {
                    int32_t* cast_data_result = (int32_t*)result->data;
                    int32_t alpha_int = (int32_t)alpha;
                    
                    char* cast_summand_data = (char*)summand->data;
                    char* cast_sumend_data =  (char*)sumend->data;
                    
                    for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {                    
                        for(size_t j = 0; j < numel_sumend; j++) {
                            void* elem_summand = cast_summand_data + (i * numel_sumend + j) * summand_step;
                            void* elem_sumend = cast_sumend_data + j * sumend_step;
                        
                            cast_data_result[i * numel_sumend + j] = nnl2_convert_to_int32(elem_summand, summand_dtype) + (nnl2_convert_to_int32(elem_sumend, sumend_dtype) * alpha_int);
                        }
                    }
                    
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                    nnl2_free_tensor(result);
                    return NULL;
                }
            }
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
    } else {
        NNL2_ERROR("Cannot broadcast sumend tensor");
        return NULL;
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for AXPY operation with broadcasting
 */
Implementation axpy_broadcasting_backends[] = {
    REGISTER_BACKEND(naive_axpy_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for AXPY operation with broadcasting
 * @ingroup backend_system
 */
axpybroadcastingfn axpy_broadcasting;

/**
 * @brief Sets the backend for AXPY operation with broadcasting
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for AXPY operation with broadcasting
 */
void set_axpy_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpy_broadcasting_backends, axpy_broadcasting, backend_name);
}

/** @brief
 * The function is exclusively for the lisp wrapper, 
 * and in C, use nnl2_copy (the same argument list as here)
 *
 ** @param tensor
 * Input tensor
 *
 ** @param cast_to
 * Specifies which type of tensor to cast
 *
 ** @return
 * Tensor with the same data but a new (specified) type
 */
Tensor* nnl2_cast(Tensor* tensor, TensorType cast_to) {
	return nnl2_copy(tensor, cast_to);
}

/** @brief
 * Reshapes a tensor to a new shape with optional wildcard support
 *
 ** @param tensor 
 * Pointer to the input tensor to be reshaped
 *
 ** @param new_shape
 * Array containing the target shape dimensions
 *
 ** @param new_shape_len
 * Number of dimensions in the new shape
 *
 ** @param force
 * If true, bypasses element count validation
 *
 ** @note
 * Wildcard dimension (-1) must appear at most once in the new_shape array
 *
 ** @warning
 * Using force=true can lead to undefined behavior if shapes are incompatible
 *
 ** @exception NNL2_ERROR_MEMORY_ALLOCATION
 * Failed to allocate memory for shape buffer
 *
 ** @exception NNL2_ERROR_SHAPE_OVERFLOW 
 * Shape product would exceed maximum size
 *
 ** @exception NNL2_ERROR_WILDCARD_COUNT 
 * More than one wildcard dimension found
 *
 ** @exception NNL2_ERROR_WILDCARD_COMPUTE 
 * Cannot compute wildcard dimension
 *
 ** @exception NNL2_ERROR_SHAPE_MISMATCH 
 * Element count doesn't match and force=false
 *
 ** @exception NNL2_ERROR_TENSOR_ALLOCATION 
 * Failed to allocate new tensor
 *
 ** @exception NNL2_ERROR_UNSUPPORTED_TYPE 
 * Unsupported tensor data type
 *
 ** @code
 * // Example 1: Without wildcard
 * // Original tensor shape: [2, 3] (6 elements)
 * Tensor* original_tensor_no_wildcard = nnl2_zeros((int[]){2, 3}, 2, FLOAT64) // [[0, 0, 0], [0, 0, 0]]
 * Tensor* reshaped_tensor_no_wildcard = nnl2_naive_reshape(original_tensor_no_wildcard, (int[]){3, 2}, 2, false) // [[0, 0], [0, 0], [0, 0]]
 * // Result: shape [2, 3] -> [3, 2] (element count matches: 2 * 3 = 6 and 3 * 2 = 6)
 ** @endcode
 **
 ** @code
 * // Example 2: With wildcard
 * // Original tensor shape: [3, 4] (12 elements)
 * Tensor* original_tensor_with_wildcard = nnl2_zeros((int[]){3, 4}, 2, FLOAT64) // [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
 * Tensor* reshaped_tensor_with_wildcard = nnl2_naive_reshape(original_tensor_with_wildcard, (int[]){4, -1}, 2, false) // -1 -> 3 ([4, 3]), [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
 ** @endcode
 **
 ** @see nnl2_empty
 ** @see nnl2_free_tensor
 ** @see product
 **/
Tensor* nnl2_naive_reshape(Tensor* tensor, int32_t* new_shape, int32_t new_shape_len, bool force) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
	#endif
	
	// Early return if shapes are identical
    if (tensor->rank == new_shape_len && memcmp(tensor->shape, new_shape, new_shape_len * sizeof(int32_t)) == 0) {
        return nnl2_copy(tensor, tensor->dtype); 
    }
	
	// Calculate total elements from original tensor
    size_t total_elems = product(tensor->shape, tensor->rank);
	
	// Allocate temporary buffer for shape processing (including wildcard resolution)
    int32_t* wildcard_shape = malloc(new_shape_len * sizeof(int32_t));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!wildcard_shape) {
			NNL2_ERROR("Memory allocation failed");
			return NULL;
		}
	#endif
    
    int32_t wildcard_index = NNL2_WILDCARD_DIM;		// Index of wildcard dimension (-1 if not found)
    size_t wildcard_count = 0;						// Number of wildcard dimensions found
    size_t wildcard_shape_product = 1;  			// Product of non-wildcard dimensions
    
    for(int32_t i = 0; i < new_shape_len; i++) {
        wildcard_shape[i] = new_shape[i];
        if(new_shape[i] == NNL2_WILDCARD_DIM) {  // if(new_shape[i] == -1)
			// Found wildcard dimension
            wildcard_index = i;
            wildcard_count++;
        } else if (new_shape[i] < NNL2_WILDCARD_DIM) {
			NNL2_ERROR("Invalid shape dimension: %d", new_shape[i]);
			free(wildcard_shape);
			return NULL;
		} else {
			// Non-wildcard dimension
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if (new_shape[i] > 0) {
					// Check for multiplication overflow
					if (wildcard_shape_product > SIZE_MAX / new_shape[i]) {
						NNL2_ERROR("Shape product overflow at dimension %d: %d * %d would exceed maximum size", i, wildcard_shape_product, new_shape[i]);
						free(wildcard_shape);
						return NULL;
					}
				}
			#endif
			
            wildcard_shape_product *= new_shape[i];
        }
    }
    
	// Validate wildcard count (0 or 1 allowed)
    if(wildcard_count > 1) {
        NNL2_ERROR("Must have at most one wildcard (-1), found %d", wildcard_count);
        free(wildcard_shape);
        return NULL;
    }
    
    if(wildcard_count == 1) {
		// Handle wildcard dimension case
        if(wildcard_shape_product == 0 || total_elems % wildcard_shape_product != 0) {
            NNL2_ERROR("Cannot compute wildcard: %d %% %d != 0", total_elems, wildcard_shape_product);
            free(wildcard_shape);
            return NULL;
        }
        
		// Calculate and set the wildcard dimension value
        int32_t wildcard_value = total_elems / wildcard_shape_product;
        wildcard_shape[wildcard_index] = wildcard_value;
    } else {
		// No wildcard case
        if(total_elems != wildcard_shape_product && !force) {
            NNL2_ERROR("Number of elements for reshape does not match: expected %d, got %d", total_elems, wildcard_shape_product);
            free(wildcard_shape);
            return NULL;
        }
    }
    
	// Create new tensor with the resolved shape
    Tensor* new_tensor = nnl2_empty(wildcard_shape, new_shape_len, tensor->dtype);
    free(wildcard_shape); // Free temporary shape buffer
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(new_tensor == NULL) {
			NNL2_ERROR("Tensor allocation failed");
		}
	#endif
    
	// Copy data from original tensor to reshaped tensor
    switch(tensor->dtype) {
        case FLOAT64: {
            double* reshape_data = (double*)new_tensor->data;  // Casting
            double* original_data = (double*)tensor->data;
            for(size_t it = 0; it < total_elems; it++) reshape_data[it] = original_data[it]; // Copying
            break;
        }
        
        case FLOAT32: {
            float* reshape_data = (float*)new_tensor->data;
            float* original_data = (float*)tensor->data;
            for(size_t it = 0; it < total_elems; it++) reshape_data[it] = original_data[it];
            break;
        }
        
        case INT32: {
            int32_t* reshape_data = (int32_t*)new_tensor->data;
            int32_t* original_data = (int32_t*)tensor->data;
            for(size_t it = 0; it < total_elems; it++) reshape_data[it] = original_data[it];
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            nnl2_free_tensor(new_tensor);
            return NULL;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return new_tensor;
}
 
/** 
 * @ingroup backend_system
 * @brief Backend implementations for reshape operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_reshape: Basic reference implementation
 * 
 * @see nnl2_naive_reshape
 */
Implementation reshape_backends[] = {
    REGISTER_BACKEND(nnl2_naive_reshape, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for reshape operation
 * @ingroup backend_system 
 */
reshapefn nnl2_reshape;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(reshape);

/** 
 * @brief Sets the backend for reshape operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_reshape_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(reshape_backends, nnl2_reshape, backend_name, CURRENT_BACKEND(reshape));
}

/** 
 * @brief Gets the name of the active backend for reshape operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_reshape_backend() {
    return CURRENT_BACKEND(reshape);
}

/** 
 * @brief Function declaration for getting all `reshape` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(reshape);

/**
 * @brief Function declaration for getting the number of all `reshape` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(reshape);

// Reinterpret definition (start)

/** @brief
 * Helper function to create a view of the entire tensor
 * Used internally by reinterpret_reshape for identical shape case
 *
 ** @param tensor
 * Pointer to the source tensor
 *
 ** @param indices
 * Optional indices for partial view (NULL for full view)
 *
 ** @param num_indices
 * Number of indices (0 for full view)
 **/
static Tensor* nnl2_create_view(Tensor* tensor, int32_t* indices, uint8_t num_indices);

/** @brief
 * Reshapes a tensor to a new shape with optional wildcard support, returning a view
 * This function creates a new tensor that shares data with the original tensor
 *
 ** @param tensor 
 * Pointer to the input tensor to be reshaped
 *
 ** @param new_shape
 * Array containing the target shape dimensions
 *
 ** @param new_shape_len
 * Number of dimensions in the new shape
 *
 ** @param force
 * If true, bypasses element count validation
 *
 ** @note
 * Wildcard dimension (-1) must appear at most once in the new_shape array
 * The returned tensor shares data with the original tensor - modifications affect both
 *
 ** @warning
 * Using force=true can lead to undefined behavior if shapes are incompatible
 * The original tensor must not be freed while reinterpreted views exist
 *
 ** @exception NNL2_ERROR_SHAPE_OVERFLOW 
 * Shape product would exceed maximum size
 *
 ** @exception NNL2_ERROR_WILDCARD_COUNT 
 * More than one wildcard dimension found
 *
 ** @exception NNL2_ERROR_WILDCARD_COMPUTE 
 * Cannot compute wildcard dimension
 *
 ** @exception NNL2_ERROR_SHAPE_MISMATCH 
 * Element count doesn't match and force=false
 *
 ** @code
 * // Example: Create view with different shape
 * Tensor* original = nnl2_zeros((int[]){2, 3}, 2, FLOAT64);
 * Tensor* view = nnl2_naive_reinterpret(original, (int[]){3, 2}, 2, false);
 * // Both tensors share the same underlying data
 ** @endcode
 **
 ** @code
 * // Example: Create view with different shape with wildcard
 * Tensor* original = nnl2_zeros((int[]){2, 3}, 2, FLOAT64);
 * Tensor* view = nnl2_naive_reinterpret(original, (int[]){3, -1}, 2, false); // -1 Is wildcard. New shape: [3, 2]
 ** @endcode
 ** @see nnl2_naive_reshape
 ** @see nnl2_naive_view
 **/
Tensor* nnl2_naive_reinterpret(Tensor* tensor, int32_t* new_shape, int32_t new_shape_len, bool force) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
    #endif
    
    // Early return if shapes are identical
    if (tensor->rank == new_shape_len && memcmp(tensor->shape, new_shape, new_shape_len * sizeof(int32_t)) == 0) {
        return nnl2_create_view(tensor, NULL, 0); 
    }
    
    // Calculate total elements from original tensor
    size_t total_elems = product(tensor->shape, tensor->rank);
    
    // Process shape and handle wildcards
    int32_t wildcard_index = -1; 
    size_t wildcard_count = 0;
    size_t wildcard_shape_product = 1;
    
    for(int32_t i = 0; i < new_shape_len; i++) {
        if(new_shape[i] == -1) { // Wildcard dimension
            wildcard_index = i;
            wildcard_count++;
        } else if (new_shape[i] < 0) { // Negative but not wildcard
            NNL2_ERROR("Invalid shape dimension: %d", new_shape[i]);
            return NULL;
        } else {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                if (new_shape[i] > 0) {
                    if (wildcard_shape_product > SIZE_MAX / new_shape[i]) {
                        NNL2_ERROR("Shape product overflow at dimension %d", i);
                        return NULL;
                    }
                }
            #endif
			
            wildcard_shape_product *= new_shape[i];
        }
    }
    
    // Validate wildcard count
    if(wildcard_count > 1) {
        NNL2_ERROR("Must have at most one wildcard (-1), found %d", wildcard_count);
        return NULL;
    }
    
    // Resolve wildcard dimension if present
    int32_t* resolved_shape = new_shape;
    
    if(wildcard_count == 1) {
        if(wildcard_shape_product == 0 || total_elems % wildcard_shape_product != 0) {
            NNL2_ERROR("Cannot compute wildcard: %d %% %d != 0", total_elems, wildcard_shape_product);
            return NULL;
        }
        
        // Create temporary resolved shape using malloc
        int32_t* temp_shape = (int32_t*)malloc(new_shape_len * sizeof(int32_t));
        if (!temp_shape) {
            NNL2_ERROR("Failed to allocate temporary shape");
            return NULL;
        }
        
        memcpy(temp_shape, new_shape, new_shape_len * sizeof(int32_t));
        temp_shape[wildcard_index] = total_elems / wildcard_shape_product;
        resolved_shape = temp_shape;
    } else {
        // No wildcard
        if(total_elems != wildcard_shape_product && !force) {
            NNL2_ERROR("Number of elements for reshape does not match: expected %d, got %d", total_elems, wildcard_shape_product);
            return NULL;
        }
    }
    
    // Create view tensor 
    Tensor* view_tensor = (Tensor*)malloc(sizeof(Tensor));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!view_tensor) {
            NNL2_ERROR("Failed to allocate view tensor");
            if (wildcard_count == 1) free((void*)resolved_shape);
            return NULL;
        }
    #endif
    
    // Initialize view tensor structure
    view_tensor->dtype = tensor->dtype;
    view_tensor->rank = new_shape_len;
    view_tensor->data = tensor->data; // Shared data
	view_tensor->is_view = true;
    
    // Allocate and copy shape
    view_tensor->shape = (int32_t*)malloc(new_shape_len * sizeof(int32_t));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!view_tensor->shape) {
            NNL2_ERROR("Failed to allocate shape for view tensor");
            if (wildcard_count == 1) free((void*)resolved_shape);
            free(view_tensor);
            return NULL;
        }
    #endif
    
    memcpy(view_tensor->shape, resolved_shape, new_shape_len * sizeof(int32_t));
    
    // Free temporary shape if allocated it for wildcard resolution
    if (wildcard_count == 1) {
        free((void*)resolved_shape);
    }
    
    // Calculate strides for the new shape 
    view_tensor->strides = (int32_t*)malloc(new_shape_len * sizeof(int32_t));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!view_tensor->strides) {
            NNL2_ERROR("Failed to allocate strides for view tensor");
            free(view_tensor->shape);
            free(view_tensor);
            return NULL;
        }
    #endif
    
    // Compute strides for the new shape
    view_tensor->strides[new_shape_len - 1] = 1;
    for (int32_t i = new_shape_len - 2; i >= 0; i--) {
        view_tensor->strides[i] = view_tensor->strides[i + 1] * view_tensor->shape[i + 1];
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return view_tensor;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_create_view
 **/
static Tensor* nnl2_create_view(Tensor* tensor, int32_t* indices, uint8_t num_indices) {
    Tensor* view = (Tensor*)malloc(sizeof(Tensor));
    if (!view) return NULL;
    
    view->dtype = tensor->dtype;
    view->rank = tensor->rank;

    // Allocate and copy shape
    view->shape = (int32_t*)malloc(tensor->rank * sizeof(int32_t));
    if (!view->shape) {
        free(view);
        return NULL;
    }
    memcpy(view->shape, tensor->shape, tensor->rank * sizeof(int32_t));
    
    // Allocate and copy strides
    view->strides = (int32_t*)malloc(tensor->rank * sizeof(int32_t));
    if (!view->strides) {
        free(view->shape);
        free(view);
        return NULL;
    }
    memcpy(view->strides, tensor->strides, tensor->rank * sizeof(int32_t));
    
    // Calculate data offset if indices are provided
    size_t offset = 0;
    if (indices && num_indices > 0) {
        for (uint8_t i = 0; i < num_indices; i++) {
            offset += indices[i] * tensor->strides[i];
        }
    }
    
    const size_t element_size = get_dtype_size(tensor->dtype);
    view->data = (char*)tensor->data + offset * element_size;
    
    return view;
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for reinterpret operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_reinterpret: Basic reference implementation
 * 
 * @see nnl2_naive_reinterpret
 */
Implementation reinterpret_backends[] = {
    REGISTER_BACKEND(nnl2_naive_reinterpret, nnl2_naive, NAIVE_BACKEND_NAME),
};  

/**
 * @brief Function pointer for reinterpret operation
 * @ingroup backend_system 
 */
reinterpretfn nnl2_reinterpret;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(reinterpret);

/** 
 * @brief Sets the backend for reinterpret operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_reinterpret_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(reinterpret_backends, nnl2_reinterpret, backend_name, CURRENT_BACKEND(reinterpret));
}

/** 
 * @brief Gets the name of the active backend for reinterpret operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_reinterpret_backend() {
    return CURRENT_BACKEND(reinterpret);
}

/** 
 * @brief Function declaration for getting all `reinterpret` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(reinterpret);

/**
 * @brief Function declaration for getting the number of all `reinterpret` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(reinterpret);

// Reinterpret definition (end)

/** @brief
 * Gets a pointer to a tensor element by linear index
 *
 ** @param tensor 
 * Tensor to take element with linear index
 *
 ** @param at
 * Linear index
 *
 ** @return void* 
 * Pointer to the requested tensor element
 */
void* nnl2_get_raw_tensor_elem_at(Tensor* tensor, size_t at) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		size_t total_elems = product(tensor->shape, tensor->rank);
		if(at >= total_elems) {
			NNL2_ERROR("Index out of bounds: index %zd exceeds tensor size %zd", at, total_elems);
			return NULL;
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
	
	return ((char*)tensor->data + at * get_dtype_size(tensor->dtype));
}

/** @brief
 * Sets a tensor element by linear index
 *
 ** @param tensor 
 * Tensor to set element in
 *
 ** @param at
 * Linear index
 *
 ** @param with
 * Pointer to data to set
 */
void nnl2_set_raw_tensor_elem_at(Tensor* tensor, size_t at, void* with) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        size_t total_elems = product(tensor->shape, tensor->rank);
        if(at >= total_elems) {
            NNL2_ERROR("Index out of bounds: index %zd exceeds tensor size %zd", at, total_elems);
            return;
        }
    #endif
	
	size_t elem_size = get_dtype_size(tensor->dtype);
    char* dest = (char*)tensor->data + at * elem_size;
	memcpy(dest, with, elem_size);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief
 * Gets a pointer to a tensor element by coordinates (indexes)
 *
 ** @param tensor
 * Tensor to take element from
 *
 ** @param coords
 * Indices to get element
 *
 ** @param coords_len
 * Length of indices 
 *
 ** @return void* 
 * Pointer to the requested tensor element
 */
void* nnl2_get_raw_tensor_elem(Tensor* tensor, int32_t* coords, int32_t coords_len) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (coords_len != tensor->rank) {
			NNL2_ERROR("Length of indexes (%d) doesn't match tensor dimension (%d)", coords_len, tensor->rank);
			return NULL;
		}
		
		for (int i = 0; i < coords_len; i++) {
			if (coords[i] < 0 || coords[i] >= tensor->shape[i]) {
				NNL2_ERROR("Index at %d is out of bounds (tensor shape at %d is %d)", coords[i], i, tensor->shape[i]);
				return NULL;
			}
		}
	#endif
	
			
	size_t offset = 0;
	for (int i = 0; i < coords_len; i++) {
		offset += coords[i] * tensor->strides[i];
	}
	
	char* elem = (char*)tensor->data + offset * get_dtype_size(tensor->dtype);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
    
    return elem;
}

/** @brief
 * Sets a tensor element by coordinates (indexes)
 *
 ** @param tensor
 * Tensor to set element in
 *
 ** @param coords
 * Indices to set element at
 *
 ** @param coords_len
 * Length of indices
 *
 ** @param with
 * Pointer to data to set
 */
void nnl2_set_raw_tensor_elem(Tensor* tensor, int32_t* coords, int32_t coords_len, void* with) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (coords_len != tensor->rank) {
            NNL2_ERROR("Length of indexes (%d) doesn't match tensor dimension (%d)", coords_len, tensor->rank);
            return;
        }
        
        for (int i = 0; i < coords_len; i++) {
            if (coords[i] < 0 || coords[i] >= tensor->shape[i]) {
                NNL2_ERROR("Index at %d is out of bounds (tensor shape at %d is %d)", coords[i], i, tensor->shape[i]);
                return;
            }
        }
    #endif
    
    size_t offset = 0;
    for (int i = 0; i < coords_len; i++) {
        offset += coords[i] * tensor->strides[i];
    }
    
    size_t elem_size = get_dtype_size(tensor->dtype);
    char* dest = (char*)tensor->data + offset * elem_size;
    memcpy(dest, with, elem_size);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif  /** NNL2_TENSOR_ACCESSORS_H **/ 
