#include "nnl2_core.h"
#include "nnl2_log.h"
#include "nnl2_tensor_backend.h"
#include "nnl2_backend_system_docs.h"

#include <string.h>
#include <math.h>

#ifndef NNL2_TENSOR_ACCESSORS
#define NNL2_TENSOR_ACCESSORS

#ifdef _WIN32
 
#include <malloc.h>

#define ALLOC_ALIGNED(ptr, alignment, size) \
    do { \
        ptr = _aligned_malloc(size, alignment); \
        if (ptr == NULL) { fprintf(stderr, "Error (Hello from C!): Failed to allocate memory\n"); } \
    } while(0)
#define FREE_ALIGNED(ptr) _aligned_free(ptr)

#else
	
#define _POSIX_C_SOURCE 200809L

#include <stdlib.h> 

#define ALLOC_ALIGNED(ptr, alignment, size)  \
	do { \
        int err = posix_memalign(&ptr, alignment, size); \
        if (err != 0) { fprintf(stderr, "Error (Hello from C!): Failed to allocate memory\n"); } \
    } while(0)
#define FREE_ALIGNED(ptr) free(ptr)

#endif

#define NUM_TENSOR_TYPES 3

#if defined(__AVX512F__)
	#define TENSOR_MEM_ALIGNMENT 64
#elif defined(__AVX2__)
	#define TENSOR_MEM_ALIGNMENT 32
#elif defined(__AVX__)
	#define TENSOR_MEM_ALIGNMENT 32
#else 
	#define TENSOR_MEM_ALIGNMENT 16
#endif

#if defined(_MSC_VER)
    #define NNL2_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define NNL2_FORCE_INLINE inline __attribute__((always_inline))
#else
    #define NNL2_FORCE_INLINE inline
#endif

#ifdef OPENBLAS_AVAILABLE
#include <cblas.h>
#endif

/// @{

/** @brief
 * I had to delete all these macros after a certain period of time. 
 * if you see this, it means that I forgot, please let me know
 */
#define current_backend(name) current_##name##_backend_name 
#define make_current_backend(name) static char current_##name##_backend_name[MAX_BACKEND_NAME_LENGTH] = ""

/// @}
	
#define CURRENT_BACKEND(name) current_##name##_backend_name 
#define MAKE_CURRENT_BACKEND(name) static char current_##name##_backend_name[MAX_BACKEND_NAME_LENGTH] = ""

#define NAIVE_BACKEND_NAME "NAIVE"
#define AVX128_BACKEND_NAME "AVX128"
#define AVX256_BACKEND_NAME "AVX256"
#define BLAS_BACKEND_NAME "BLAS"

#define MIN_TENSOR_DIMENSION 0

#define NNL2_SAFETY_MODE_OFF 0
#define NNL2_SAFETY_MODE_MIN 1
#define NNL2_SAFETY_MODE_MODERATE 2
#define NNL2_SAFETY_MODE_MAX 3

#define NNL2_SAFETY_MODE NNL2_SAFETY_MODE_MAX

#define NNL2_STRIDES_LAST_DIM_INDEX 1
#define NNL2_STRIDES_SECOND_LAST_DIM_INDEX 2
#define NNL2_STRIDES_NEXT_DIM_INDEX 1

// NNL2

/** @file nnl2_tensor_accessors.h
 ** @brief Contains all operations for accessing tensors of type tref
 ** @copyright MIT License
 *
 * The file fully includes tensor accessors, 
 * i.e. tref getters and setters, as well as 
 * some additional functions
 *
 ** Filepath: nnl2/src/c/nnl2_tensor_accessors.h
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
 * Returns the size of the TensorType data type in bytes
 *
 * This function uses a constant size array to quickly determine the size of the
 * data type represented by the TensorType enumerations
 *
 ** @param dtype
 * The data type of the tensor (TensorType)
 *
 ** @return
 * The size of the specified data type in bytes
 *
 ** @note
 * Can perform additional checks depending on the safety level
 *
 ** @note
 * In debug mode, it can output additional information to the console
 *
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 **/
inline size_t get_dtype_size(TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();	
	    NNL2_DEBUG("Function get_dtype_size was called with dtype=%d", dtype);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if(dtype > (NUM_TENSOR_TYPES - 1)) {
			NNL2_ERROR("Invalid data type: %d", dtype);
		}
	#endif
		
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL    
		NNL2_FUNC_EXIT();	
	#endif
	
	return (const size_t[]){sizeof(int), sizeof(float), sizeof(double)}[dtype]; 
}	

/** @brief
 * Calculates the total number of elements in the tensor specified by the shape (for calculating memory)
 *
 ** @param lst
 * Pointer to an array of integers representing the tensor's shape
 *
 ** @param len
 * Length of the array `lst`, which is the number of dimensions in the tensor
 *
 ** @return 
 * Total number of elements in the tensor
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * size_t num_elements = product(shape, 3); // num_elements will be 24
 ** @endcode
 **
 ** @note
 * Uses a forced inline
 *
 ** @note
 * Displays a lot of debug information at the 
 * maximum debug level. If the debug level is 
 * lower than the maximum, nothing is displayed
 *
 ** @note
 * Сan perform additional checks at a high level of safety
 *
 ** @note
 * In debug mode, it can output additional information to the console
 *
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 ** @see NNL2_FORCE_INLINE
 **/
NNL2_FORCE_INLINE size_t product(const int32_t* lst, int32_t len) { // todo rename from product to nnl2_product
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();	
		int32_t original_len = len;
	#endif
	
	// Additional checks when the debugging level is sufficient 
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		if (lst == NULL) {
			NNL2_ERROR("product(): NULL pointer passed as shape array");
			return 0;
		}
	
		if (len <= 0) {
			NNL2_ERROR("product(): Invalid length %d", len);
			return 0;
		}
	#endif
	
	size_t acc = 1;
	while (len--) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE 
			if (*lst <= 0) {
				NNL2_ERROR("product(): Invalid dimension value %d", *lst);
				return 0;
			}
		#endif
		
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
			if (acc > SIZE_MAX / (size_t)(*lst)) {
				NNL2_ERROR("Multiplication overflow in product()");
			}
        #endif
		
		// Calculation
		acc *= (size_t)*lst++;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_DEBUG("Product calculated for shape[%d], result=%zu", original_len, acc);
		NNL2_FUNC_EXIT();
	#endif
	
	return acc;
}

/** @brief
 * Creates a new tensor without initializing the data
 *
 * This function allocates memory for the Tensor structure and its data,
 * using the provided shape and data type. The data memory is not (!) initialized
 *
 * The created tensor will have the numel field pre-calculated for
 * optimal performance in subsequent operations
 *
 * And also, the tensor has strides fields that will be initialized inside
 *
 ** @param shape
 * A pointer to an array of integers representing the tensor's shape
 *
 ** @param rank
 * The number of dimensions of the tensor
 *
 ** @param dtype
 * Tensor data type (TensorType)
 *
 ** @return 
 * pointer to a new tensor or NULL in case of an error
 *
 ** @details
 * The function firstly:
 *
 *
 *** Checks the input parameters for correctness (namely shape, rank, dtype)
 ** Then
 *** Allocates memory for tensor structure
 ** Then
 *** Allocates memory for the shape array and copies the data into it
 ** Then
 *** Calculates the total number of elements and strides
 ** Then
 *** Checks for multiplication overflow to prevent size_t 
 *** overflow when calculating total memory size
 ** Then 
 *** Allocates aligned memory for the tensor data (tensor->data)
 ** Finally
 *** Returns a pointer to the created tensor
 *
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * Tensor* my_tensor = nnl2_empty(shape, 3, FLOAT32);
 * if (my_tensor == NULL) {
 *     // Handle error
 * }
 ** @endcode
 ** @note
 * The function performs comprehensive error checking including:
 ** - NULL pointer validation
 ** - Parameter range validation  
 ** - Memory allocation failure handling
 ** - Multiplication overflow protection
 ** - Type safety checks
 *
 ** @note
 * In debug mode, it can output additional information to the console
 *
 ** @note
 * Сan perform additional checks at a high level of safety
 *
 ** @warning
 * Do not forget to free the memory allocated for the tensor using free_tensor after using it
 *
 ** @see free_tensor
 ** @see get_dtype_size
 ** @see product
 ** @see ALLOC_ALIGNED
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 ** @see NNL2_STRIDES_LAST_DIM_INDEX
 ** @see NNL2_STRIDES_SECOND_LAST_DIM_INDEX
 ** @see NNL2_STRIDES_NEXT_DIM_INDEX
 *
 ** @exception[out_of_memory] 
 * If memory allocation fails for tensor structure, shape array, or data
 *
 ** @exception[invalid_argument]
 * If shape is NULL, rank <= 0, invalid dtype, or any dimension <= 0
 *
 ** @exception[overflow_error]
 * If the calculated total size would overflow size_t
 *
 **/
Tensor* nnl2_naive_empty(const int32_t* shape, const int32_t rank, const TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif

	// Checks the input parameters for correctness
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (shape == NULL) {
			NNL2_ERROR("Bad shape pointer (NULL)"); 
			return NULL;
		}	
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (rank <= 0) {
			NNL2_ERROR("Bad rank (%d). Rank must be positive", rank);
			return NULL;
		}
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (dtype < 0 || dtype >= NUM_TENSOR_TYPES) {
			NNL2_ERROR("Bad tensor type (%d)", dtype);
			return NULL;
		}
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		for (int i = 0; i < rank; i++) {
			if (shape[i] <= 0) {
				NNL2_ERROR("Bad shape dimension at %d (%d). Dimensions must be positive", i, shape[i]);
				return NULL;
			}
		}
	#endif
	
	// Allocating memory for tensor structure

	Tensor* tensor = malloc(sizeof(Tensor));
	
	if (tensor == NULL) {
        NNL2_ERROR("Failed to allocate memory for Tensor structure");
        return NULL;
    }
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	
	// Allocates memory for the shape array and copies the data into it
	
	size_t total_elems = 1;
	tensor->shape = malloc(rank * sizeof(int));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor->shape == NULL) {
			NNL2_ERROR("Failed to allocate memory for shape");
			free(tensor);
			return NULL;
		}
	#endif
	
	memcpy(tensor->shape, shape, rank * sizeof(int));
	
	tensor->strides = malloc(rank);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor->strides == NULL) {
			NNL2_ERROR("Failed to allocate memory for strides");
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif
	
	// Initializes the step for the last dimension
	// Stride for the innermost (last) dimension is 1
	tensor->strides[rank - NNL2_STRIDES_LAST_DIM_INDEX] = 1;
	total_elems *= shape[rank - NNL2_STRIDES_LAST_DIM_INDEX];
	
	// Going from the second-to-last dimension to the first (from left to right in shape)
	for(int i = rank - NNL2_STRIDES_SECOND_LAST_DIM_INDEX; i >= 0; i--) {
		tensor->strides[i] = tensor->strides[i + NNL2_STRIDES_NEXT_DIM_INDEX] * shape[i + NNL2_STRIDES_NEXT_DIM_INDEX];
		total_elems *= shape[i];
	}

	tensor->numel = total_elems;
	
	size_t type_size = get_dtype_size(dtype);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		// Check for multiplication overflow
		if (type_size != 0 && total_elems > SIZE_MAX / type_size) {
			NNL2_ERROR("Tensor size too large, multiplication overflow (elems: %zu, type size: %zu)", total_elems, type_size);
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif

	size_t total_size = total_elems * type_size;
	
	// Allocates aligned memory for tensor data (tensor->data)
	
	void* data;
	
	ALLOC_ALIGNED(data, (size_t)TENSOR_MEM_ALIGNMENT, total_size);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (data == NULL) {
			NNL2_ERROR("Failed to allocate aligned memory");
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif
	
	tensor->data = data;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Created empty tensor: %dD, elems=%zu, size=%zu bytes", rank, total_elems, total_size);
		#endif
	
		NNL2_FUNC_EXIT();
	#endif
	
	return tensor;
}

/** @ingroup backend_system
 ** @brief Backend implementations for empty tensor creation
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_empty: Basic reference implementation
 *
 ** @see REGISTER_BACKEND
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_naive_empty
 ** @see nnl2_naive
 **/
Implementation nnl2_empty_backends[] = {
	REGISTER_BACKEND(nnl2_naive_empty, nnl2_naive, NAIVE_BACKEND_NAME)
};

/**
 * @brief Function pointer for the active empty() backend
 * @ingroup backend_system 
 */
fn_empty nnl2_empty;

/** 
 * @brief Sets the backend for empty tensor creation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 */
void nnl2_set_empty_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nnl2_empty_backends, nnl2_empty, backend_name);
}

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(nnl2_empty);

/** 
 * @brief Gets the name of the active backend for empty()
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* nnl2_get_empty_backend() {
	return CURRENT_BACKEND(nnl2_empty);
}

/** 
 * @brief Function declaration for getting all `empty` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(nnl2_empty);

/**
 * @brief Function declaration for getting the number of all `empty` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(nnl2_empty);

/** @brief
 * Creates a new tensor and initializes all elements to zero
 *
 * This function allocates memory for tensor structure and its data,
 * using the provided shape and data type
 *
 * Memory is initialized to zero
 *
 * The created tensor will have the numel field pre-calculated for
 * optimal performance in subsequent operations
 *
 * And also, the tensor has strides fields that will be initialized inside
 *
 ** @param shape
 * A pointer to an array of integers representing the tensor's shape
 *
 ** @param rank
 * The number of dimensions of the tensor
 *
 ** @param dtype
 * Tensor data type (TensorType)
 *
 ** @return 
 * pointer to a new tensor or NULL in case of an error
 *
 ** @details
 * The function firstly:
 *
 *
 *** Checks the input parameters for correctness
 ** Then
 *** Allocates memory for tensor structure
 ** Then
 *** Allocates memory for the shape array and copies the data into it
 ** Then
 *** Calculates the total number of elements and strides
 ** Then
 *** Checks for multiplication overflow to prevent size_t 
 *** overflow when calculating total memory size
 ** Then 
 *** Allocates aligned memory for the tensor data (tensor->data)
 ** Finally
 *** Returns a pointer to the created tensor
 *
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * Tensor* my_tensor = zeros(shape, 3, FLOAT32);
 ** @endcode
 **
 ** @note
 * The function performs comprehensive error checking including:
 ** - NULL pointer validation
 ** - Parameter range validation  
 ** - Memory allocation failure handling
 ** - Multiplication overflow protection
 ** - Type safety checks
 **
 ** @note
 * The function is almost the same as empty except 
 * that it fills the memory with zeros and is made 
 * separately for performance reasons, as it is not 
 * efficient to create an empty tensor and then fill 
 * it with zeros
 *
 ** @note
 * In debug mode, it can output additional information to the console
 *
 ** @note
 * Сan perform additional checks at a high level of safety
 *
 ** @warning
 * Do not forget to free the memory allocated for the tensor using free_tensor after using it
 *
 ** @see free_tensor
 ** @see get_dtype_size
 ** @see product
 ** @see ALLOC_ALIGNED
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 ** @see NNL2_STRIDES_LAST_DIM_INDEX
 ** @see NNL2_STRIDES_SECOND_LAST_DIM_INDEX
 ** @see NNL2_STRIDES_NEXT_DIM_INDEX
 *
 ** @exception[out_of_memory] 
 * If memory allocation fails for tensor structure, shape array, or data
 *
 ** @exception[invalid_argument]
 * If shape is NULL, rank <= 0, invalid dtype, or any dimension <= 0
 *
 ** @exception[overflow_error]
 * If the calculated total size would overflow size_t
 *
 **/
Tensor* nnl2_naive_zeros(const int* shape, int rank, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif
	
	// Checks the input parameters for correctness
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (shape == NULL) {
			NNL2_ERROR("Bad shape pointer");
			return NULL;
		}	
	#endif	
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (rank <= MIN_TENSOR_DIMENSION) {
			NNL2_ERROR("Bad rank (%d). Rank must be positive", rank);
			return NULL;
		}
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (dtype < 0 || dtype >= NUM_TENSOR_TYPES) {
			NNL2_ERROR("Invalid tensor data type (%d). Must be in range [0, %d]", dtype, NUM_TENSOR_TYPES - 1);
			return NULL;
		}
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		for (int i = 0; i < rank; i++) {
			if (shape[i] <= 0) {
				NNL2_ERROR("Bad shape dimension at %d (%d). Dimensions must be positive",  i, shape[i]);
				return NULL;
			}
		}
	#endif
	
	// Allocating memory for tensor structure

	Tensor* tensor = malloc(sizeof(Tensor));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor == NULL) {
			NNL2_ERROR("Failed to allocate memory for Tensor structure");
			return NULL;
		}
	#endif
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	
	// Allocates memory for the shape array and copies the data into it
	 
	tensor->shape = malloc(rank * sizeof(int));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor->shape == NULL) {
			NNL2_ERROR("Failed to allocate memory for shape");
			free(tensor);
			return NULL;
		}
	#endif
	
	size_t total_elems = 1; // Initializing total_elems
	memcpy(tensor->shape, shape, rank * sizeof(int));
	
	// Calculates the total size of the data required for the tensor
	
	tensor->strides = malloc(rank);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor->strides == NULL) {
			NNL2_ERROR("Failed to allocate memory for strides");
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif
	
	// Initializes the step for the last dimension
	// Stride for the innermost (last) dimension is 1
	tensor->strides[rank - NNL2_STRIDES_LAST_DIM_INDEX] = 1;
	total_elems *= shape[rank - NNL2_STRIDES_LAST_DIM_INDEX];
	
	// Going from the second-to-last dimension to the first (from left to right in shape)
	for(int i = rank - NNL2_STRIDES_SECOND_LAST_DIM_INDEX; i >= 0; i--) {
		tensor->strides[i] = tensor->strides[i + NNL2_STRIDES_NEXT_DIM_INDEX] * shape[i + NNL2_STRIDES_NEXT_DIM_INDEX];
		total_elems *= shape[i];
	}
	
	tensor->numel = total_elems;
	
	size_t type_size = get_dtype_size(dtype);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		// Check for multiplication overflow
		if (type_size != 0 && total_elems > SIZE_MAX / type_size) {
			NNL2_ERROR("Tensor size too large, multiplication overflow (elems: %zu, type size: %zu)", total_elems, type_size);
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif
	
	size_t total_size = total_elems * type_size;
	
	// Allocates aligned memory for tensor data (tensor->data)
	 
	void* data;
	
	ALLOC_ALIGNED(data, (size_t)TENSOR_MEM_ALIGNMENT, total_size);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (data == NULL) {
			NNL2_ERROR("Failed to allocate aligned memory");
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif
	
	tensor->data = data;
	memset(tensor->data, 0, total_size);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Created zeros tensor: %dD, elems=%zu, size=%zu bytes", rank, total_elems, total_size);
		#endif
	
		NNL2_FUNC_EXIT();
	#endif
	
	return tensor;
}

/** @ingroup backend_system
 ** @brief Backend implementations for zeros tensor creation
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_zeros: Basic reference implementation
 *
 ** @see REGISTER_BACKEND
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_naive_empty
 ** @see nnl2_naive
 **/
Implementation nnl2_zeros_backends[] = {
	REGISTER_BACKEND(nnl2_naive_zeros, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for the active zeros() backend 
 * @ingroup backend_system 
 */
fn_zeros nnl2_zeros;

/** 
 * @brief Sets the backend for zeros tensor creation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 */
void set_zeros_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nnl2_zeros_backends, nnl2_zeros, backend_name);
}

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(nnl2_zeros);

/** 
 * @brief Gets the name of the active backend for zeros()
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* nnl2_get_zeros_backend() {
	return CURRENT_BACKEND(nnl2_zeros);
}

/** 
 * @brief Function declaration for getting all `zeros` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(nnl2_zeros);

/**
 * @brief Function declaration for getting the number of all `zeros` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(nnl2_zeros);

/** @brief
 * Frees the memory allocated for the tensor.
 *
 ** @param tensor 
 * A pointer to a tensor whose memory needs to be freed. If tensor is null, the function does nothing.
 *
 ** @note
 * After calling this function tensor pointer becomes invalid. 
 * Do not attempt to access the tensor after it has been freed (although you'll try it anyway, you idiot).
 *
 ** @note 
 * Additional checks are added depending on the safety level
 *
 ** @see FREE_ALIGNED
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 **/
void nnl2_free_tensor(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
		if (tensor == NULL) {
			NNL2_ERROR("Invalid data (NULL) was passed to the tensor release");
			return;
		}
	#endif
	
	// Additional checks
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (tensor->rank < 0) {
			NNL2_ERROR("Invalid tensor rank: %d", tensor->rank);
		}
		
		if (tensor->shape == NULL && tensor->rank > 0) {
			NNL2_ERROR("NULL shape array with rank %d", tensor->rank);
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_DEBUG("Freeing tensor [%p] (rank: %d)", (void*)tensor, tensor->rank);
    #endif
	
	#if NNL2_SAFETY_MODE <= NNL2_SAFETY_MODE_MODERATE
		free(tensor->shape); 
		free(tensor->strides);
		FREE_ALIGNED(tensor->data);
	#else
		// Safe freeing
		if (tensor->shape != NULL) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Freeing shape array [%p]", (void*)tensor->shape);
			#endif
			
			free(tensor->shape);
		}
    
		if (tensor->strides != NULL) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Freeing strides [%p]", (void*)tensor->strides);
			#endif
			
			free(tensor->strides);
		}
	
		if (tensor->data != NULL) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Freeing data [%p]", (void*)tensor->data);
			#endif
			
			FREE_ALIGNED(tensor->data);
		}
	#endif
	
	free(tensor);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
}

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
	#endif
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
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
	#endif

	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
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

	// Calculate total elements in subtensor
	subtensor->numel = 1;
    for (int i = 0; i < subtensor->rank; i++) {
        subtensor->numel *= subtensor->shape[i];
    }

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
 ** @see nnl2_naive_tref_getter
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
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
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
    subtensor->numel = 1;
    for (int i = 0; i < subtensor->rank; i++) {
        subtensor->numel *= subtensor->shape[i];
    }

    // Allocate independent aligned memory for data and copy from source
    size_t data_size = subtensor->numel * element_size;
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
Implementation nnl2_tref_getter_backends[] = {
	REGISTER_BACKEND(nnl2_naive_tref_getter, nnl2_naive, NAIVE_BACKEND_NAME),
};

trefgetterfn nnl2_tref_getter;

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

void naive_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value;
			volatile int32_t* data = (int32_t*)tensor->data;	
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value;
			volatile float* data = (float*)tensor->data;
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value;
			volatile double* data = (double*)tensor->data;
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
	}
}

#ifdef __AVX__
void avx_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if (total_elems == 0) return;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value;
			int32_t* data = (int32_t*)tensor->data;
			
			__m256i avx_filler = _mm256_set1_epi32(filler);
				
			size_t avx_iters = total_elems / 8; 
			for (size_t i = 0; i < avx_iters; i++) { 
				_mm256_storeu_si256((__m256i*)(data + i * 8), avx_filler);
			}

			for (size_t j = avx_iters * 8; j < total_elems; j++) {
				data[j] = filler;
			}	
				
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value;
			float* data = (float*)tensor->data;
			
			__m256 avx_filler = _mm256_set1_ps(filler);
			
			size_t avx_iters = total_elems / 8; 
			for (size_t i = 0; i < avx_iters; i++) {
				_mm256_storeu_ps(data + i * 8, avx_filler);
			}

			for (size_t j = avx_iters * 8; j < total_elems; j++) {
				data[j] = filler;
			}
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value;
			double* data = (double*)tensor->data;
			
			__m256d avx_filler = _mm256_set1_pd(filler);
			
			size_t it = 0;
			
			for(; it < total_elems - 3; it += 4) _mm256_storeu_pd(data + it, avx_filler);
			for(size_t j = it; j < total_elems; j++) data[j] = filler;
			
			break;
		}
	}
}
#endif

Implementation inplace_fill_backends[] = {
	REGISTER_BACKEND(naive_inplace_fill, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
	REGISTER_BACKEND(avx_inplace_fill, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
};

fn_inplace_fill inplace_fill;

void set_inplace_fill_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(inplace_fill_backends, inplace_fill, backend_name);
}

make_current_backend(inplace_fill);

const char* get_inplace_fill_backend() {
	return current_backend(inplace_fill);
}

DEFINE_GET_BACKENDS_FUNCTION(inplace_fill);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(inplace_fill);

Tensor* ones(const int* shape, int rank, TensorType dtype) {
    Tensor* tensor_t = nnl2_empty(shape, rank, dtype);

    switch(dtype) {
        case INT32: {
            int32_t filler = 1;
            inplace_fill(tensor_t, &filler, dtype);
            break;
        }
        
        case FLOAT32: {		
            float filler = 1.0;
            inplace_fill(tensor_t, &filler, dtype);
            break;
        }
        
        case FLOAT64: {
            double filler = 1.0;
            inplace_fill(tensor_t, &filler, dtype);
            break;
        }
        
        default: {
            fprintf(stderr, "Error (Hello from C!): Unknown data type");
            nnl2_free_tensor(tensor_t);  
            return NULL;
        }
    }
    
    return tensor_t;
}

Tensor* full(const int* shape, int rank, TensorType dtype, void* filler) {
	Tensor* tensor_t = nnl2_empty(shape, rank, dtype);
	inplace_fill(tensor_t, filler, dtype);
	return tensor_t;
}

void naive_sgemminplace(const nnl2_order order, const nnl2_transpose transa, 
                        const nnl2_transpose transb, const int m, const int n, 
                        const int k, const float alpha, const Tensor* a, const int lda,
                        const Tensor* b, const int ldb, const float beta, Tensor* c,
                        const int ldc) {

    if (!a || !b || !c || !a->data || !b->data || !c->data) {
        fprintf(stderr, "Error (Hello from C!): Null pointer passed as argument (matmul)");
        return;
    }
    
    if (m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        fprintf(stderr, "Error (Hello from C!): Invalid dimensions provided (matmul)");
        return;
    }
    
    int a_cols = (transa == nnl2Trans) ? m : k;
    int b_cols = (transb == nnl2Trans) ? k : n;
    
    if (lda < a_cols) {
        fprintf(stderr, "Error (Hello from C!): lda is less than number of columns of a matrix! (matmul)");
        return;
    }
    
    if (ldb < b_cols) {  
        fprintf(stderr, "Error (Hello from C!): ldb is less than number of columns of b matrix! (matmul)");
        return;
    }

    if (ldc < n) {    
        fprintf(stderr, "Error (Hello from C!): ldc is less than n! (matmul)");
        return;
    }

    volatile float* data_a = (volatile float*)a->data;
    volatile float* data_b = (volatile float*)b->data;
    volatile float* data_c = (volatile float*)c->data;                          
    
    if(order == nnl2RowMajor){
        for(volatile int i = 0; i < m; i++) {
            for(volatile int j = 0; j < n; j++) {    
                float acc = 0.0;
        
                for(volatile int l = 0; l < k; l++) {
                    float a_val; 
                    float b_val; 
                
                    if (transa == nnl2Trans) {
                        a_val = *(data_a + l * lda + i);
                    } else {
                        a_val = *(data_a + i * lda + l);
                    }
                    
                    if (transb == nnl2Trans) {
                        b_val = *(data_b + j * ldb + l);
                    } else {
                        b_val = *(data_b + l * ldb + j);
                    }
                
                    acc += a_val * b_val;
                }
            
                if(beta == 0) {
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    } else {
        for(volatile int i = 0; i < m; i++) {
            for(volatile int j = 0; j < n; j++) {
                float acc = 0.0;
                
                for(volatile int l = 0; l < k; l++) {
                    float a_val;
                    float b_val;
                
                    if (transa == nnl2Trans) {
                        a_val = *(data_a + i * lda + l);
                    } else {
                        a_val = *(data_a + l * lda + i);
                    }
                    
                    if (transb == nnl2Trans) {
                        b_val = *(data_b + l * ldb + j);
                    } else {
                        b_val = *(data_b + j * ldb + l);
                    }
                
                    acc += a_val * b_val;
                }
                
                if(beta == 0) {
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    }
}

#ifdef OPENBLAS_AVAILABLE
void blas_sgemminplace(const nnl2_order order, const nnl2_transpose transa, 
                       const nnl2_transpose transb, const int m, const int n, 
                       const int k, const float alpha, const Tensor* a, const int lda,
                       const Tensor* b, const int ldb, const float beta, Tensor* c,
                       const int ldc) {

	float* a_data = (float*)a->data;
	float* b_data = (float*)b->data;
	float* c_data = (float*)c->data;
	
	CBLAS_ORDER cblas_order;
	
	switch(order) {
		case nnl2RowMajor:
			cblas_order = CblasRowMajor;
			break;
			
		case nnl2ColMajor:
			cblas_order = CblasColMajor;
			break;
			
		default: {
			fprintf(stderr, "Error (Hello from C!): Unknown order (matmul)\n");
			return;
		}
	}
	
	CBLAS_TRANSPOSE cblas_transa;
	CBLAS_TRANSPOSE cblas_transb;
	
	switch(transa) {
		case nnl2NoTrans:
			cblas_transa = CblasNoTrans;
			break;
			
		case nnl2Trans:
			cblas_transa = CblasTrans;
			break;
			
		default: {
			fprintf(stderr, "Error (Hello from C!): Unknown trans (a matrix) (matmul)\n");
			return;
		}
	}
	
	switch(transb) {
		case nnl2NoTrans:
			cblas_transb = CblasNoTrans;
			break;
			
		case nnl2Trans:
			cblas_transb = CblasTrans;
			break;
			
		default: {
			fprintf(stderr, "Error (Hello from C!): Unknown trans (b matrix) (matmul)\n");
			return;
		}
	}
						   
	cblas_sgemm(cblas_order, cblas_transa, cblas_transb, m, n, k, alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
}
#endif

Implementation sgemminplace_backends[] = {	
	REGISTER_BACKEND(naive_sgemminplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef OPENBLAS_AVAILABLE
	REGISTER_BACKEND(blas_sgemminplace, nnl2_blas, BLAS_BACKEND_NAME),
	#endif
};

sgemminplacefn sgemminplace;

void set_sgemminplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sgemminplace_backends, sgemminplace, backend_name);
}

void naive_dgemminplace(const nnl2_order order, const nnl2_transpose transa,
                        const nnl2_transpose transb, const int m, const int n,
                        const int k, const double alpha, const Tensor* a, const int lda,
                        const Tensor* b, const int ldb, const double beta, Tensor* c,
                        const int ldc) {						

    if (!a || !b || !c || !a->data || !b->data || !c->data) {
        fprintf(stderr, "Error (Hello from C!): Null pointer passed as argument (matmul)");
        return;
    }

    if (m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
        fprintf(stderr, "Error (Hello from C!): Invalid dimensions provided (matmul)");
        return;
    }

    int a_cols = (transa == nnl2Trans) ? m : k;
    int b_cols = (transb == nnl2Trans) ? k : n;

    if (lda < a_cols) {
        fprintf(stderr, "Error (Hello from C!): lda is less than number of columns of a matrix! (matmul)");
        return;
    }

    if (ldb < b_cols) {
        fprintf(stderr, "Error (Hello from C!): ldb is less than number of columns of b matrix! (matmul)");
        return;
    }

    if (ldc < n) {
        fprintf(stderr, "Error (Hello from C!): ldc is less than n! (matmul)");
        return;
    }

    volatile double* data_a = (volatile double*)a->data;
    volatile double* data_b = (volatile double*)b->data;
    volatile double* data_c = (volatile double*)c->data;

    if(order == nnl2RowMajor){
        for(volatile int i = 0; i < m; i++) {
            for(volatile int j = 0; j < n; j++) {
                double acc = 0.0;

                for(volatile int l = 0; l < k; l++) {
                    double a_val;
                    double b_val;

                    if (transa == nnl2Trans) {
                        a_val = *(data_a + l * lda + i);
                    } else {
                        a_val = *(data_a + i * lda + l);
                    }

                    if (transb == nnl2Trans) {
                        b_val = *(data_b + j * ldb + l);
                    } else {
                        b_val = *(data_b + l * ldb + j);
                    }

                    acc += a_val * b_val;
                }

                if(beta == 0) {
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    } else {
        for(volatile int i = 0; i < m; i++) {
            for(volatile int j = 0; j < n; j++) {
                double acc = 0.0;

                for(volatile int l = 0; l < k; l++) {
                    double a_val;
                    double b_val;

                    if (transa == nnl2Trans) {
                        a_val = *(data_a + i * lda + l);
                    } else {
                        a_val = *(data_a + l * lda + i);
                    }

                    if (transb == nnl2Trans) {
                        b_val = *(data_b + l * ldb + j);
                    } else {
                        b_val = *(data_b + j * ldb + l);
                    }

                    acc += a_val * b_val;
                }

                if(beta == 0) {
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    }
}

#ifdef OPENBLAS_AVAILABLE
void blas_dgemminplace(const nnl2_order order, const nnl2_transpose transa, 
                       const nnl2_transpose transb, const int m, const int n, 
                       const int k, const double alpha, const Tensor* a, const int lda,
                       const Tensor* b, const int ldb, const double beta, Tensor* c,
                       const int ldc) {

	double* a_data = (double*)a->data;
	double* b_data = (double*)b->data;
	double* c_data = (double*)c->data;
	
	CBLAS_ORDER cblas_order;
	
	switch(order) {
		case nnl2RowMajor:
			cblas_order = CblasRowMajor;
			break;
			
		case nnl2ColMajor:
			cblas_order = CblasColMajor;
			break;
			
		default: {
			fprintf(stderr, "Error (Hello from C!): Unknown order (matmul)\n");
			return;
		}
	}
	
	CBLAS_TRANSPOSE cblas_transa;
	CBLAS_TRANSPOSE cblas_transb;
	
	switch(transa) {
		case nnl2NoTrans:
			cblas_transa = CblasNoTrans;
			break;
			
		case nnl2Trans:
			cblas_transa = CblasTrans;
			break;
			
		default: {
			fprintf(stderr, "Error (Hello from C!): Unknown trans (a matrix) (matmul)\n");
			return;
		}
	}
	
	switch(transb) {
		case nnl2NoTrans:
			cblas_transb = CblasNoTrans;
			break;
			
		case nnl2Trans:
			cblas_transb = CblasTrans;
			break;
			
		default: {
			fprintf(stderr, "Error (Hello from C!): Unknown trans (b matrix) (matmul)\n");
			return;
		}
	}
						   
	cblas_dgemm(cblas_order, cblas_transa, cblas_transb, m, n, k, alpha, a_data, lda, b_data, ldb, beta, c_data, ldc);
}
#endif

Implementation dgemminplace_backends[] = {
	REGISTER_BACKEND(naive_dgemminplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef OPENBLAS_AVAILABLE
	REGISTER_BACKEND(blas_dgemminplace, nnl2_blas, BLAS_BACKEND_NAME),
	#endif
};

dgemminplacefn dgemminplace;
make_current_backend(gemm);

void set_dgemminplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(dgemminplace_backends, dgemminplace, backend_name, current_backend(gemm));
}

const char* get_gemm_backend() {
	return current_backend(gemm);
}

DEFINE_GET_BACKENDS_FUNCTION(dgemminplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(dgemminplace);

Tensor* sgemm(const nnl2_order order, const nnl2_transpose transa, 
			  const nnl2_transpose transb, const int m, const int n, 
			  const int k, const float alpha, const Tensor* a, const int lda,
			  const Tensor* b, const int ldb, const float beta) {
	
	int shape_c[] = {m, n};
	int rank_c = 2;
	TensorType type_c = FLOAT32;
	
	Tensor* c = ones(shape_c, rank_c, type_c);
	
	sgemminplace(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, n);
	
	return c;
}

Tensor* dgemm(const nnl2_order order, const nnl2_transpose transa, 
			  const nnl2_transpose transb, const int m, const int n, 
			  const int k, const double alpha, const Tensor* a, const int lda,
			  const Tensor* b, const int ldb, const double beta) {
	
	int shape_c[] = {m, n};
	int rank_c = 2;
	TensorType type_c = FLOAT64;
	
	Tensor* c = ones(shape_c, rank_c, type_c);
	
	dgemminplace(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, n);
	
	return c;
}

Tensor* gemm(const nnl2_order order, const nnl2_transpose transa, 
			 const nnl2_transpose transb, const int m, const int n, 
		     const int k, const double alpha, const Tensor* a, const int lda,
			 const Tensor* b, const int ldb, const double beta) {
				
	TensorType dtype = a->dtype;
	
	switch(dtype) {
		case FLOAT64: return dgemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta);
		case FLOAT32: return sgemm(order, transa, transb, m, n, k, (const float)alpha, a, lda, b, ldb, (const float)beta);
		
		default: {
			fprintf(stderr, "Unsupported data type!");
			return NULL;
		}
	}
}

void gemminplace(const nnl2_order order, const nnl2_transpose transa, 
					const nnl2_transpose transb, const int m, const int n, 
					const int k, const double alpha, const Tensor* a, const int lda,
					const Tensor* b, const int ldb, const double beta,
					Tensor* c, const int ldc) {

	TensorType dtype = a->dtype;
	
	switch(dtype) {
		case FLOAT64:
			dgemminplace(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
			break;
			
		case FLOAT32: 
			sgemminplace(order, transa, transb, m, n, k, (const float)alpha, a, lda, b, ldb, (const float)beta, c, ldc);
			break;
		
		default: {
			fprintf(stderr, "Unsupported data type!");
			return;
		}
	}			
}

void print_1d_tensor(Tensor* tensor) {		
	if (tensor == NULL) return;
	
	printf("#<NNL2:TENSOR/");
	
	int rows = tensor->shape[0];
	TensorType dtype_tensor = tensor->dtype;
	
	char* type_name = get_tensortype_name(dtype_tensor);
	
	printf("%s [%d]:", type_name, rows);
	
	switch(dtype_tensor) {
		case FLOAT64: {
			double* data_t = (double*)tensor->data;
			for(int i = 0; i < rows; i++) printf("\n    %f", data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* data_t = (float*)tensor->data;
			for(int i = 0; i < rows; i++) printf("\n    %f", data_t[i]);
			break;
		}
		
		case INT32: {
			int32_t* data_t = (int32_t*)tensor->data;
			for(int i = 0; i < rows; i++) printf("\n    %d", data_t[i]);
			break;
		}
		
		default: {
			fprintf(stderr, "UNKNOWN DATA TYPE: %s\n", type_name);
			break;
		}
	}
	
	printf(">\n");
}

void print_2d_tensor(Tensor* tensor) {
	if (tensor == NULL) return;
	
	printf("#<NNL2:TENSOR/");
	
	int rows = tensor->shape[0];
	int cols = tensor->shape[1];
	
	TensorType dtype_tensor = tensor->dtype;
	
	char* type_name = get_tensortype_name(dtype_tensor);
	
	printf("%s [%dx%d]:", type_name, rows, cols);
	
	switch(dtype_tensor) {
		case FLOAT64: {
			for(int i = 0; i < rows; i++) {
				printf("\n");		
				
				for(int j = 0; j < cols; j++) {
					int index = (i * cols) + j;
					double* data_t = (double*)tensor->data;
					printf("     %f", data_t[index]);
				}
			}
			
			break;
		}
		
		case FLOAT32: {
			for(int i = 0; i < rows; i++) {
				printf("\n");
				
				for(int j = 0; j < cols; j++) {
					int index = (i * cols) + j;
					float* data_t = (float*)tensor->data;
					printf("     %f", data_t[index]);
				}
			}
			
			break;
		}
		
		case INT32: {
			for(int i = 0; i < rows; i++) {
				printf("\n");
				
				for(int j = 0; j < cols; j++) {
					int index = (i * cols) + j;
					int32_t* data_t = (int32_t*)tensor->data;
					printf("     %d", data_t[index]);
				}
			}
			
			break;
		}
	}
	
	printf(">\n");
}

void print_huge_tensor(Tensor* tensor) {
	if (tensor == NULL) return;
	
	printf("#<NNL2:TENSOR/");
	
	TensorType dtype_tensor = tensor->dtype;
    char* type_name = get_tensortype_name(dtype_tensor);
	
	printf("%s [", type_name);
	
	if (tensor->rank > 0) {
        printf("%d", tensor->shape[0]);
        for (int i = 1; i < tensor->rank; i++) {
            printf("x%d", tensor->shape[i]);
        }
    }
	
    printf("]>");
}

void print_tensor(Tensor* tensor) {
	int rank = tensor->rank;
	
	if(rank <= 0)      {return;}
	else if(rank == 1) {print_1d_tensor(tensor);}
	else if(rank == 2) {print_2d_tensor(tensor);}
	else 			   {print_huge_tensor(tensor);}
}

int get_tensor_rank(Tensor* tensor) {
	return tensor->rank;
}

TensorType get_tensor_dtype(Tensor* tensor) {
	return tensor->dtype;
}

int* get_tensor_shape(Tensor* tensor) {
	return tensor->shape;
}

void naive_addinplace(Tensor* summand, const Tensor* addend) {
	size_t len = product(summand->shape, summand->rank);
	
	TensorType dtype_summand = summand->dtype;
	TensorType dtype_addend = addend->dtype;
	
	if(dtype_summand != dtype_addend) {
		fprintf(stderr, "Error (Hello from C!): In add (in-place) data-types are other\n");
		return;
	}
	
	switch(dtype_summand) {
		case FLOAT64: {
			volatile double* data_summand = (double*)summand->data;
			volatile double* data_addend = (double*)addend->data;
	
			for(size_t i = 0; i < len; i++) {
				data_summand[i] += data_addend[i];
			}
			
			break;
		}
		
		case FLOAT32: {
			volatile float* data_summand = (float*)summand->data;
			volatile float* data_addend = (float*)addend->data;
	
			for(size_t i = 0; i < len; i++) {
				data_summand[i] += data_addend[i];
			}
			
			break;
		}
		
		case INT32: {
			volatile int32_t* data_summand = (int32_t*)summand->data;
			volatile int32_t* data_addend = (int32_t*)addend->data;
	
			for(size_t i = 0; i < len; i++) {
				data_summand[i] += data_addend[i];
			}
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data type (add in-place)");
			return;
		}
	}
}

#ifdef __AVX__
void avx_addinplace(Tensor* summand, const Tensor* addend) {
    size_t len = product(summand->shape, summand->rank);

    TensorType dtype_summand = summand->dtype;
    TensorType dtype_addend = addend->dtype;

    if (dtype_summand != dtype_addend) {
        fprintf(stderr, "Error (Hello from C!): In add (in-place) data-types are other\n");
        return;
    }
	
	switch (dtype_summand) {
        case FLOAT64: {
            double* data_summand = (double*)summand->data;
            double* data_addend = (double*)addend->data;

            size_t i = 0;
			
            for(; i + 3 < len; i += 4) {
                __m256d v_summand = _mm256_loadu_pd(&data_summand[i]);
                __m256d v_addend = _mm256_loadu_pd(&data_addend[i]);
				
                __m256d v_result = _mm256_add_pd(v_summand, v_addend);
				
                _mm256_storeu_pd(&data_summand[i], v_result);
            }
			
			for(; i < len; i++) data_summand[i] += data_addend[i];
			
			break;
		}
		
		case FLOAT32: {
            float* data_summand = (float*)summand->data;
            float* data_addend = (float*)addend->data;

            size_t i = 0;
			
            for(; i + 7 < len; i += 8) {
                __m256 v_summand = _mm256_loadu_ps(&data_summand[i]);
                __m256 v_addend = _mm256_loadu_ps(&data_addend[i]);
				
                __m256 v_result = _mm256_add_ps(v_summand, v_addend);
				
                _mm256_storeu_ps(&data_summand[i], v_result);
            }

            for(; i < len; i++) data_summand[i] += data_addend[i];

            break;
        }
		
		case INT32: {
            int32_t* data_summand = (int32_t*)summand->data;
            int32_t* data_addend = (int32_t*)addend->data;

            size_t i = 0;
			
            for(; i + 7 < len; i += 8) {
                __m256i v_summand = _mm256_loadu_si256((__m256i*)&data_summand[i]);
                __m256i v_addend = _mm256_loadu_si256((__m256i*)&data_addend[i]);
				
                __m256i v_result = _mm256_add_epi32(v_summand, v_addend);
				
                _mm256_storeu_si256((__m256i*)&data_summand[i], v_result);
            }

            for(; i < len; i++) data_summand[i] += data_addend[i];

            break;
        }
		
		default: {
            fprintf(stderr, "Error (Hello from C!): Bad data type (add in-place)");
            return;
        }
	}
}
#endif

addinplacefn addinplace;
make_current_backend(addinplace);

Implementation addinplace_backends[] = {
	REGISTER_BACKEND(naive_addinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
	REGISTER_BACKEND(avx_addinplace, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
};

void set_addinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(addinplace_backends, addinplace, backend_name, current_backend(addinplace));
}

const char* get_addinplace_backend() {
	return current_backend(addinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(addinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(addinplace);

void naive_subinplace(Tensor* summand, const Tensor* addend) {
	size_t len = product(summand->shape, summand->rank);
	
	TensorType dtype_summand = summand->dtype;
	TensorType dtype_addend = addend->dtype;
	
	if(dtype_summand != dtype_addend) {
		fprintf(stderr, "Error (Hello from C!): In sub (in-place) data-types are other\n");
		return;
	}
	
	switch(dtype_summand) {
		case FLOAT64: {
			volatile double* data_summand = (double*)summand->data;
			volatile double* data_addend = (double*)addend->data;
	
			for(size_t i = 0; i < len; i++) {
				data_summand[i] -= data_addend[i];
			}
			
			break;
		}
		
		case FLOAT32: {
			volatile float* data_summand = (float*)summand->data;
			volatile float* data_addend = (float*)addend->data;
	
			for(size_t i = 0; i < len; i++) {
				data_summand[i] -= data_addend[i];
			}
			
			break;
		}
		
		case INT32: {
			volatile int32_t* data_summand = (int32_t*)summand->data;
			volatile int32_t* data_addend = (int32_t*)addend->data;
	
			for(size_t i = 0; i < len; i++) {
				data_summand[i] -= data_addend[i];
			}
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data type (sub in-place)");
			return;
		}
	}
}

#ifdef __AVX__
void avx_subinplace(Tensor* minuend, const Tensor* subtrahend) {
    size_t len = product(minuend->shape, minuend->rank);

    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;

    if (dtype_minuend != dtype_subtrahend) {
        fprintf(stderr, "Error (Hello from C!): In sub (in-place) data-types are other\n");
        return;
    }

    switch (dtype_minuend) {
        case FLOAT64: {
            double* data_minuend = (double*)minuend->data;
            double* data_subtrahend = (double*)subtrahend->data;

            size_t i = 0;

            for(; i + 3 < len; i += 4) {
                __m256d v_minuend = _mm256_loadu_pd(&data_minuend[i]);
                __m256d v_subtrahend = _mm256_loadu_pd(&data_subtrahend[i]);

                __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);

                _mm256_storeu_pd(&data_minuend[i], v_result);
            }

            for(; i < len; i++) data_minuend[i] -= data_subtrahend[i];

            break;
        }

        case FLOAT32: {
            float* data_minuend = (float*)minuend->data;
            float* data_subtrahend = (float*)subtrahend->data;

            size_t i = 0;

            for(; i + 7 < len; i += 8) {
                __m256 v_minuend = _mm256_loadu_ps(&data_minuend[i]);
                __m256 v_subtrahend = _mm256_loadu_ps(&data_subtrahend[i]);

                __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);

                _mm256_storeu_ps(&data_minuend[i], v_result);
            }

            for(; i < len; i++) data_minuend[i] -= data_subtrahend[i];

            break;
        }

        case INT32: {
            int32_t* data_minuend = (int32_t*)minuend->data;
            int32_t* data_subtrahend = (int32_t*)subtrahend->data;

            size_t i = 0;

            for(; i + 7 < len; i += 8) {
                __m256i v_minuend = _mm256_loadu_si256((__m256i*)&data_minuend[i]);
                __m256i v_subtrahend = _mm256_loadu_si256((__m256i*)&data_subtrahend[i]);

                __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);

                _mm256_storeu_si256((__m256i*)&data_minuend[i], v_result);
            }

            for(; i < len; i++) data_minuend[i] -= data_subtrahend[i];

            break;
        }

        default: {
            fprintf(stderr, "Error (Hello from C!): Bad data type (sub in-place)");
            return;
        }
    }
}
#endif

Implementation subinplace_backends[] = {
	REGISTER_BACKEND(naive_subinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
	REGISTER_BACKEND(avx_subinplace, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
};

subinplacefn subinplace;
make_current_backend(subinplace);

void set_subinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(subinplace_backends, subinplace, backend_name, current_backend(subinplace));
}

const char* get_subinplace_backend() {
	return current_backend(subinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(subinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(subinplace);

int get_mem_alignment() {
	return TENSOR_MEM_ALIGNMENT;
}

int get_size(Tensor* tensor) {
	return product(tensor->shape, tensor->rank);
}

int get_size_in_bytes(Tensor* tensor) {
	return product(tensor->shape, tensor->rank) * get_dtype_size(tensor->dtype);
}

Tensor* naive_add(const Tensor* summand, const Tensor* addend) {
	size_t len = product(summand->shape, summand->rank);
	
	TensorType dtype_summand = summand->dtype;
	TensorType dtype_addend = addend->dtype;
	
	if(dtype_summand != dtype_addend) {
		fprintf(stderr, "Error (Hello from C!): In sub (in-place) data-types are other\n");
		return NULL;
	}
	
	Tensor* amount = nnl2_zeros(summand->shape, summand->rank, dtype_summand);
	
	switch(dtype_summand) {
		case FLOAT64: {
			volatile double* data_summand = (double*)summand->data;
			volatile double* data_addend = (double*)addend->data;
			volatile double* data_amount = (double*)amount->data;
	
			for(size_t i = 0; i < len; i++) {
				data_amount[i] = data_summand[i] + data_addend[i];
			}
			
			break;
		}
		
		case FLOAT32: {
			volatile float* data_summand = (float*)summand->data;
			volatile float* data_addend = (float*)addend->data;
			volatile float* data_amount = (float*)amount->data;
	
			for(size_t i = 0; i < len; i++) {
				data_amount[i] = data_summand[i] + data_addend[i];
			}
			
			break;
		}
		
		case INT32: {
			volatile int32_t* data_summand = (int32_t*)summand->data;
			volatile int32_t* data_addend = (int32_t*)addend->data;
			volatile int32_t* data_amount = (int32_t*)amount->data;
	
			for(size_t i = 0; i < len; i++) {
				data_amount[i] = data_summand[i] + data_addend[i];
			}
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data type (sub in-place)");
			return NULL;
		}
	}
	
	return amount;
}

#ifdef __AVX__
Tensor* avx_add(const Tensor* summand, const Tensor* addend) {
    size_t len = product(summand->shape, summand->rank);
    
    TensorType dtype_summand = summand->dtype;
    TensorType dtype_addend = addend->dtype;
    
    if(dtype_summand != dtype_addend) {
        fprintf(stderr, "Error (Hello from C!): In add (in-place) data-types are other\n");
        return NULL;
    }
    
    Tensor* sum = nnl2_zeros(summand->shape, summand->rank, dtype_summand);
    
    switch(dtype_summand) {
        case FLOAT64: {
            double* data_summand = (double*)summand->data;
            double* data_addend = (double*)addend->data;
            double* data_sum = (double*)sum->data;

            size_t i = 0;
			
            for(; i + 3 < len; i += 4) {
                __m256d v_summand = _mm256_loadu_pd(&data_summand[i]);
                __m256d v_addend = _mm256_loadu_pd(&data_addend[i]);
				
                __m256d v_result = _mm256_add_pd(v_summand, v_addend);
				
                _mm256_storeu_pd(&data_sum[i], v_result);
            }
			
            for(; i < len; i++) {
                data_sum[i] = data_summand[i] + data_addend[i];
            }
			
            break;
        }
        
        case FLOAT32: {
            float* data_summand = (float*)summand->data;
            float* data_addend = (float*)addend->data;
            float* data_sum = (float*)sum->data;

            size_t i = 0;
			
            for(; i + 7 < len; i += 8) {
                __m256 v_summand = _mm256_loadu_ps(&data_summand[i]);
                __m256 v_addend = _mm256_loadu_ps(&data_addend[i]);
				
                __m256 v_result = _mm256_add_ps(v_summand, v_addend);
				
                _mm256_storeu_ps(&data_sum[i], v_result);
            }
			
            for(; i < len; i++) {
                data_sum[i] = data_summand[i] + data_addend[i];
            }
			
            break;
        }
        
        case INT32: {
            int32_t* data_summand = (int32_t*)summand->data;
            int32_t* data_addend = (int32_t*)addend->data;
            int32_t* data_sum = (int32_t*)sum->data;

            size_t i = 0;
            for(; i + 7 < len; i += 8) {
                __m256i v_summand = _mm256_loadu_si256((__m256i*)&data_summand[i]);
                __m256i v_addend = _mm256_loadu_si256((__m256i*)&data_addend[i]);
				
                __m256i v_result = _mm256_add_epi32(v_summand, v_addend);
				
                _mm256_storeu_si256((__m256i*)&data_sum[i], v_result);
            }
			
            for(; i < len; i++) {
                data_sum[i] = data_summand[i] + data_addend[i];
            }
			
            break;
        }
        
        default: {
            fprintf(stderr, "Error (Hello from C!): Bad data type (add in-place)");
            return NULL;
        }
    }
    
    return sum;
}
#endif

Implementation add_backends[] = {
	REGISTER_BACKEND(naive_add, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
	REGISTER_BACKEND(avx_add, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
};

addfn add;
make_current_backend(add);

void set_add_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(add_backends, add, backend_name, current_backend(add));
}

const char* get_add_backend() {
	return current_backend(add);
}

DEFINE_GET_BACKENDS_FUNCTION(add);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(add);

Tensor* naive_sub(const Tensor* minuend, const Tensor* subtrahend) {
	size_t len = product(minuend->shape, minuend->rank);
	
	TensorType dtype_minuend = minuend->dtype;
	TensorType dtype_subtrahend = subtrahend->dtype;
	
	if(dtype_minuend != dtype_subtrahend) {
		fprintf(stderr, "Error (Hello from C!): In sub (in-place) data-types are other\n");
		return NULL;
	}
	
	Tensor* difference = nnl2_zeros(minuend->shape, minuend->rank, dtype_minuend);
	
	switch(dtype_minuend) {
		case FLOAT64: {
			volatile double* data_minuend = (double*)minuend->data;
			volatile double* data_subtrahend = (double*)subtrahend->data;
			volatile double* data_difference = (double*)difference->data;
	
			for(size_t i = 0; i < len; i++) {
				data_difference[i] = data_minuend[i] - data_subtrahend[i];
			}
			
			break;
		}
		
		case FLOAT32: {
			volatile float* data_minuend = (float*)minuend->data;
			volatile float* data_subtrahend = (float*)subtrahend->data;
			volatile float* data_difference = (float*)difference->data;
	
			for(size_t i = 0; i < len; i++) {
				data_difference[i] = data_minuend[i] - data_subtrahend[i];
			}
			
			break;
		}
		
		case INT32: {
			volatile int32_t* data_minuend = (int32_t*)minuend->data;
			volatile int32_t* data_subtrahend = (int32_t*)subtrahend->data;
			volatile int32_t* data_difference = (int32_t*)difference->data;
	
			for(size_t i = 0; i < len; i++) {
				data_difference[i] = data_minuend[i] - data_subtrahend[i];
			}
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data type (sub in-place)");
			return NULL;
		}
	}
	
	return difference;
}

#ifdef __AVX__
Tensor* avx_sub(const Tensor* minuend, const Tensor* subtrahend) {
    size_t len = product(minuend->shape, minuend->rank);
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
    
    if(dtype_minuend != dtype_subtrahend) {
        fprintf(stderr, "Error (Hello from C!): In sub data-types are other\n");
        return NULL;
    }
    
    Tensor* difference = nnl2_zeros(minuend->shape, minuend->rank, dtype_minuend);
    
    switch(dtype_minuend) {
        case FLOAT64: {
            double* data_minuend = (double*)minuend->data;
            double* data_subtrahend = (double*)subtrahend->data;
            double* data_difference = (double*)difference->data;

            size_t i = 0;
			
            for(; i + 3 < len; i += 4) {
                __m256d v_minuend = _mm256_loadu_pd(&data_minuend[i]);
                __m256d v_subtrahend = _mm256_loadu_pd(&data_subtrahend[i]);
				
                __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
				
                _mm256_storeu_pd(&data_difference[i], v_result);
            }
			
            for(; i < len; i++) {
                data_difference[i] = data_minuend[i] - data_subtrahend[i];
            }
			
            break;
        }
        
        case FLOAT32: {
            float* data_minuend = (float*)minuend->data;
            float* data_subtrahend = (float*)subtrahend->data;
            float* data_difference = (float*)difference->data;

            size_t i = 0;
			
            for(; i + 7 < len; i += 8) {
                __m256 v_minuend = _mm256_loadu_ps(&data_minuend[i]);
                __m256 v_subtrahend = _mm256_loadu_ps(&data_subtrahend[i]);
				
                __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
				
                _mm256_storeu_ps(&data_difference[i], v_result);
            }
			
            for(; i < len; i++) {
                data_difference[i] = data_minuend[i] - data_subtrahend[i];
            }
			
            break;
        }
        
        case INT32: {
            int32_t* data_minuend = (int32_t*)minuend->data;
            int32_t* data_subtrahend = (int32_t*)subtrahend->data;
            int32_t* data_difference = (int32_t*)difference->data;

            size_t i = 0;
            for(; i + 7 < len; i += 8) {
                __m256i v_minuend = _mm256_loadu_si256((__m256i*)&data_minuend[i]);
                __m256i v_subtrahend = _mm256_loadu_si256((__m256i*)&data_subtrahend[i]);
				
                __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
				
                _mm256_storeu_si256((__m256i*)&data_difference[i], v_result);
            }
			
            for(; i < len; i++) {
                data_difference[i] = data_minuend[i] - data_subtrahend[i];
            }
			
            break;
        }
        
        default: {
            fprintf(stderr, "Error (Hello from C!): Bad data type (sub)");
            return NULL;
        }
    }
    
    return difference;
}
#endif

Implementation sub_backends[] = {
	REGISTER_BACKEND(naive_sub, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
	REGISTER_BACKEND(avx_sub, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
};

subfn sub;
make_current_backend(sub);

void set_sub_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sub_backends, sub, backend_name, current_backend(sub));
}

const char* get_sub_backend() {
	return current_backend(sub);
}

DEFINE_GET_BACKENDS_FUNCTION(sub);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sub);

void naive_mulinplace(Tensor* multiplicand, const Tensor* multiplier) {
	size_t len = product(multiplicand->shape, multiplicand->rank);
	
	TensorType dtype_multiplicand = multiplicand->dtype;
	TensorType dtype_multiplier = multiplier->dtype;
	
	if(dtype_multiplicand != dtype_multiplier) {
		fprintf(stderr, "Error (Hello from C!): In mul (in-place) data-types are other\n");
		return;
	}
	
	switch(dtype_multiplicand) {
		case FLOAT64: {
			volatile double* multiplicand_data = (double*)multiplicand->data;
			volatile double* multiplier_data = (double*)multiplier->data;
			
			for(size_t it = 0; it < len; it++) {
				multiplicand_data[it] *= multiplier_data[it];
			}
			
			break;
		}
		
		case FLOAT32: {
			volatile float* multiplicand_data = (float*)multiplicand->data;
			volatile float* multiplier_data = (float*)multiplier->data;
			
			for(size_t it = 0; it < len; it++) {
				multiplicand_data[it] *= multiplier_data[it];
			}
			
			break;
		}
		
		case INT32: {
			volatile int32_t* multiplicand_data = (int32_t*)multiplicand->data;
			volatile int32_t* multiplier_data = (int32_t*)multiplier->data;
			
			for(size_t it = 0; it < len; it++) {
				multiplicand_data[it] *= multiplier_data[it];
			}
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data type (mul in-place)");
			return;
		}
	}
}

Implementation mulinplace_backends[] = {
	REGISTER_BACKEND(naive_mulinplace, nnl2_naive, "NAIVE"),
};

mulinplacefn mulinplace;
make_current_backend(mulinplace);

void set_mulinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mulinplace_backends, mulinplace, backend_name, current_backend(mulinplace));
}

const char* get_mulinplace_backend() {
	return current_backend(mulinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(mulinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mulinplace);

void naive_divinplace(Tensor* dividend, const Tensor* divisor) {
	size_t len = product(dividend->shape, dividend->rank);
	
	TensorType dtype_dividend = dividend->dtype;
	TensorType dtype_divisor = divisor->dtype;
	
	if(dtype_dividend != dtype_divisor) {
		fprintf(stderr, "Error (Hello from C!): In div (in-place) data-types are other\n");
		return;
	}
	
	switch(dtype_dividend) {
		case FLOAT64: {
			volatile double* dividend_data = (double*)dividend->data;
			volatile double* divisor_data = (double*)divisor->data;
			
			for(size_t it = 0; it < len; it++) {
				dividend_data[it] /= divisor_data[it];
			}
			
			break;
		}
		
		case FLOAT32: {
			volatile float* dividend_data = (float*)dividend->data;
			volatile float* divisor_data = (float*)divisor->data;
			
			for(size_t it = 0; it < len; it++) {
				dividend_data[it] /= divisor_data[it];
			}
			
			break;
		}
		
		case INT32: {
			volatile int32_t* dividend_data = (int32_t*)dividend->data;
			volatile int32_t* divisor_data = (int32_t*)divisor->data;
			
			for(size_t it = 0; it < len; it++) {
				dividend_data[it] /= divisor_data[it];
			}
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data type (mul in-place)");
			return;
		}
	}
}

Implementation divinplace_backends[] = {
	REGISTER_BACKEND(naive_divinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

divinplacefn divinplace;
make_current_backend(divinplace);

void set_divinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(divinplace_backends, divinplace, backend_name, current_backend(divinplace));
}

const char* get_divinplace_backend() {
	return current_backend(divinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(divinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(divinplace);

Tensor* naive_mul(const Tensor* multiplicand, const Tensor* multiplier) {
    size_t len = product(multiplicand->shape, multiplicand->rank);
    
    TensorType dtype_multiplicand = multiplicand->dtype;
    TensorType dtype_multiplier = multiplier->dtype;
    
    if(dtype_multiplicand != dtype_multiplier) {
        fprintf(stderr, "Error (Hello from C!): In mul (in-place) data-types are other\n");
        return NULL;
    }
    
    Tensor* product = nnl2_zeros(multiplicand->shape, multiplicand->rank, dtype_multiplicand);
    
    switch(dtype_multiplicand) {
        case FLOAT64: {
            volatile double* data_multiplicand = (double*)multiplicand->data;
            volatile double* data_multiplier = (double*)multiplier->data;
            volatile double* data_product = (double*)product->data;
    
            for(size_t i = 0; i < len; i++) {
                data_product[i] = data_multiplicand[i] * data_multiplier[i];
            }
            
            break;
        }
        
        case FLOAT32: {
            volatile float* data_multiplicand = (float*)multiplicand->data;
            volatile float* data_multiplier = (float*)multiplier->data;
            volatile float* data_product = (float*)product->data;
    
            for(size_t i = 0; i < len; i++) {
                data_product[i] = data_multiplicand[i] * data_multiplier[i];
            }
            
            break;
        }
        
        case INT32: {
            volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
            volatile int32_t* data_multiplier = (int32_t*)multiplier->data;
            volatile int32_t* data_product = (int32_t*)product->data;
    
            for(size_t i = 0; i < len; i++) {
                data_product[i] = data_multiplicand[i] * data_multiplier[i];
            }
            
            break;
        }
        
        default: {
            fprintf(stderr, "Error (Hello from C!): Bad data type (mul in-place)");
            return NULL;
        }
    }
    
    return product;
}

Implementation mul_backends[] = {
	REGISTER_BACKEND(naive_mul, nnl2_naive, NAIVE_BACKEND_NAME),
};

mulfn mul;
make_current_backend(mul);

void set_mul_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mul_backends, mul, backend_name, current_backend(mul));
}

const char* get_mul_backend() {
	return current_backend(mul);
}

DEFINE_GET_BACKENDS_FUNCTION(mul);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mul);

Tensor* naive_div(const Tensor* dividend, const Tensor* divisor) {
    size_t len = product(dividend->shape, dividend->rank);
    
    TensorType dtype_dividend = dividend->dtype;
    TensorType dtype_divisor = divisor->dtype;
    
    if (dtype_dividend != dtype_divisor) {
        fprintf(stderr, "Error (Hello from C!): In div (in-place) data-types are different\n");
        return NULL;
    }
    
    Tensor* quotient = nnl2_zeros(dividend->shape, dividend->rank, dtype_dividend);
    
    switch (dtype_dividend) {
        case FLOAT64: {
            volatile double* data_dividend = (double*)dividend->data;
            volatile double* data_divisor = (double*)divisor->data;
            volatile double* data_quotient = (double*)quotient->data;
    
            for (size_t i = 0; i < len; i++) {			
                if (data_divisor[i] == 0.0) {
                    fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);	
                    return NULL;
                }
				
                data_quotient[i] = data_dividend[i] / data_divisor[i];
            }
			
            break;
        }
        
        case FLOAT32: {
            volatile float* data_dividend = (float*)dividend->data;
            volatile float* data_divisor = (float*)divisor->data;
            volatile float* data_quotient = (float*)quotient->data;
    
            for (size_t i = 0; i < len; i++) {
                if (data_divisor[i] == 0.0f) {
                    fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);
                    nnl2_free_tensor(quotient);
                    return NULL;
                }
				
                data_quotient[i] = data_dividend[i] / data_divisor[i];
            }
			
            break;
        }
        
        case INT32: {
            volatile int32_t* data_dividend = (int32_t*)dividend->data;
            volatile int32_t* data_divisor = (int32_t*)divisor->data;
            volatile int32_t* data_quotient = (int32_t*)quotient->data;
    
            for (size_t i = 0; i < len; i++) {
                if (data_divisor[i] == 0) {
                    fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);
                    return NULL;
                }
				
                data_quotient[i] = data_dividend[i] / data_divisor[i];
            }
			
            break;
        }
        
        default: {
            fprintf(stderr, "Error (Hello from C!): Bad data type (div)\n");
            nnl2_free_tensor(quotient);
            return NULL;
        }
    }
    
    return quotient;
}

Implementation div_backends[] = {
	REGISTER_BACKEND(naive_div, nnl2_naive, NAIVE_BACKEND_NAME),
};

divfn nnl2_div;
make_current_backend(div);

void set_div_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(div_backends, div, backend_name, current_backend(div));
}

const char* get_div_backend() {
	return current_backend(div);
}

DEFINE_GET_BACKENDS_FUNCTION(div);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(div);

void naive_powinplace(Tensor* base, const Tensor* exponent) {
	size_t len = product(base->shape, base->rank);
    
    TensorType dtype_base = base->dtype;
    TensorType dtype_exponent = exponent->dtype;
    
    if(dtype_base != dtype_exponent) {
        fprintf(stderr, "Error (Hello from C!): In pow (in-place) data-types are different\n");
        return;
    }
    
    switch(dtype_base) {
        case FLOAT64: {
            volatile double* base_data = (double*)base->data;
            volatile double* exponent_data = (double*)exponent->data;
            
            for(size_t it = 0; it < len; it++) {
                base_data[it] = pow(base_data[it], exponent_data[it]);
            }
            
            break;
        }
        
        case FLOAT32: {
            volatile float* base_data = (float*)base->data;
            volatile float* exponent_data = (float*)exponent->data;
            
            for(size_t it = 0; it < len; it++) {
                base_data[it] = powf(base_data[it], exponent_data[it]);
            }
            
            break;
        }
        
        case INT32: {
            volatile int32_t* base_data = (int32_t*)base->data;
            volatile int32_t* exponent_data = (int32_t*)exponent->data;
            
            for(size_t it = 0; it < len; it++) {
                base_data[it] = (int32_t)pow(base_data[it], exponent_data[it]);
            }
            
            break;
        }
        
        default: {
            fprintf(stderr, "Error (Hello from C!): Bad data type (pow in-place)");
            return;
        }
    }
}
	
Implementation powinplace_backends[] = {
	REGISTER_BACKEND(naive_powinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	
	
powinplacefn powinplace;
make_current_backend(powinplace);

void set_powinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(powinplace_backends, powinplace, backend_name, current_backend(powinplace));
}

const char* get_powinplace_backend() {
	return current_backend(powinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(powinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(powinplace);

Tensor* naive_pow(const Tensor* base, const Tensor* exponent) {
    size_t len = product(base->shape, base->rank);
    
    TensorType dtype_base = base->dtype;
    TensorType dtype_exponent = exponent->dtype;
    
    if(dtype_base != dtype_exponent) {
        fprintf(stderr, "Error (Hello from C!): In pow data-types are different\n");
        return NULL;
    }
    
    Tensor* result = nnl2_zeros(base->shape, base->rank, dtype_base);
    
    switch(dtype_base) {
        case FLOAT64: {
            volatile double* data_base = (double*)base->data;
            volatile double* data_exponent = (double*)exponent->data;
            volatile double* data_result = (double*)result->data;
    
            for(size_t i = 0; i < len; i++) {
                data_result[i] = pow(data_base[i], data_exponent[i]);
            }
            
            break;
        }
        
        case FLOAT32: {
            volatile float* data_base = (float*)base->data;
            volatile float* data_exponent = (float*)exponent->data;
            volatile float* data_result = (float*)result->data;
    
            for(size_t i = 0; i < len; i++) {
                data_result[i] = powf(data_base[i], data_exponent[i]);
            }
            
            break;
        }
        
        case INT32: {
            volatile int32_t* data_base = (int32_t*)base->data;
            volatile int32_t* data_exponent = (int32_t*)exponent->data;
            volatile int32_t* data_result = (int32_t*)result->data;
    
            for(size_t i = 0; i < len; i++) {
                data_result[i] = (int32_t)pow(data_base[i], data_exponent[i]);
            }
            
            break;
        }
        
        default: {
            fprintf(stderr, "Error (Hello from C!): Bad data type (pow)");
            return NULL;
        }
    }
    
    return result;
}

Implementation pow_backends[] = {
	REGISTER_BACKEND(naive_pow, nnl2_naive, NAIVE_BACKEND_NAME),
};	
	
powfn nnl2_pow;
make_current_backend(pow);

void set_pow_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(pow_backends, pow, backend_name, current_backend(pow));
}

const char* get_pow_backend() {
	return current_backend(pow);
}

DEFINE_GET_BACKENDS_FUNCTION(pow);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(pow);

void naive_expinplace(Tensor* tensor) {
	size_t len = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = exp(tensor_data[it]);
			break;
		}
			
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = expf(tensor_data[it]);
			break;	
		}
			
		case INT32: {
			volatile int32_t* tensor_data = (int32_t*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = (int32_t)exp((double)tensor_data[it]);
			break;	
		}
			
		default: {
			fprintf(stderr, "Error (Hello from C!): Unsupported data-type (exp in-place)");
			return;
		}
	}
}	

Implementation expinplace_backends[] = {
	REGISTER_BACKEND(naive_expinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

expinplacefn expinplace;
make_current_backend(expinplace);

void set_expinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(expinplace_backends, expinplace, backend_name, current_backend(expinplace));
}

const char* get_expinplace_backend() {
	return current_backend(expinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(expinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(expinplace);

Tensor* naive_exp(const Tensor* tensor) {
	size_t len = product(tensor->shape, tensor->rank);
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;
			volatile double* result_data = (double*)tensor->data;
			for(size_t it = 0; it < len; it++) result_data[it] = exp(tensor_data[it]);
			break;
		}
		
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;
			volatile float* result_data = (float*)tensor->data;
			for(size_t it = 0; it < len; it++) result_data[it] = expf(tensor_data[it]);
			break;
		}
		
		case INT32: {
			volatile int32_t* tensor_data = (int32_t*)tensor->data;
			volatile int32_t* result_data = (int32_t*)tensor->data;
			for(size_t it = 0; it < len; it++) result_data[it] = (int32_t)exp((double)tensor_data[it]);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Unsupported data-type (exp)");
			return NULL;
		}
	}
	
	return result;
}

Implementation exp_backends[] = {
	REGISTER_BACKEND(naive_exp, nnl2_naive, NAIVE_BACKEND_NAME),
};	

expfn nnl2_exp;
make_current_backend(exp);

void set_exp_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(exp_backends, exp, backend_name, current_backend(exp));
}

const char* get_exp_backend() {
	return current_backend(exp);
}

DEFINE_GET_BACKENDS_FUNCTION(exp);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(exp);

void naive_loginplace(Tensor* tensor) {
	size_t len = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = log(tensor_data[it]);
			break;
		}
			
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = logf(tensor_data[it]);
			break;	
		}
			
		case INT32: {
			volatile int32_t* tensor_data = (int32_t*)tensor->data;		
			for(size_t it = 0; it < len; it++) tensor_data[it] = (int32_t)log((double)tensor_data[it]);
			break;	
		}
			
		default: {
			fprintf(stderr, "Error (Hello from C!): Unsupported data-type (log in-place)");
			return;
		}
	}
}

Implementation loginplace_backends[] = {
	REGISTER_BACKEND(naive_loginplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

loginplacefn loginplace;
make_current_backend(loginplace);

void set_loginplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(loginplace_backends, loginplace, backend_name, current_backend(loginplace));
}

const char* get_loginplace_backend() {
	return current_backend(loginplace);
}

DEFINE_GET_BACKENDS_FUNCTION(loginplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(loginplace);

Tensor* naive_log(const Tensor* tensor) {
	size_t len = product(tensor->shape, tensor->rank);
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;
			volatile double* result_data = (double*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = log(tensor_data[it]);
			break;
		}
		
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;
			volatile float* result_data = (float*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = logf(tensor_data[it]);
			break;
		}
		
		case INT32: {
			volatile int32_t* tensor_data = (int32_t*)tensor->data;
			volatile int32_t* result_data = (int32_t*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = (int32_t)log((double)tensor_data[it]);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Unsupported data-type (log)");
			return NULL;
		}
	}
	
	return result;
}

Implementation log_backends[] = {
	REGISTER_BACKEND(naive_log, nnl2_naive, NAIVE_BACKEND_NAME),
};	

logfn nnl2_logarithm;
make_current_backend(log);

void set_log_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(log_backends, log, backend_name, current_backend(log));
}

const char* get_log_backend() {
	return current_backend(log);
}

DEFINE_GET_BACKENDS_FUNCTION(log);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(log);

void tensor_set_subtensor(Tensor* dest, int* dest_shape, int dest_rank, Tensor* src, int* src_shape, int src_rank);

void naive_tref_setter(Tensor* tensor, int* shape, int rank, void* change_with, bool is_tensor) {
    TensorType tensor_dtype = tensor->dtype;

    if (is_tensor) {
        Tensor* sub_tensor = (Tensor*)change_with;

        if (tensor->rank - rank != sub_tensor->rank) {
            fprintf(stderr, "Error (Hello from C!): Rank mismatch in tensor assignment\n");
            return;
        }
        
        for (int i = 0; i < sub_tensor->rank; i++) {
            if (tensor->shape[rank + i] != sub_tensor->shape[i]) {
                fprintf(stderr, "Error (Hello from C!): Shape mismatch in dimension %d\n", i);
                return;
            }
        }
        
        if (tensor->dtype != sub_tensor->dtype) {
            fprintf(stderr, "Error (Hello from C!): Data type mismatch in tensor assignment\n");
            return;
        }
        
        int* full_dest_shape = malloc(sizeof(int) * tensor->rank);
        if (!full_dest_shape) {
            fprintf(stderr, "Error (Hello from C!): Memory allocation failed\n");
            return;
        }
        
        memcpy(full_dest_shape, shape, sizeof(int) * rank);
        
        for (int i = rank; i < tensor->rank; i++) {
            full_dest_shape[i] = -1;
        }
        
        int* src_iter_shape = malloc(sizeof(int) * sub_tensor->rank);
        if (!src_iter_shape) {
            free(full_dest_shape);
            fprintf(stderr, "Error (Hello from C!): Memory allocation failed\n");
            return;
        }
        
        for (int i = 0; i < sub_tensor->rank; i++) {
            src_iter_shape[i] = -1;
        }
        
        tensor_set_subtensor(tensor, full_dest_shape, rank, sub_tensor, src_iter_shape, 0);
        
        free(src_iter_shape);
        free(full_dest_shape);
        return;
    }
    
    for(int i = 0; i < rank; i++) {
        if(shape[i] == -1) {
            for(int shape_i = 0; shape_i < tensor->shape[i]; shape_i++) {
                shape[i] = shape_i;
                naive_tref_setter(tensor, shape, rank, change_with, is_tensor);
            }
            
            shape[i] = -1;
            return;
        }
    }
    
    if(rank == tensor->rank) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (tref setter)\n");
                return;
            }
        }
    } else {
        int new_rank = rank + 1;
        int* subshape = malloc(sizeof(int) * new_rank);
        if (!subshape) {
            fprintf(stderr, "Error (Hello from C!): Memory allocation failed\n");
            return;
        }

        memcpy(subshape, shape, sizeof(int) * rank);

        for (int i = 0; i < tensor->shape[rank]; i++) {
            subshape[rank] = i;
            naive_tref_setter(tensor, subshape, new_rank, change_with, is_tensor);
        }

        free(subshape);
    }
}

void tensor_set_subtensor(Tensor* dest, int* dest_shape, int dest_rank, Tensor* src, int* src_shape, int src_rank) {
    if (src_rank == src->rank) {
        void* dest_ptr = nnl2_view(dest, dest_shape, dest_rank);
        void* src_ptr = nnl2_view(src, src_shape, src_rank);
        
        size_t type_size;
        switch (dest->dtype) {
            case FLOAT64: type_size = sizeof(double); break;
            case FLOAT32: type_size = sizeof(float); break;
            case INT32: type_size = sizeof(int32_t); break;
            default: fprintf(stderr, "Error (Hello from C!): Unsupported data type\n"); return;
        }
        
        memcpy(dest_ptr, src_ptr, type_size);
        return;
    }

    if (src_shape[src_rank] == -1) {
        for (int i = 0; i < src->shape[src_rank]; i++) {
            src_shape[src_rank] = i;
            dest_shape[dest_rank] = i;
            tensor_set_subtensor(dest, dest_shape, dest_rank + 1, src, src_shape, src_rank + 1);
        }
        src_shape[src_rank] = -1;
        dest_shape[dest_rank] = -1;
    } else {
        dest_shape[dest_rank] = src_shape[src_rank];
        tensor_set_subtensor(dest, dest_shape, dest_rank + 1, src, src_shape, src_rank + 1);
    }
}

Implementation tref_setter_backends[] = {
	REGISTER_BACKEND(naive_tref_setter, nnl2_naive, NAIVE_BACKEND_NAME),
};	

trefsetterfn tref_setter;

void set_tref_setter_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(tref_setter_backends, tref_setter, backend_name);
}

void naive_scaleinplace(Tensor* tensor, float multiplier) {
	void* data = tensor->data;
	int num_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* data_t = (double*)data;
			for(int i = 0; i < num_elems; i++) data_t[i] *= (double)multiplier;
			break;
		}
		
		case FLOAT32: {
			float* data_t = (float*)data;
			for(int i = 0; i < num_elems; i++) data_t[i] *= multiplier;
			break;
		}
		
		case INT32: {
			int32_t* data_t = (int32_t*)data;
			for(int i = 0; i < num_elems; i++) data_t[i] = (int32_t)round(data_t[i] * multiplier);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive scale in-place)\n");
			return;
		}
	}
}

Implementation scaleinplace_backends[] = {
	REGISTER_BACKEND(naive_scaleinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

scaleinplacefn scaleinplace;
make_current_backend(scaleinplace);

void set_scaleinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(scaleinplace_backends, scaleinplace, backend_name, current_backend(scaleinplace));
}

const char* get_scaleinplace_backend() {
	return current_backend(scaleinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(scaleinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(scaleinplace);

Tensor* naive_scale(const Tensor* tensor, float multiplier) {
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	void* data_original = tensor->data;
	void* data_result = result->data;
	int num_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* data_t = (double*)data_original;
			double* data_o = (double*)data_result;
			for(int i = 0; i < num_elems; i++) data_o[i] = data_t[i] * (double)multiplier;
			break;
		}
		
		case FLOAT32: {
			float* data_t = (float*)data_original;
			float* data_o = (float*)data_result;
			for(int i = 0; i < num_elems; i++) data_o[i] = data_t[i] * multiplier;
			break;
		}
		
		case INT32: {
			int32_t* data_t = (int32_t*)data_original;
			int32_t* data_o = (int32_t*)data_result;
			for(int i = 0; i < num_elems; i++) data_o[i] = (int32_t)round(data_t[i] * multiplier);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive scale in-place)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation scale_backends[] = {
	REGISTER_BACKEND(naive_scale, nnl2_naive, NAIVE_BACKEND_NAME),
};	

scalefn scale;
make_current_backend(scale);

void set_scale_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(scale_backends, scale, backend_name, current_backend(scale));
}

const char* get_scale_backend() {
	return current_backend(scale);
}

DEFINE_GET_BACKENDS_FUNCTION(scale);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(scale);

Tensor* empty_like(const Tensor* tensor) {
	return nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
}

Tensor* zeros_like(const Tensor* tensor) {
	return nnl2_zeros(tensor->shape, tensor->rank, tensor->dtype);
}

Tensor* ones_like(const Tensor* tensor) {
	return ones(tensor->shape, tensor->rank, tensor->dtype);
}

Tensor* full_like(const Tensor* tensor, void* filler) {
	return full(tensor->shape, tensor->rank, tensor->dtype, filler);
}

void naive_maxinplace(Tensor* tensora, const Tensor* tensorb) {
	TensorType typea = tensora->dtype, typeb = tensorb->dtype;
	
	if(typea != typeb) {
		fprintf(stderr, "Error (Hello from C!): Data types are different (max in-place)\n");
		return;
	}
	
	int total_elems = product(tensora->shape, tensora->rank);
	
	void* data_a = tensora->data;
	void* data_b = tensorb->data;
	
	switch(typea) {
		case FLOAT64: {
			double* cast_data_a = (double*)data_a;
			double* cast_data_b = (double*)data_b;
			
			for(int i = 0; i < total_elems; i++) cast_data_a[i] = MAX(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		case FLOAT32: {
			float* cast_data_a = (float*)data_a;
			float* cast_data_b = (float*)data_b;
			
			for(int i = 0; i < total_elems; i++) cast_data_a[i] = MAX(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		case INT32: {
			int32_t* cast_data_a = (int32_t*)data_a;
			int32_t* cast_data_b = (int32_t*)data_b;
			
			for(int i = 0; i < total_elems; i++) cast_data_a[i] = MAX(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (max in-place)\n");
			return;
		}
	}
}

Implementation maxinplace_backends[] = {
	REGISTER_BACKEND(naive_maxinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

maxinplacefn maxinplace;
make_current_backend(maxinplace);

void set_maxinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(maxinplace_backends, maxinplace, backend_name, current_backend(maxinplace));
}

const char* get_maxinplace_backend() {
	return current_backend(maxinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(maxinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(maxinplace);

void naive_mininplace(Tensor* tensora, const Tensor* tensorb) {
	TensorType typea = tensora->dtype, typeb = tensorb->dtype;
	
	if(typea != typeb) {
		fprintf(stderr, "Error (Hello from C!): Data types are different (min in-place)\n");
		return;
	}
	
	int total_elems = product(tensora->shape, tensora->rank);
	
	void* data_a = tensora->data;
	void* data_b = tensorb->data;
	
	switch(typea) {
		case FLOAT64: {
			double* cast_data_a = (double*)data_a;
			double* cast_data_b = (double*)data_b;
			
			for(int i = 0; i < total_elems; i++) cast_data_a[i] = MIN(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		case FLOAT32: {
			float* cast_data_a = (float*)data_a;
			float* cast_data_b = (float*)data_b;
			
			for(int i = 0; i < total_elems; i++) cast_data_a[i] = MIN(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		case INT32: {
			int32_t* cast_data_a = (int32_t*)data_a;
			int32_t* cast_data_b = (int32_t*)data_b;
			
			for(int i = 0; i < total_elems; i++) cast_data_a[i] = MIN(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (min in-place)\n");
			return;
		}
	}
}

Implementation mininplace_backends[] = {
	REGISTER_BACKEND(naive_mininplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

mininplacefn mininplace;
make_current_backend(mininplace);

void set_mininplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mininplace_backends, mininplace, backend_name, current_backend(mininplace));
}

const char* get_mininplace_backend() {
	return current_backend(mininplace);
}

DEFINE_GET_BACKENDS_FUNCTION(mininplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mininplace);

Tensor* naive_max(const Tensor* tensora, const Tensor* tensorb) {
	TensorType typea = tensora->dtype, typeb = tensorb->dtype;
	
	if(typea != typeb) {
		fprintf(stderr, "Error (Hello from C!): Data types are different (max in-place)\n");
		return NULL;
	}
	
	int total_elems = product(tensora->shape, tensora->rank);
	
	Tensor* result = nnl2_empty(tensora->shape, tensora->rank, typea);
	
	void* data_a = tensora->data;
	void* data_b = tensorb->data;
	void* data_result = result->data;
	
	switch(typea) {
		case FLOAT64: {
			double* cast_data_a = (double*)data_a;
			double* cast_data_b = (double*)data_b;
			double* cast_data_result = (double*)data_result;
			
			for(int i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		case FLOAT32: {
			float* cast_data_a = (float*)data_a;
			float* cast_data_b = (float*)data_b;
			float* cast_data_result = (float*)data_result;
			
			for(int i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		case INT32: {
			int32_t* cast_data_a = (int32_t*)data_a;
			int32_t* cast_data_b = (int32_t*)data_b;
			int32_t* cast_data_result = (int32_t*)data_result;
			
			for(int i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (max in-place)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation max_backends[] = {
	REGISTER_BACKEND(naive_max, nnl2_naive, NAIVE_BACKEND_NAME),
};	

maxfn nnl2_max;
make_current_backend(max);

void set_max_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(max_backends, nnl2_max, backend_name, current_backend(max));
}

const char* get_max_backend() {
	return current_backend(max);
}

DEFINE_GET_BACKENDS_FUNCTION(max);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(max);

Tensor* naive_min(const Tensor* tensora, const Tensor* tensorb) {
	TensorType typea = tensora->dtype, typeb = tensorb->dtype;
	
	if(typea != typeb) {
		fprintf(stderr, "Error (Hello from C!): Data types are different (min in-place)\n");
		return NULL;
	}
	
	int total_elems = product(tensora->shape, tensora->rank);
	
	Tensor* result = nnl2_empty(tensora->shape, tensora->rank, typea);
	
	void* data_a = tensora->data;
	void* data_b = tensorb->data;
	void* data_result = result->data;
	
	switch(typea) {
		case FLOAT64: {
			double* cast_data_a = (double*)data_a;
			double* cast_data_b = (double*)data_b;
			double* cast_data_result = (double*)data_result;
			
			for(int i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		case FLOAT32: {
			float* cast_data_a = (float*)data_a;
			float* cast_data_b = (float*)data_b;
			float* cast_data_result = (float*)data_result;
			
			for(int i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		case INT32: {
			int32_t* cast_data_a = (int32_t*)data_a;
			int32_t* cast_data_b = (int32_t*)data_b;
			int32_t* cast_data_result = (int32_t*)data_result;
			
			for(int i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_a[i], cast_data_b[i]);
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (min in-place)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation min_backends[] = {
	REGISTER_BACKEND(naive_min, nnl2_naive, NAIVE_BACKEND_NAME),
};	

minfn nnl2_min;
make_current_backend(min);

void set_min_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(min_backends, nnl2_min, backend_name, current_backend(min));
}

const char* get_min_backend() {
	return current_backend(min);
}

DEFINE_GET_BACKENDS_FUNCTION(min);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(min);

void naive_absinplace(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) cast_data[i] = fabs(cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) cast_data[i] = fabsf(cast_data[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(int i = 0; i < total_elems; i++) cast_data[i] = abs(cast_data[i]);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (abs in-place)\n");
			return;
		}
	}
}

Implementation absinplace_backends[] = {
	REGISTER_BACKEND(naive_absinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

absinplacefn absinplace;
make_current_backend(absinplace);

void set_absinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(absinplace_backends, absinplace, backend_name, current_backend(absinplace));
}

const char* get_absinplace_backend() {
	return current_backend(absinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(absinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(absinplace);

Tensor* naive_abs(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = fabs(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = fabsf(cast_data_t[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = abs(cast_data_t[i]);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (abs)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation abs_backends[] = {
	REGISTER_BACKEND(naive_abs, nnl2_naive, NAIVE_BACKEND_NAME),
};	

absfn nnl2_abs;
make_current_backend(abs);

void set_abs_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(abs_backends, nnl2_abs, backend_name, current_backend(abs));
}

const char* get_abs_backend() {
	return current_backend(abs);
}

DEFINE_GET_BACKENDS_FUNCTION(abs);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(abs);

void* get_tensor_data(Tensor* tensor) {
	return tensor->data;
}

float* internal_get_float_data_tensor(Tensor* tensor) {
	return (float*)tensor->data;
}

Tensor* naive_hstack(const Tensor* tensora, const Tensor* tensorb) {
	TensorType typea = tensora->dtype;
	TensorType typeb = tensorb->dtype;
	
	int ranka = tensora->rank;
	int rankb = tensorb->rank;
	
	if(typea != typeb) {
		fprintf(stderr, "Error (Hello from C!): Data types are different (naive-hstack)\n");
		return NULL;
	}
	
	if(ranka != rankb) {
		fprintf(stderr, "Error (Hello from C!): Tensors ranks are different (naive-hstack)\n");
		return NULL;
	}
	
    size_t sizea = product(tensora->shape, tensora->rank);
	size_t sizeb = product(tensorb->shape, tensorb->rank);
	
	int* shapea = tensora->shape;
	int* shapeb = tensorb->shape;
	
	Tensor* result;
	
	if(ranka == 1) {
		int* shapec = malloc(sizeof(int));
		
		if (shapec == NULL) {
			fprintf(stderr, "Error (Hello from C!): Memory allocation failed! (naive-hstack)\n");
			return NULL; 
		}
		
		shapec[0] = shapea[0] + shapeb[0];
		result = nnl2_empty(shapec, 1, typea);
		free(shapec); 
		
		size_t item_size = get_dtype_size(typea);
		
		size_t total_size_a = sizea * item_size;
        size_t total_size_b = sizeb * item_size;
		
		void* dataa = tensora->data;
		void* datab = tensorb->data;
		
		memcpy(result->data, dataa, total_size_a);
		memcpy((char*)result->data + total_size_a, datab, total_size_b); 
	} else {
		int* shapec = malloc(ranka * sizeof(int));
		
		if (shapec == NULL) {
			fprintf(stderr, "Error (Hello from C!): Memory allocation failed! (naive-hstack)\n");
			return NULL; 
		}
		
		for(int i = 0; i < ranka; i++) {
			if(i == 1) {
				shapec[i] = shapea[i] + shapeb[i];
			} else {
				shapec[i] = shapea[i];
			}
		}

		result = nnl2_empty(shapec, ranka, typea);
		free(shapec); 
		
		size_t item_size = get_dtype_size(typea);
		
		size_t outer_dim = shapea[0];  

		size_t row_size_a = product(shapea + 1, ranka - 1) * item_size;
        size_t row_size_b = product(shapeb + 1, rankb - 1) * item_size;	
		
		char* src_a = tensora->data;
        char* src_b = tensorb->data;
        char* dst = result->data;
		
		for(size_t i = 0; i < outer_dim; i++) {
            memcpy(dst, src_a, row_size_a);
            src_a += row_size_a;
            dst += row_size_a;
            
            memcpy(dst, src_b, row_size_b);
            src_b += row_size_b;
            dst += row_size_b;
        }
	}
	
	return result;
}

Implementation hstack_backends[] = {
	REGISTER_BACKEND(naive_hstack, nnl2_naive, NAIVE_BACKEND_NAME),
};	

hstackfn hstack;
make_current_backend(hstack);

void set_hstack_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(hstack_backends, hstack, backend_name, current_backend(hstack));
}

const char* get_hstack_backend() {
	return current_backend(hstack);
}

DEFINE_GET_BACKENDS_FUNCTION(hstack);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(hstack);

Tensor* naive_vstack(const Tensor* tensora, const Tensor* tensorb) {
	TensorType typea = tensora->dtype;
	TensorType typeb = tensorb->dtype;
	
	int ranka = tensora->rank;
	int rankb = tensorb->rank;
	
	if(typea != typeb) {
		fprintf(stderr, "Error (Hello from C!): Data types are different (naive-vstack)\n");
		return NULL;
	}
	
    size_t sizea = product(tensora->shape, tensora->rank);
	size_t sizeb = product(tensorb->shape, tensorb->rank);
	
	void* dataa = tensora->data;
	void* datab = tensorb->data;
	
	int* shapea = tensora->shape;
	int* shapeb = tensorb->shape;
	
	Tensor* result = NULL;
	
	if(ranka == 1 && rankb == 1) {
		int* shapec = malloc(2 * sizeof(int));
		
		if (shapec == NULL) {
			fprintf(stderr, "Error (Hello from C!): Memory allocation failed! (naive-vstack)\n");
			return NULL; 
		}
		
		shapec[1] = shapea[0];
		shapec[0] = 2;
		result = nnl2_empty(shapec, 2, typea);
		free(shapec); 
		
		size_t item_size = get_dtype_size(typea);
		
		size_t total_size_a = sizea * item_size;
        size_t total_size_b = sizeb * item_size;
		
		memcpy(result->data, dataa, total_size_a);
		memcpy((char*)result->data + total_size_a, datab, total_size_b); 
	} 
	
	else if(ranka == 2 && rankb == 1) {
		int* shapec = malloc(2 * sizeof(int));
		
		if (shapec == NULL) {
            fprintf(stderr, "Error (Hello from C!): Memory allocation failed! (naive-vstack)\n");
            return NULL; 
        }
		
		shapec[0] = shapea[0] + 1;
        shapec[1] = shapea[1];
		
		result = nnl2_empty(shapec, 2, typea);
        free(shapec);
        
        size_t item_size = get_dtype_size(typea);
        size_t row_size = shapea[1] * item_size;
		
		memcpy(result->data, dataa, shapea[0] * row_size);
		memcpy((char*)result->data + shapea[0] * row_size, datab, row_size);
	} 
	
	else if(ranka == 1 && rankb == 2) {
		int* shapec = malloc(2 * sizeof(int));
        
        if (shapec == NULL) {
            fprintf(stderr, "Error (Hello from C!): Memory allocation failed! (naive-vstack)\n");
            return NULL; 
        }
		
		shapec[0] = shapeb[0] + 1;
        shapec[1] = shapeb[1];
		
		result = nnl2_empty(shapec, 2, typea);
        free(shapec);
		
		size_t item_size = get_dtype_size(typea);
        size_t row_size = shapeb[1] * item_size;
		
		memcpy(result->data, dataa, row_size);
		memcpy((char*)result->data + row_size, datab, shapeb[0] * row_size);
	} 
	
	else {
		int* shapec = malloc(ranka * sizeof(int));
		
		if (shapec == NULL) {
			fprintf(stderr, "Error (Hello from C!): Memory allocation failed! (naive-vstack)\n");
			return NULL; 
		}
		
		shapec[0] = shapea[0] + shapeb[0];
		
		for(int i = 1; i < ranka; i++) {
			shapec[i] = shapea[i];
		}

		result = nnl2_empty(shapec, ranka, typea);
		free(shapec); 
		
		size_t item_size = get_dtype_size(typea);
		
		size_t total_size_a = sizea * item_size;
        size_t total_size_b = sizeb * item_size;

		memcpy(result->data, dataa, total_size_a); 
        memcpy((char*)result->data + total_size_a, datab, total_size_b); 
	}
	
	return result;
}

Implementation vstack_backends[] = {
	REGISTER_BACKEND(naive_vstack, nnl2_naive, NAIVE_BACKEND_NAME),
};	

vstackfn vstack;
make_current_backend(vstack);

void set_vstack_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(vstack_backends, vstack, backend_name, current_backend(vstack));
}

const char* get_vstack_backend() {
	return current_backend(vstack);
}

DEFINE_GET_BACKENDS_FUNCTION(vstack);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(vstack);

void naive_reluinplace(Tensor* tensor) {
	int total_elems = product(tensor->shape, tensor->rank);	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_relu_float64_inplace(&cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_relu_float32_inplace(&cast_data[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_relu_int32_inplace(&cast_data[i]);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (relu in-place)\n");
			return;
		}
	}
}

Implementation reluinplace_backends[] = {
	REGISTER_BACKEND(naive_reluinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

reluinplacefn reluinplace;
make_current_backend(reluinplace);

void set_reluinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(reluinplace_backends, reluinplace, backend_name, current_backend(reluinplace));
}

const char* get_reluinplace_backend() {
	return current_backend(reluinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(reluinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(reluinplace);

Tensor* naive_relu(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_float64(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_float32(cast_data_t[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_relu_int32(cast_data_t[i]);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (relu)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation relu_backends[] = {
	REGISTER_BACKEND(naive_relu, nnl2_naive, NAIVE_BACKEND_NAME),
};	

relufn relu;
make_current_backend(relu);

void set_relu_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(relu_backends, relu, backend_name, current_backend(relu));
}

const char* get_relu_backend() {
	return current_backend(relu);
}

DEFINE_GET_BACKENDS_FUNCTION(relu);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(relu);

void naive_leakyreluinplace(Tensor* tensor, float alpha) {
	int total_elems = product(tensor->shape, tensor->rank);	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_leaky_relu_float64_inplace(&cast_data[i], alpha);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_leaky_relu_float32_inplace(&cast_data[i], alpha);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_leaky_relu_int32_inplace(&cast_data[i], alpha);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (leaky-relu in-place)\n");
			return;
		}
	}
}

Implementation leakyreluinplace_backends[] = {
	REGISTER_BACKEND(naive_leakyreluinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

leakyreluinplacefn leakyreluinplace;
make_current_backend(leakyreluinplace);

void set_leakyreluinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(leakyreluinplace_backends, leakyreluinplace, backend_name, current_backend(leakyreluinplace));
}

const char* get_leakyreluinplace_backend() {
	return current_backend(leakyreluinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(leakyreluinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(leakyreluinplace);

Tensor* naive_leakyrelu(Tensor* tensor, float alpha) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	bool float64_conversion = false;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_float64(cast_data_t[i], alpha);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_float32(cast_data_t[i], alpha);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			
			for(int i = 0; i < total_elems; i++) {
				if(cast_data_t[i] < 0) {
					float result_val = cast_data_t[i] * alpha;
					
					if(fmodf(result_val, 1.0f) != 0.0f) {
						float64_conversion = true;
						break;
					}
				}
			}
			
			if(float64_conversion) {
				nnl2_free_tensor(result);
				
				result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
				data_r = result->data;
				
				double* cast_data_r_f64 = (double*)data_r;
				
				for(int i = 0; i < total_elems; i++) {
					if(cast_data_t[i] >= 0) {
						cast_data_r_f64[i] = (double)cast_data_t[i];
					} else {
						cast_data_r_f64[i] = (double)(cast_data_t[i] * alpha);
					}
				}
			} else {
				for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_int32(cast_data_t[i], alpha);
			}
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (leaky-relu)\n");
			return NULL;	
		}
	}
	
	return result;
}

Implementation leakyrelu_backends[] = {
	REGISTER_BACKEND(naive_leakyrelu, nnl2_naive, NAIVE_BACKEND_NAME),
};	

leakyrelufn leakyrelu;
make_current_backend(leakyrelu);

void set_leakyrelu_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(leakyrelu_backends, leakyrelu, backend_name, current_backend(leakyrelu));
}

const char* get_leakyrelu_backend() {
	return current_backend(leakyrelu);
}

DEFINE_GET_BACKENDS_FUNCTION(leakyrelu);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(leakyrelu);

void naive_sigmoidinplace(Tensor* tensor) {
	int total_elems = product(tensor->shape, tensor->rank);	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_sigmoid_float64_inplace(&cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_sigmoid_float32_inplace(&cast_data[i]);
			break;
		}
		
		case INT32: {
			fprintf(stderr, "Error (Hello from C!): Sigmoid in-place cannot be applied to the provided tensor\n");
			exit(EXIT_FAILURE);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (sigmoid in-place)\n");
			return;
		}
	}
}

Implementation sigmoidinplace_backends[] = {
	REGISTER_BACKEND(naive_sigmoidinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

sigmoidinplacefn sigmoidinplace;
make_current_backend(sigmoidinplace);

void set_sigmoidinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoidinplace_backends, sigmoidinplace, backend_name, current_backend(sigmoidinplace));
}

const char* get_sigmoidinplace_backend() {
	return current_backend(sigmoidinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(sigmoidinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoidinplace);

Tensor* naive_sigmoid(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float64((double)cast_data_t[i]); 
			break;
		}
		
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float64(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float32(cast_data_t[i]);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (sigmoid)\n");
			return NULL;
		}
	}
	
	return result;
}


Implementation sigmoid_backends[] = {
	REGISTER_BACKEND(naive_sigmoid, nnl2_naive, NAIVE_BACKEND_NAME),
};	

sigmoidfn sigmoid;
make_current_backend(sigmoid);

void set_sigmoid_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoid_backends, sigmoid, backend_name, current_backend(sigmoid));
}

const char* get_sigmoid_backend() {
	return current_backend(sigmoid);
}

DEFINE_GET_BACKENDS_FUNCTION(sigmoid);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoid);

void naive_tanhinplace(Tensor* tensor) {
	int total_elems = product(tensor->shape, tensor->rank);	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) cast_data[i] = tanh(cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) cast_data[i] = tanhf(cast_data[i]);
			break;
		}
		
		case INT32: {
			fprintf(stderr, "Error (Hello from C!): Tanh in-place cannot be applied to the provided tensor\n");
			exit(EXIT_FAILURE);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (tanh in-place)\n");
			return;
		}
	}
}

Implementation tanhinplace_backends[] = {
	REGISTER_BACKEND(naive_tanhinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

tanhinplacefn tanhinplace;
make_current_backend(tanhinplace);

void set_tanhinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(tanhinplace_backends, tanhinplace, backend_name, current_backend(tanhinplace));
}

const char* get_tanhinplace_backend() {
	return current_backend(tanhinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(tanhinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(tanhinplace);

Tensor* naive_tanh(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = tanh((double)cast_data_t[i]); 
			break;
		}
		
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = tanh(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = tanhf(cast_data_t[i]);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (tanh)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation tanh_backends[] = {
	REGISTER_BACKEND(naive_tanh, nnl2_naive, NAIVE_BACKEND_NAME),
};	

tanhfn nnl2_tanh;
make_current_backend(tanh);

void set_tanh_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(tanh_backends, nnl2_tanh, backend_name, current_backend(tanh));
}

const char* get_tanh_backend() {
	return current_backend(tanh);
}

DEFINE_GET_BACKENDS_FUNCTION(tanh);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(tanh);

Tensor* naive_concat(const Tensor* tensora, const Tensor* tensorb, int axis) {
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;

    int ranka = tensora->rank;
    int rankb = tensorb->rank;
    
    if (typea != typeb) {
        fprintf(stderr, "Error (Hello from C!): Data types are different (naive-concat)\n");
        return NULL;
    }

    if (ranka != rankb) {
        fprintf(stderr, "Error (Hello from C!): Ranks are different (naive-concat)\n");
        return NULL;
    }

    if (axis < 0 || axis >= ranka) {
        fprintf(stderr, "Error (Hello from C!): Invalid axis (naive-concat)\n");
        return NULL;
    }

    for (int i = 0; i < ranka; i++) {
        if (i != axis && tensora->shape[i] != tensorb->shape[i]) {
            fprintf(stderr, "Error (Hello from C!): Incompatible shapes along axis %d (naive-concat)\n", i);
            return NULL;
        }
    }
    
    int* shapea = tensora->shape;
    int* shapeb = tensorb->shape;

    int* shapec = malloc(ranka * sizeof(int));
    if (shapec == NULL) {
        fprintf(stderr, "Error (Hello from C!): Memory allocation failed! (naive-concat)\n");
        return NULL;
    }
    
    for (int i = 0; i < ranka; i++) {
        shapec[i] = (i == axis) ? (shapea[i] + shapeb[i]) : shapea[i];
    }    
    
    Tensor* result = nnl2_empty(shapec, ranka, typea);
    
    if (result == NULL) {
        free(shapec);
        return NULL;
    }
    
    free(shapec);

    size_t item_size = get_dtype_size(typea);
    
    size_t outer_stride = 1;
    for (int i = 0; i < axis; i++) {
        outer_stride *= tensora->shape[i];
    }
    
    size_t inner_stride = 1;
    for (int i = axis + 1; i < ranka; i++) {
        inner_stride *= tensora->shape[i];
    }
    
    size_t a_step = tensora->shape[axis] * inner_stride;
    size_t b_step = tensorb->shape[axis] * inner_stride;
    
    char* a_data = (char*)tensora->data;
    char* b_data = (char*)tensorb->data;
    char* c_data = (char*)result->data;
    
    for (size_t outer = 0; outer < outer_stride; outer++) {
        memcpy(c_data, a_data, a_step * item_size);
        c_data += a_step * item_size;
        a_data += a_step * item_size;
        
        memcpy(c_data, b_data, b_step * item_size);
        c_data += b_step * item_size;
        b_data += b_step * item_size;
    }

    return result;
}

Implementation concat_backends[] = {
	REGISTER_BACKEND(naive_concat, nnl2_naive, NAIVE_BACKEND_NAME),
};	

concatfn nnl2_concat;
make_current_backend(concat);

void set_concat_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(concat_backends, nnl2_concat, backend_name, current_backend(concat));
}

const char* get_concat_backend() {
	return current_backend(concat);
}

DEFINE_GET_BACKENDS_FUNCTION(concat);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(concat);

Tensor* naive_randn(int* shape, int rank, TensorType dtype, void* from, void* to) {
	Tensor* result = nnl2_empty(shape, rank, dtype);
	
	size_t total_elems = product(shape, rank);
	
	switch(dtype) {
		case FLOAT64: {
			double from_cast = *((double*)from);
			double to_cast = *((double*)to);
			double* data = (double*)result->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + (to_cast - from_cast) * ((double)rand() / RAND_MAX);
			break;
		}
		
		case FLOAT32: {
			float from_cast = *((float*)from);
			float to_cast = *((float*)to);
			float* data = (float*)result->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + (to_cast - from_cast) * ((float)rand() / RAND_MAX);
			break;
		}
		
		case INT32: {
			int32_t from_cast = *((int32_t*)from);
			int32_t to_cast = *((int32_t*)to);
			int32_t* data = (int32_t*)result->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + rand() % (to_cast - from_cast + 1);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive-randn)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation randn_backends[] = {
	REGISTER_BACKEND(naive_randn, nnl2_naive, NAIVE_BACKEND_NAME),
};	

randnfn randn;
make_current_backend(randn);

void set_randn_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(randn_backends, randn, backend_name, current_backend(randn));
}

const char* get_randn_backend() {
	return current_backend(randn);
}

DEFINE_GET_BACKENDS_FUNCTION(randn);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(randn);

Tensor* naive_xavier(int* shape, int rank, TensorType dtype, int in, int out, float gain, float distribution) {
	if(dtype == INT32) {
		fprintf(stderr, "Error (Hello from C!): Provided an not supported type (xavier-naive)\n");
		return NULL;
	}
	
	Tensor* result = nnl2_empty(shape, rank, dtype);
	size_t total_elems = product(shape, rank);
	float stddev = gain * sqrt(distribution / (in + out));
	
	switch(dtype) {
		case FLOAT64: {
			double* data = (double*)result->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = -stddev + (stddev - -stddev) * ((float)rand() / RAND_MAX);
			break;
		}
		
		case FLOAT32: {
			float* data = (float*)result->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = -stddev + (stddev - -stddev) * ((float)rand() / RAND_MAX);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (xavier-naive)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation xavier_backends[] = {
	REGISTER_BACKEND(naive_xavier, nnl2_naive, NAIVE_BACKEND_NAME),
};	

xavierfn xavier;
make_current_backend(xavier);

void set_xavier_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(xavier_backends, xavier, backend_name, current_backend(xavier));
}

const char* get_xavier_backend() {
	return current_backend(xavier);
}

DEFINE_GET_BACKENDS_FUNCTION(xavier);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(xavier);

void naive_transposeinplace(Tensor* tensor) {
	if(tensor->rank < 2) {
		fprintf(stderr, "Error (Hello from C!): Provided an incorrect tensor at transpose in-place\n");
		return;
	}
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	int* shape = tensor->shape;
	
	int rows = shape[0];
	int cols = shape[1];
	
	switch(tensor->dtype) {
		case FLOAT64: {
			size_t total_bytes = total_elems * sizeof(double);
			
			double* trans_data = (double*)malloc(total_bytes);
			double* cast_data = (double*)tensor->data;
			
			if (trans_data == NULL) {
				fprintf(stderr, "Error (Hello from C!): Memory allocation failed\n");
				return;
			}
			
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;
					int trans_index = j * rows + i;
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		case FLOAT32: {
			size_t total_bytes = total_elems * sizeof(float);
			
			float* trans_data = (float*)malloc(total_bytes);
			float* cast_data = (float*)tensor->data;
			
			if (trans_data == NULL) {
				fprintf(stderr, "Error (Hello from C!): Memory allocation failed\n");
				return;
			}
			
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) { 
					int orig_index = i * cols + j;
					int trans_index = j * rows + i;
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		case INT32: {
			size_t total_bytes = total_elems * sizeof(int32_t);
			
			int32_t* trans_data = (int32_t*)malloc(total_bytes);
			int32_t* cast_data = (int32_t*)tensor->data;
			
			if (trans_data == NULL) {
				fprintf(stderr, "Error (Hello from C!): Memory allocation failed\n");
				return;
			}
			
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) { 
					int orig_index = i * cols + j;
					int trans_index = j * rows + i;
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (transpose in-place)\n");
			return;
		}
	}
}

Implementation transposeinplace_backends[] = {
	REGISTER_BACKEND(naive_transposeinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

transposeinplacefn transposeinplace;
make_current_backend(transposeinplace);

void set_transposeinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transposeinplace_backends, transposeinplace, backend_name, current_backend(transposeinplace));
}

const char* get_transposeinplace_backend() {
	return current_backend(transposeinplace);
}

DEFINE_GET_BACKENDS_FUNCTION(transposeinplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transposeinplace);

Tensor* naive_transpose(const Tensor* tensor) {
	if(tensor->rank < 2) {
		fprintf(stderr, "Error (Hello from C!): Provided an incorrect tensor at transpose in-place\n");
		return NULL;
	}
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);

	int rows = tensor->shape[0];
	int cols = tensor->shape[1];
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* src_data = (double*)tensor->data;
			double* dest_data = (double*)result->data;

			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;
					int trans_index = j * rows + i;

					dest_data[trans_index] = src_data[orig_index];
				}
			}

			break;
		}
		
		case FLOAT32: {
			float* src_data = (float*)tensor->data;
			float* dest_data = (float*)result->data;

			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;
					int trans_index = j * rows + i;

					dest_data[trans_index] = src_data[orig_index];
				}
			}

			break;
		}
		
		case INT32: {
			int32_t* src_data = (int32_t*)tensor->data;
			int32_t* dest_data = (int32_t*)result->data;

			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;
					int trans_index = j * rows + i;

					dest_data[trans_index] = src_data[orig_index];
				}
			}

			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (transpose)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation transpose_backends[] = {
	REGISTER_BACKEND(naive_transpose, nnl2_naive, NAIVE_BACKEND_NAME),
};	

transposefn transpose;
make_current_backend(transpose);

void set_transpose_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transpose_backends, transpose, backend_name, current_backend(transpose));
}

const char* get_transpose_backend() {
	return current_backend(transpose);
}

DEFINE_GET_BACKENDS_FUNCTION(transpose);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transpose);

void naive_sum(const Tensor* tensor, int* axes, int num_axes, void* result) {
	bool reduce_all = false;
	
	if(num_axes == 1 && axes[0] == 0) {
		reduce_all = true;
	} 
	
	if(reduce_all) {
		size_t total_elems = product(tensor->shape, tensor->rank);
		
		switch(tensor->dtype) {
			case FLOAT64: {
                double* cast_data = (double*)tensor->data;
                double acc = 0.0;
                for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
                *((double*)result) = acc; 
                break;
            }
	
			case FLOAT32: {
                float* cast_data = (float*)tensor->data;
                float acc = 0.0f;
                for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
                *((float*)result) = acc; 
                break;
            }
			
			case INT32: {
                int32_t* cast_data = (int32_t*)tensor->data;
                int32_t acc = 0;
                for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
                *((int32_t*)result) = acc; 
                break;
            }
			
			default: {
				fprintf(stderr, "Error (Hello from C!): Bad data-type (naive-sum)\n");
				return;
			}
		}
		
	} else {
		fprintf(stderr, "Error (Hello from C!): Sum axes in development\n");
		return;
	}
}

Implementation sum_backends[] = {
	REGISTER_BACKEND(naive_sum, nnl2_naive, NAIVE_BACKEND_NAME),
};	

sumfn nnl2_sum;
make_current_backend(sum);

void set_sum_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sum_backends, nnl2_sum, backend_name, current_backend(sum));
}

const char* get_sum_backend() {
	return current_backend(sum);
}

DEFINE_GET_BACKENDS_FUNCTION(sum);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sum);

void naive_l2norm(const Tensor* tensor, int* axes, int num_axes, void* result) {
	bool reduce_all = false;
	
	if(num_axes == 1 && axes[0] == 0) {
		reduce_all = true;
	} 
	
	if(reduce_all) {
		size_t total_elems = product(tensor->shape, tensor->rank);
		
		switch(tensor->dtype) {
			case FLOAT64: {
                double* cast_data = (double*)tensor->data;
                double acc = 0.0;
                for (size_t it = 0; it < total_elems; it++) {
                    double val = cast_data[it];
                    acc += val * val;
                }
                *((double*)result) = sqrt(acc);
                break;
            }
    
            case FLOAT32: {
                float* cast_data = (float*)tensor->data;
                float acc = 0.0f;
                for (size_t it = 0; it < total_elems; it++) {
                    float val = cast_data[it];
                    acc += val * val;
                }
                *((float*)result) = sqrtf(acc); 
                break;
            }
            
            case INT32: {
                int32_t* cast_data = (int32_t*)tensor->data;
                int32_t acc = 0;
                for (size_t it = 0; it < total_elems; it++) {
                    int32_t val = cast_data[it];
                    acc += val * val; 
                }
                *((int32_t*)result) = (int32_t)sqrt(acc);  
                break;
            }
			
			default: {
				fprintf(stderr, "Error (Hello from C!): Bad data-type (naive l2-norm)\n");
				return;
			}
		}
	} else {
		fprintf(stderr, "Error (Hello from C!): Norm axes in development\n");
		return;
	}
}

Implementation l2norm_backends[] = {
	REGISTER_BACKEND(naive_l2norm, nnl2_naive, NAIVE_BACKEND_NAME),
};	

l2normfn l2norm;
make_current_backend(l2norm);

void set_l2norm_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(l2norm_backends, l2norm, backend_name, current_backend(l2norm));
}

const char* get_l2norm_backend() {
	return current_backend(l2norm);
}

DEFINE_GET_BACKENDS_FUNCTION(l2norm);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(l2norm);

Tensor* naive_copy(const Tensor* tensor) {
	TensorType dtype = tensor->dtype;
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data;
			double* cast_data_copy = (double*)result->data;	
			for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_copy = (float*)result->data;	
			for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_copy = (int32_t*)result->data;	
			for(size_t it = 0; it < total_elems; it++) cast_data_copy[it] = cast_data_original[it];
			break;
		} 
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (copy)\n");
			return NULL;
		}
	}

	return result;
}

Implementation copy_backends[] = {
	REGISTER_BACKEND(naive_copy, nnl2_naive, NAIVE_BACKEND_NAME),
};	

copyfn nnl2_copy;
make_current_backend(copy);

void set_copy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(copy_backends, nnl2_copy, backend_name, current_backend(copy));
}

const char* get_copy_backend() {
	return current_backend(copy);
}

DEFINE_GET_BACKENDS_FUNCTION(copy);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(copy);

void naive_add_incf_inplace(Tensor* tensor, void* inc) {
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)tensor->data;
			double increment = *((double*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] += increment;
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)tensor->data;
			float increment = *((float*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] += increment;
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)tensor->data;
			int32_t increment = *((int32_t*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] += increment;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive add-incf-inplace)\n");
			return;
		}
	}
}	

Implementation add_incf_inplace_backends[] = {
	REGISTER_BACKEND(naive_add_incf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

addincfinplacefn add_incf_inplace;

void set_add_incf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_incf_inplace_backends, add_incf_inplace, backend_name);
}

Tensor* naive_add_incf(const Tensor* tensor, void* inc) {
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data;
			double* cast_data_result = (double*)result->data;
			double increment = *((double*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment;
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float increment = *((float*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment;
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t increment = *((int32_t*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive add-incf-inplace)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation add_incf_backends[] = {
	REGISTER_BACKEND(naive_add_incf, nnl2_naive, NAIVE_BACKEND_NAME),
};	

addincffn add_incf;

void set_add_incf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_incf_inplace_backends, add_incf_inplace, backend_name);
}

void naive_sub_decf_inplace(Tensor* tensor, void* inc) {
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)tensor->data;
			double increment = *((double*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] -= increment;
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)tensor->data;
			float increment = *((float*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] -= increment;
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)tensor->data;
			int32_t increment = *((int32_t*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] -= increment;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive sub-incf-inplace)\n");
			return;
		}
	}
}	

Implementation sub_decf_inplace_backends[] = {
	REGISTER_BACKEND(naive_sub_decf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

subdecfinplacefn sub_decf_inplace;

void set_sub_decf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_decf_inplace_backends, sub_decf_inplace, backend_name);
}

Tensor* naive_sub_decf(const Tensor* tensor, void* inc) {
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data;
			double* cast_data_result = (double*)result->data;
			double increment = *((double*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - increment;
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float increment = *((float*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - increment;
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t increment = *((int32_t*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - increment;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive sub-decf)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation sub_decf_backends[] = {
	REGISTER_BACKEND(naive_sub_decf, nnl2_naive, NAIVE_BACKEND_NAME),
};	

subdecffn sub_decf;

void set_sub_decf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_decf_backends, sub_decf, backend_name);
}

void naive_mul_mulf_inplace(Tensor* tensor, void* mulf) {
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)tensor->data;
			double multiply = *((double*)mulf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)tensor->data;
			float multiply = *((float*)mulf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)tensor->data;
			int32_t multiply = *((int32_t*)mulf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive mul-mulf-inplace)\n");
			return;
		}
	}
}	

Implementation mul_mulf_inplace_backends[] = {
	REGISTER_BACKEND(naive_mul_mulf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

mulmulfinplacefn mul_mulf_inplace;

void set_mul_mulf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_mulf_inplace_backends, mul_mulf_inplace, backend_name);
}

Tensor* naive_mul_mulf(const Tensor* tensor, void* mulf) {
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data;
			double* cast_data_result = (double*)result->data;
			double multiply = *((double*)mulf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float multiply = *((float*)mulf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t multiply = *((int32_t*)mulf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive mul-mulf)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation mul_mulf_backends[] = {
	REGISTER_BACKEND(naive_mul_mulf, nnl2_naive, NAIVE_BACKEND_NAME),
};	

mulmulffn mul_mulf;

void set_mul_mulf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_mulf_backends, mul_mulf, backend_name);
}

void naive_div_divf_inplace(Tensor* tensor, void* divf) {
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)tensor->data;
			double dif = *((double*)divf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] /= dif;
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)tensor->data;
			float dif = *((float*)divf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] /= dif;
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)tensor->data;
			int32_t dif = *((int32_t*)divf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] /= dif;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive div-divf-inplace)\n");
			return;
		}
	}
}	

Implementation div_divf_inplace_backends[] = {
	REGISTER_BACKEND(naive_div_divf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

divdivfinplacefn div_divf_inplace;

void set_div_divf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_divf_inplace_backends, div_divf_inplace, backend_name);
}

Tensor* naive_div_divf(const Tensor* tensor, void* divf) {
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data;
			double* cast_data_result = (double*)result->data;
			double dif = *((double*)divf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / dif;
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float dif = *((float*)divf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / dif;
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t dif = *((int32_t*)divf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / dif;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive div-divf)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation div_divf_backends[] = {
	REGISTER_BACKEND(naive_div_divf, nnl2_naive, NAIVE_BACKEND_NAME),
};	

divdivffn div_divf;

void set_div_divf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_divf_backends, div_divf, backend_name);
}

void naive_pow_powf_inplace(Tensor* tensor, void* powf_arg) {
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)tensor->data;
			double pof = *((double*)powf_arg);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = pow(cast_data[i], pof);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)tensor->data;
			float pof = *((float*)powf_arg);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = powf(cast_data[i], pof);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)tensor->data;
			int32_t pof = *((int32_t*)powf_arg);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = (int32_t)pow((double)cast_data[i], pof);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive pow-powf-inplace)\n");
			return;
		}
	}
}	

Implementation pow_powf_inplace_backends[] = {
	REGISTER_BACKEND(naive_pow_powf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

powpowfinplacefn pow_powf_inplace;

void set_pow_powf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_powf_inplace_backends, pow_powf_inplace, backend_name);
}

Tensor* naive_pow_powf(const Tensor* tensor, void* powf_arg) {
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data;
			double* cast_data_result = (double*)result->data;
			double pof = *((double*)powf_arg);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = pow(cast_data_original[i], pof);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float pof = *((float*)powf_arg);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = powf(cast_data_original[i], pof);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t pof = *((int32_t*)powf_arg);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = (int32_t)pow((double)cast_data_original[i], pof);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive pow-powf)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation pow_powf_backends[] = {
	REGISTER_BACKEND(naive_pow_powf, nnl2_naive, NAIVE_BACKEND_NAME),
};	

powpowffn pow_powf;

void set_pow_powf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_powf_backends, pow_powf, backend_name);
}

void naive_max_maxf_inplace(Tensor* tensor, void* maxf) {
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)tensor->data;
			double mxf = *((double*)maxf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], mxf);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)tensor->data;
			float mxf = *((float*)maxf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], mxf);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)tensor->data;
			int32_t mxf = *((int32_t*)maxf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], mxf);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive max-maxf-inplace)\n");
			return;
		}
	}
}	

Implementation max_maxf_inplace_backends[] = {
	REGISTER_BACKEND(naive_max_maxf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

maxmaxfinplacefn max_maxf_inplace;

void set_max_maxf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_maxf_inplace_backends, max_maxf_inplace, backend_name);
}

Tensor* naive_max_maxf(const Tensor* tensor, void* maxf) {
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data;
			double* cast_data_result = (double*)result->data;
			double mxf = *((double*)maxf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], mxf);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float mxf = *((float*)maxf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], mxf);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t mxf = *((int32_t*)maxf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], mxf);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive pow-maxf)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation max_maxf_backends[] = {
	REGISTER_BACKEND(naive_max_maxf, nnl2_naive, NAIVE_BACKEND_NAME),
};	

maxmaxffn max_maxf;

void set_max_maxf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_maxf_backends, max_maxf, backend_name);
}

void naive_min_minf_inplace(Tensor* tensor, void* minf) {
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)tensor->data;
			double mnf = *((double*)minf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], mnf);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)tensor->data;
			float mnf = *((float*)minf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], mnf);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)tensor->data;
			int32_t mnf = *((int32_t*)minf);
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], mnf);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive min-minf-inplace)\n");
			return;
		}
	}
}	

Implementation min_minf_inplace_backends[] = {
	REGISTER_BACKEND(naive_min_minf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

minminfinplacefn min_minf_inplace;

void set_min_minf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_minf_inplace_backends, min_minf_inplace, backend_name);
}

Tensor* naive_min_minf(const Tensor* tensor, void* minf) {
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data;
			double* cast_data_result = (double*)result->data;
			double mnf = *((double*)minf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], mnf);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float mnf = *((float*)minf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], mnf);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t mnf = *((int32_t*)minf);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], mnf);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (naive min-minf)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation min_minf_backends[] = {
	REGISTER_BACKEND(naive_min_minf, nnl2_naive, NAIVE_BACKEND_NAME),
};	

minminffn min_minf;

void set_min_minf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_minf_backends, min_minf, backend_name);
}

void naive_add_broadcasting_inplace(Tensor* summand, const Tensor* sumend) {
	size_t numel_summand = product(summand->shape, summand->rank);
	size_t numel_sumend = product(sumend->shape, sumend->rank);
	
	if((numel_summand % numel_sumend) == 0) {
		switch(summand->dtype) {
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
				fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-add inplace)\n");
				return;
			}
		}	
	} 
	
	else {
		fprintf(stderr, "Error (Hello from C!): Can't broadcast sumend-tensor (naive broadcasting-add inplace)\n");
		return;
	}
}

Implementation add_broadcasting_inplace_backends[] = {
	REGISTER_BACKEND(naive_add_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

addbroadcastinginplacefn add_broadcasting_inplace;

void set_add_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_broadcasting_inplace_backends, add_broadcasting_inplace, backend_name);
}

Tensor* naive_add_broadcasting(const Tensor* summand, const Tensor* sumend) {	
	size_t numel_summand = product(summand->shape, summand->rank);
	size_t numel_sumend = product(sumend->shape, sumend->rank);
	
	if((numel_summand % numel_sumend) == 0) {
		Tensor* result = nnl2_empty(summand->shape, summand->rank, summand->dtype);
		
		switch(summand->dtype) {
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
				fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-add)\n");
				return NULL;
			}
		}
		
		return result;
	}
	
	else {
		fprintf(stderr, "Error (Hello from C!): Can't broadcast sumend-tensor (naive broadcasting-add)\n");
		return NULL;
	}
}

Implementation add_broadcasting_backends[] = {
	REGISTER_BACKEND(naive_add_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};	

addbroadcastingfn add_broadcasting;

void set_add_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_broadcasting_backends, add_broadcasting, backend_name);
}

void naive_sub_broadcasting_inplace(Tensor* minuend, const Tensor* subtrahend) {
	size_t numel_minuend = product(minuend->shape, minuend->rank);
	size_t numel_subtrahend = product(subtrahend->shape, subtrahend->rank);
	
	if((numel_minuend % numel_subtrahend) == 0) {
		switch(minuend->dtype) {
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
				fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-sub inplace)\n");
				return;
			}
		}	
	} 
	
	else {
		fprintf(stderr, "Error (Hello from C!): Can't broadcast subtrahend-tensor (naive broadcasting-sub inplace)\n");
		return;
	}
}

Implementation sub_broadcasting_inplace_backends[] = {
	REGISTER_BACKEND(naive_sub_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

subbroadcastinginplacefn sub_broadcasting_inplace;

void set_sub_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_broadcasting_inplace_backends, sub_broadcasting_inplace, backend_name);
}

Tensor* naive_sub_broadcasting(const Tensor* minuend, const Tensor* subtrahend) {
	size_t numel_minuend = product(minuend->shape, minuend->rank);
	size_t numel_subtrahend = product(subtrahend->shape, subtrahend->rank);
	
	if((numel_minuend % numel_subtrahend) == 0) {
		Tensor* result = nnl2_empty(minuend->shape, minuend->rank, minuend->dtype);
		
		switch(minuend->dtype) {
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
				fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-sub)\n");
				return NULL;
			}
		}
		
		return result;
	}
	
	else {
		fprintf(stderr, "Error (Hello from C!): Can't broadcast subtrahend-tensor (naive broadcasting-sub)\n");
		return NULL;
	}
}

Implementation sub_broadcasting_backends[] = {
	REGISTER_BACKEND(naive_sub_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};	

subbroadcastingfn sub_broadcasting;

void set_sub_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_broadcasting_backends, sub_broadcasting, backend_name);
}

void naive_mul_broadcasting_inplace(Tensor* multiplicand, const Tensor* multiplier) {
    size_t numel_multiplicand = product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = product(multiplier->shape, multiplier->rank);

    if((numel_multiplicand % numel_multiplier) == 0) {
        switch(multiplicand->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-mul inplace)\n");
                return;
            }
        }
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast multiplier-tensor (naive broadcasting-mul inplace)\n");
        return;
    }
}

Implementation mul_broadcasting_inplace_backends[] = {
	REGISTER_BACKEND(naive_mul_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

mulbroadcastinginplacefn mul_broadcasting_inplace;

void set_mul_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_broadcasting_inplace_backends, mul_broadcasting_inplace, backend_name);
}

Tensor* naive_mul_broadcasting(const Tensor* multiplicand, const Tensor* multiplier) {
    size_t numel_multiplicand = product(multiplicand->shape, multiplicand->rank);
    size_t numel_multiplier = product(multiplier->shape, multiplier->rank);

    if((numel_multiplicand % numel_multiplier) == 0) {
        Tensor* result = nnl2_empty(multiplicand->shape, multiplicand->rank, multiplicand->dtype);

        switch(multiplicand->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-mul)\n");
                return NULL;
            }
        }

        return result;
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast multiplier-tensor (naive broadcasting-mul)\n");
        return NULL;
    }
}

Implementation mul_broadcasting_backends[] = {
	REGISTER_BACKEND(naive_mul_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};	

mulbroadcastingfn mul_broadcasting;

void set_mul_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_broadcasting_backends, mul_broadcasting, backend_name);
}

void naive_div_broadcasting_inplace(Tensor* dividend, const Tensor* divisor) {
    size_t numel_dividend = product(dividend->shape, dividend->rank);
    size_t numel_divisor = product(divisor->shape, divisor->rank);

    if((numel_dividend % numel_divisor) == 0) {
        switch(dividend->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-div inplace)\n");
                return;
            }
        }
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast divisor-tensor (naive broadcasting-div inplace)\n");
        return;
    }
}

Implementation div_broadcasting_inplace_backends[] = {
	REGISTER_BACKEND(naive_div_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

divbroadcastinginplacefn div_broadcasting_inplace;

void set_div_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_broadcasting_inplace_backends, div_broadcasting_inplace, backend_name);
}

Tensor* naive_div_broadcasting(const Tensor* dividend, const Tensor* divisor) {
    size_t numel_dividend = product(dividend->shape, dividend->rank);
    size_t numel_divisor = product(divisor->shape, divisor->rank);

    if((numel_dividend % numel_divisor) == 0) {
        Tensor* result = nnl2_empty(dividend->shape, dividend->rank, dividend->dtype);

        switch(dividend->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-div)\n");
                return NULL;
            }
        }

        return result;
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast divisor-tensor (naive broadcasting-div)\n");
        return NULL;
    }
}

Implementation div_broadcasting_backends[] = {
	REGISTER_BACKEND(naive_div_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};	

divbroadcastingfn div_broadcasting;

void set_div_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_broadcasting_backends, div_broadcasting, backend_name);
}

void naive_pow_broadcasting_inplace(Tensor* base, const Tensor* exponent) {
    size_t numel_base = product(base->shape, base->rank);
    size_t numel_exponent = product(exponent->shape, exponent->rank);

    if((numel_base % numel_exponent) == 0) {
        switch(base->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-pow inplace)\n");
                return;
            }
        }
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast exponent-tensor (naive broadcasting-pow inplace)\n");
        return;
    }
}

Implementation pow_broadcasting_inplace_backends[] = {
	REGISTER_BACKEND(naive_pow_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

powbroadcastinginplacefn pow_broadcasting_inplace;

void set_pow_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_broadcasting_inplace_backends, pow_broadcasting_inplace, backend_name);
}

Tensor* naive_pow_broadcasting(const Tensor* base, const Tensor* exponent) {
    size_t numel_base = product(base->shape, base->rank);
    size_t numel_exponent = product(exponent->shape, exponent->rank);

    if((numel_base % numel_exponent) == 0) {
        Tensor* result = nnl2_empty(base->shape, base->rank, base->dtype);

        switch(base->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-pow)\n");
                return NULL;
            }
        }

        return result;
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast exponent-tensor (naive broadcasting-pow)\n");
        return NULL;
    }
}

Implementation pow_broadcasting_backends[] = {
	REGISTER_BACKEND(naive_pow_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};	

powbroadcastingfn pow_broadcasting;

void set_pow_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_broadcasting_backends, add_broadcasting, backend_name);
}

void naive_max_broadcasting_inplace(Tensor* x, const Tensor* y) {
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);

    if((numel_x % numel_y) == 0) {
        switch(x->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-max inplace)\n");
                return;
            }
        }
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast y-tensor (naive broadcasting-max inplace)\n");
        return;
    }
}

Implementation max_broadcasting_inplace_backends[] = {
	REGISTER_BACKEND(naive_max_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

maxbroadcastinginplacefn max_broadcasting_inplace;

void set_max_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_broadcasting_inplace_backends, max_broadcasting_inplace, backend_name);
}

void naive_min_broadcasting_inplace(Tensor* x, const Tensor* y) {
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);

    if((numel_x % numel_y) == 0) {
        switch(x->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-min inplace)\n");
                return;
            }
        }
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast y-tensor (naive broadcasting-min inplace)\n");
        return;
    }
}	

Implementation min_broadcasting_inplace_backends[] = {
	REGISTER_BACKEND(naive_min_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

minbroadcastinginplacefn min_broadcasting_inplace;

void set_min_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_broadcasting_inplace_backends, min_broadcasting_inplace, backend_name);
}

Tensor* naive_max_broadcasting(const Tensor* x, const Tensor* y) {
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);

    if((numel_x % numel_y) == 0) {
        Tensor* result = nnl2_empty(x->shape, x->rank, x->dtype);

        switch(x->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-max)\n");
                return NULL;
            }
        }

        return result;
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast y-tensor (naive broadcasting-max)\n");
        return NULL;
    }
}

Implementation max_broadcasting_backends[] = {
	REGISTER_BACKEND(naive_max_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};	

maxbroadcastingfn max_broadcasting;

void set_max_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_broadcasting_backends, max_broadcasting, backend_name);
}

Tensor* naive_min_broadcasting(const Tensor* x, const Tensor* y) {
    size_t numel_x = product(x->shape, x->rank);
    size_t numel_y = product(y->shape, y->rank);

    if((numel_x % numel_y) == 0) {
        Tensor* result = nnl2_empty(x->shape, x->rank, x->dtype);

        switch(x->dtype) {
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
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting-min)\n");
                return NULL;
            }
        }

        return result;
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast y-tensor (naive broadcasting-min)\n");
        return NULL;
    }
}

Implementation min_broadcasting_backends[] = {
	REGISTER_BACKEND(naive_min_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};	

minbroadcastingfn min_broadcasting;

void set_min_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_broadcasting_backends, min_broadcasting, backend_name);
}

void naive_fill_tensor_with_data(Tensor* tensor, void* data, size_t num_elems) {
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
}

Implementation fill_tensor_with_data_backends[] = {
	REGISTER_BACKEND(naive_fill_tensor_with_data, nnl2_naive, NAIVE_BACKEND_NAME),
};	

filltensorwithdatafn fill_tensor_with_data;

void set_fill_tensor_with_data_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(fill_tensor_with_data_backends, fill_tensor_with_data, backend_name);
}

Tensor* make_tensor_from_flatten(void* arr, size_t num_elems_arr, int* shape, int rank, TensorType dtype) {
	size_t num_elems_tensor = product(shape, rank);
	
	if(num_elems_tensor != num_elems_arr) {
		fprintf(stderr, "Error (Hello from C!): The number of elements in the specified array does not match the specified shapes\n");
		return NULL;
	}
	
	Tensor* result = nnl2_empty(shape, rank, dtype);
	fill_tensor_with_data(result, arr, num_elems_tensor);
	return result;
}

void naive_axpy_inplace(Tensor* summand, const Tensor* sumend, float alpha) {
	size_t total_elems = product(summand->shape, summand->rank);
	
	switch(summand->dtype) {
		case FLOAT64: {
			double* summand_data = (double*)summand->data;
			double* sumend_data = (double*)sumend->data;
			for(size_t it = 0; it < total_elems; it++) summand_data[it] += (sumend_data[it] * alpha);
			break;
		}
		
		case FLOAT32: {
			float* summand_data = (float*)summand->data;
			float* sumend_data = (float*)sumend->data;
			for(size_t it = 0; it < total_elems; it++) summand_data[it] += (sumend_data[it] * alpha);
			break;
		}
		
		case INT32: {
			int32_t* summand_data = (int32_t*)summand->data;
			int32_t* sumend_data = (int32_t*)sumend->data;
			for(size_t it = 0; it < total_elems; it++) summand_data[it] += (int32_t)((float)sumend_data[it] * alpha);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (axpy in-place)\n");
			return;
		}
	}
}

Implementation axpy_inplace_backends[] = {
	REGISTER_BACKEND(naive_axpy_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

axpyinplacefn axpy_inplace;
make_current_backend(axpy_inplace);

void set_axpy_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(axpy_inplace_backends, axpy_inplace, backend_name, current_backend(axpy_inplace));
}

const char* get_axpy_inplace_backend() {
	return current_backend(axpy_inplace);
}

DEFINE_GET_BACKENDS_FUNCTION(axpy_inplace);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(axpy_inplace);

Tensor* naive_axpy(const Tensor* summand, const Tensor* sumend, float alpha) {
	size_t total_elems = product(summand->shape, summand->rank);
	Tensor* result = nnl2_empty(summand->shape, summand->rank, summand->dtype);
	
	switch(summand->dtype) {
		case FLOAT64: {
			double* summand_data = (double*)summand->data;
			double* sumend_data = (double*)sumend->data;
			double* result_data = (double*)result->data;
			for(size_t it = 0; it < total_elems; it++) result_data[it] = summand_data[it] + (sumend_data[it] * alpha);
			break;
		}
		
		case FLOAT32: {
			float* summand_data = (float*)summand->data;
			float* sumend_data = (float*)sumend->data;
			float* result_data = (float*)result->data;
			for(size_t it = 0; it < total_elems; it++) result_data[it] = summand_data[it] + (sumend_data[it] * alpha);
			break;
		}
		
		case INT32: {
			int32_t* summand_data = (int32_t*)summand->data;
			int32_t* sumend_data = (int32_t*)sumend->data;
			int32_t* result_data = (int32_t*)result->data;
			for(size_t it = 0; it < total_elems; it++) result_data[it] = summand_data[it] + (sumend_data[it] * alpha);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (axpy)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation axpy_backends[] = {
	REGISTER_BACKEND(naive_axpy, nnl2_naive, NAIVE_BACKEND_NAME),
};	

axpyfn axpy;
make_current_backend(axpy);

void set_axpy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(axpy_backends, axpy, backend_name, current_backend(axpy));
}

const char* get_axpy_backend() {
	return current_backend(axpy);
}

DEFINE_GET_BACKENDS_FUNCTION(axpy);
DEFINE_GET_NUMS_BACKENDS_FUNCTION(axpy);

void naive_axpf_inplace(Tensor* summand, void* sumend, float alpha) {
	size_t total_elems = product(summand->shape, summand->rank);
	
	switch(summand->dtype) {
		case FLOAT64: {
			double* cast_summand = (double*)summand->data;
			double cast_sumend = *((double*)sumend);
			for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha;
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
			for(size_t i = 0; i < total_elems; i++) cast_summand[i] += cast_sumend * alpha;
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (axpf in-place)\n");
			return;
		}
	}
}	

Implementation axpf_inplace_backends[] = {
	REGISTER_BACKEND(naive_axpf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

axpfinplacefn axpf_inplace;

void set_axpf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpf_inplace_backends, axpf_inplace, backend_name);
}

Tensor* naive_axpf(const Tensor* summand, void* sumend, float alpha) {
	Tensor* result = nnl2_empty(summand->shape, summand->rank, summand->dtype);
	size_t total_elems = product(summand->shape, summand->rank);
	
	switch(summand->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)summand->data;
			double* cast_data_result = (double*)result->data;
			double cast_sumend = *((double*)sumend);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (cast_sumend * alpha);
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
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + (int32_t)((double)cast_sumend * alpha);
			break;
		}
		
		default: {
			fprintf(stderr, "Error (Hello from C!): Bad data-type (axpf)\n");
			return NULL;
		}
	}
	
	return result;
}

Implementation axpf_backends[] = {
	REGISTER_BACKEND(naive_axpf, nnl2_naive, NAIVE_BACKEND_NAME),
};	

axpffn axpf;

void set_axpf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpf_backends, axpf, backend_name);
}

void naive_axpy_broadcasting_inplace(Tensor* summand, const Tensor* sumend, float alpha) {
    size_t numel_summand = product(summand->shape, summand->rank);
    size_t numel_sumend = product(sumend->shape, sumend->rank);

    if((numel_summand % numel_sumend) == 0) {
        switch(summand->dtype) {
            case FLOAT64: {
                double* cast_summand_data = (double*)summand->data;
                double* cast_sumend_data = (double*)sumend->data;

                for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                    for(size_t j = 0; j < numel_sumend; j++) {
                        cast_summand_data[i * numel_sumend + j] += cast_sumend_data[j] * alpha;
                    }
                }

                break;
            }

            case FLOAT32: {
                float* cast_summand_data = (float*)summand->data;
                float* cast_sumend_data = (float*)sumend->data;

                for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                    for(size_t j = 0; j < numel_sumend; j++) {
                        cast_summand_data[i * numel_sumend + j] = cast_sumend_data[j] * alpha;
                    }
                }

                break;
            }

            case INT32: {
                int32_t* cast_summand_data = (int32_t*)summand->data;
                int32_t* cast_sumend_data = (int32_t*)sumend->data;

                for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                    for(size_t j = 0; j < numel_sumend; j++) {
                        cast_summand_data[i * numel_sumend + j] = cast_sumend_data[j] * alpha;
                    }
                }

                break;
            }

            default: {
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive broadcasting axpy in-place)\n");
                return;
            }
        }
    }

    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast y-tensor (naive broadcasting axpy in-place)\n");
        return;
    }
}	

Implementation axpy_broadcasting_inplace_backends[] = {
	REGISTER_BACKEND(naive_axpy_broadcasting_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

axpybroadcastinginplacefn axpy_broadcasting_inplace;

void set_axpy_broadcasting_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpy_broadcasting_inplace_backends, axpy_broadcasting_inplace, backend_name);
}

Tensor* naive_axpy_broadcasting(const Tensor* summand, const Tensor* sumend, float alpha) {
    size_t numel_summand = product(summand->shape, summand->rank);
    size_t numel_sumend = product(sumend->shape, sumend->rank);

    if((numel_summand % numel_sumend) == 0) {
        Tensor* result = nnl2_empty(summand->shape, summand->rank, summand->dtype);

        switch(summand->dtype) {
            case FLOAT64: {
                double* cast_summand_data = (double*)summand->data;
                double* cast_sumend_data = (double*)sumend->data;
                double* cast_result_data = (double*)result->data;

                for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                    for(size_t j = 0; j < numel_sumend; j++) {
                        cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + (cast_sumend_data[j] * alpha); 
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

                for(size_t i = 0; i < (numel_summand / numel_sumend); i++) {
                    for(size_t j = 0; j < numel_sumend; j++) {
                        cast_result_data[i * numel_sumend + j] = cast_summand_data[i * numel_sumend + j] + (cast_sumend_data[j] * alpha); 
                    }
                }

                break;
            }

            default: {
                fprintf(stderr, "Error (Hello from C!): Bad data-type (naive axpy broadcasting)\n");
                return NULL;
            }
        }

        return result;
    }
    else {
        fprintf(stderr, "Error (Hello from C!): Can't broadcast sumend-tensor (naive axpy broadcasting)\n");
        return NULL;
    }
}

Implementation axpy_broadcasting_backends[] = {
	REGISTER_BACKEND(naive_axpy_broadcasting, nnl2_naive, NAIVE_BACKEND_NAME),
};	

axpybroadcastingfn axpy_broadcasting;

void set_axpy_broadcasting_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(axpy_broadcasting_backends, axpy_broadcasting, backend_name);
}

#endif
