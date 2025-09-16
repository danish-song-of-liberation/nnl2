#include "nnl2_core.h"
#include "nnl2_log.h"
#include "nnl2_tensor_backend.h"
#include "nnl2_backend_system_docs.h"

#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

#ifndef NNL2_TENSOR_ACCESSORS_H
#define NNL2_TENSOR_ACCESSORS_H

// NNL2

/** @file nnl2_tensor_accessors.h
 ** @brief Contains all operations for accessing tensors of type tref
 ** @copyright MIT License
 ** @date 2025
 *
 * The file fully includes tensor accessors, 
 * i.e. tref getters and setters, as well as 
 * some additional functions
 *
 ** Filepath: c
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
 
///@{ [tensor_macros]

#ifdef _WIN32 

///@{ [win32_alloc_alligned]
	
	#include <malloc.h>
	
	/** @brief 
	* Allocates aligned memory on Windows platforms
	*
	** @param ptr 
	* Pointer to be allocated
	*
	** @param alignment 
	* Memory alignment requirement
	*
	** @param size 
	* Size of memory to allocate in bytes
	*/
	#define ALLOC_ALIGNED(ptr, alignment, size) \
		do { \
			ptr = _aligned_malloc(size, alignment); \
			if (ptr == NULL) { fprintf(stderr, "Error (Hello from C!): Failed to allocate memory\n"); } \
		} while(0)
			
    /** @brief 
	* Frees aligned memory on Windows platforms
	*
    ** @param ptr 
	* Pointer to be freed
    */
	#define FREE_ALIGNED(ptr) _aligned_free(ptr)
	
///@} [win32_alloc_alligned]

#else
	
///@{ [posix_alloc_alligned]

	/** @brief 
	 * https://stackoverflow.com/questions/48332332/what-does-define-posix-source-mean 
	 */
	#define _POSIX_C_SOURCE 200809L

	#include <stdlib.h> 

	/** @brief
	* Allocates aligned memory on POSIX compliant platforms
	*
	* @param ptr 
	* Pointer to be allocated
	*
	* @param alignment 
	* Memory alignment requirement
	*
	* @param size 
	* Size of memory to allocate in bytes
	*/
	#define ALLOC_ALIGNED(ptr, alignment, size)  \
		do { \
			int err = posix_memalign(&ptr, alignment, size); \
			if (err != 0) { fprintf(stderr, "Error (Hello from C!): Failed to allocate memory\n"); } \
		} while(0)
			
    /** @brief 
	* Frees aligned memory on POSIX compliant platforms
	*
	** @param ptr 
	* Pointer to be freed
	*/	
	#define FREE_ALIGNED(ptr) free(ptr)

///@} [posix_alloc_alligned]

#endif

/** @brief Number of supported tensor data types **/
#define NUM_TENSOR_TYPES 3
	
///@{ [tensor_mem_alignment]

/** @brief Determines optimal memory alignment based on available CPU features **/
#if defined(__AVX512F__)
	#define TENSOR_MEM_ALIGNMENT 64
#elif defined(__AVX2__)
	#define TENSOR_MEM_ALIGNMENT 32
#elif defined(__AVX__)
	#define TENSOR_MEM_ALIGNMENT 32
#else 
	#define TENSOR_MEM_ALIGNMENT 16
#endif

///@} [tensor_mem_alignment]

///@{ [force_inline]

/** @brief Compiler-specific force inline directives **/
#if defined(_MSC_VER)
    #define NNL2_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define NNL2_FORCE_INLINE inline __attribute__((always_inline))
#else
    #define NNL2_FORCE_INLINE inline
#endif

///@} [force_inline]

#ifdef OPENBLAS_AVAILABLE
#include <cblas.h>
#endif

///@{ [temp_backend_macros]

/** @brief
 * I had to delete all these macros after a certain period of time. 
 * if you see this, it means that I forgot, please let me know
 */
#define current_backend(name) current_##name##_backend_name 
#define make_current_backend(name) static char current_##name##_backend_name[MAX_BACKEND_NAME_LENGTH] = ""

///@} [temp_backend_macros]
	
///@{ [backend_macros]	
	
/** @brief Current backend name reference **/	
#define CURRENT_BACKEND(name) current_##name##_backend_name 

/** @brief Creates current backend storage **/
#define MAKE_CURRENT_BACKEND(name) static char current_##name##_backend_name[MAX_BACKEND_NAME_LENGTH] = ""

///@{ [backends_names]
	#define NAIVE_BACKEND_NAME "NAIVE"
	#define UNROLL_128_BACKEND_NAME "UNROLL128"
	#define UNROLL_256_BACKEND_NAME "UNROLL256"
	#define UNROLL_512_BACKEND_NAME "UNROLL512"
	#define AVX128_BACKEND_NAME "AVX128"
	#define AVX256_BACKEND_NAME "AVX256"
	#define BLAS_BACKEND_NAME "BLAS"
	#define NNL2_OWN_NAME "NNL2 OWN IMPLEMENTATION"
///@} [backends_names]

///@} [backend_macros]	

/** @brief Minimum tensor dimension value **/
#define MIN_TENSOR_DIMENSION 0

///@{ [safety_modes]
	#define NNL2_SAFETY_MODE_OFF 0
	#define NNL2_SAFETY_MODE_MIN 1
	#define NNL2_SAFETY_MODE_MODERATE 2
	#define NNL2_SAFETY_MODE_MAX 3
///@} [safety_modes]

/** @brief Current safety mode setting **/
#define NNL2_SAFETY_MODE NNL2_SAFETY_MODE_MAX

///@{ [strides macros]
	#define NNL2_STRIDES_LAST_DIM_INDEX 1			///< Strides index for last dimension
	#define NNL2_STRIDES_SECOND_LAST_DIM_INDEX 2	///< Strides index for second last dimension
	#define NNL2_STRIDES_NEXT_DIM_INDEX 1			///< Strides index for next dimension
///@}

///@{ [errors_handling]
	#define NNL2_STACK_OVERFLOW(function) NNL2_FATAL("Stack overflow (" function ")")
	#define NNL2_STACK_UNDERFLOW(function) NNL2_FATAL("Stack underflow (" function ")")
	#define NNL2_TYPE_ERROR(transmitted_data_type) NNL2_ERROR("An unsupported/incorrect data type was passed. Enum type numbering: %d", transmitted_data_type)
	#define NNL2_TYPE_FATAL(transmitted_data_type) NNL2_FATAL("An unsupported/incorrect data type was passed. Enum type numbering: %d", transmitted_data_type)
	#define NNL2_ORDER_ERROR(transmitted_order) NNL2_ERROR("Invalid order: %d", transmitted_order)
	#define NNL2_TRANS_ERROR(transmitted_trans) NNL2_ERROR("Invalid trans: %d", transmitted_trans)
///@} [errors_handling]

/** @brief 
 * Checks for NULL pointer and returns if found
 *
 ** @param ptr 
 * Pointer to check
 *
 ** @param msg 
 * Error message to display
 */
#define NNL2_CHECK_NULL_IF_ERR_RETURN(ptr, msg) \
        if (ptr == NULL) { \
            NNL2_ERROR(msg); \
            return; \
        }
		
/** @brief 
 * Checks for NULL pointer and returns value if found
 *
 ** @param ptr 
 * Pointer to check
 *
 ** @param msg 
 * Error message to display
 *
 ** @param val
 * Value to return on error
 */		
#define NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ptr, msg, val) \
        if (ptr == NULL) { \
            NNL2_ERROR(msg); \
            return val; \
        }		

/** @brief 
 * Checks if pointer is aligned to specified boundary
 *
 * @param pntr 
 * Pointer to check
 *
 * @param alignment 
 * Alignment requirement
 *
 * @return 
 * true if aligned, false otherwise
 */
#define NNL2_IS_ALIGNED(pntr, alignment) (((uintptr_t)(pntr) % (alignment)) == 0)

///@{ [tensor_alignment_macros]
	#define NNL2_TENSOR_ALIGNMENT_64 64
	#define NNL2_TENSOR_ALIGNMENT_32 32
	#define NNL2_TENSOR_ALIGNMENT_16 16
///@} [tensor_alignment_macros]

///@{ [avx_macros]
	#define NNL2_INT32_ELEMENTS_PER_AVX256 8       ///< 256 bits / 32 bits per int32
	#define NNL2_FLOAT32_ELEMENTS_PER_AVX256 8     ///< 256 bits / 32 bits per float
	#define NNL2_FLOAT64_ELEMENTS_PER_AVX256 4     ///< 256 bits / 64 bits per double
	#define NNL2_MIN_ELEMENTS_FOR_AVX256_DOUBLE 4  ///< Minimum elements for AVX256 double processing
///@} [avx_macros]

///@{ [format_parameters]
	#define NNL2_MAX_1D_TENSOR_PRINT_ELEMENTS 100  ///< Maximum number of elements that can be printed in a 1d tensor before skipping
	#define NNL2_LARGE_TENSOR_THRESHOLD 500 	   /**< I have no idea what this does... I couldn't find a single macro with this name. 
												   /**  when I was making a smart tensor output, I was thinking about threashold as in torch, 
												   /**  but I abandoned this solution in favor of my own solution. these may be experimental 
												   /**  macros that I later removed in format function **/
	#define NNL2_LARGE_TENSOR_SAMPLE_SIZE 10	   ///< Same
	#define NNL2_1D_TENSOR_SHOW_ELEMENTS 5		   ///< Show items BEFORE and AFTER skipping
	#define NNL2_FLOAT64_FORMAT "%.6f"			   ///< Format string for double precision floating point 
	#define NNL2_FLOAT32_FORMAT "%.4f"			   ///< Format string for single precision floating point
///@} [format_parameters] 

/** @brief 
 * Validates tensor pointer and returns on error
 *
 ** @param tensor 
 * Tensor to validate
 *
 ** @param return_in_case_of_error 
 * Value to return on error
 */
#define NNL2_CHECK_TENSOR(tensor, return_in_case_of_error) \
    if (!tensor) { \
        NNL2_ERROR("Invalid tensor (NULL pointer)"); \
        return return_in_case_of_error; \
    }

/** @brief Invalid tensor type return pointer (for error handling) **/
#define NNL2_TENSOR_TYPE_INVALID_RET_PNTR ((int32_t*)-1)

/** @brief Invalid tensor type value **/
#define NNL2_TENSOR_TYPE_INVALID -1

/** @brief 
 * Maps tensor types to corresponding C types
 *
 ** @param type 
 * Tensor type enumeration
 */
#define NNL2_TENSORTYPE_TO_C_TYPE(type) \
    _Generic((type), \
        INT32: int32_t, \
        FLOAT32: float, \
        FLOAT64: double \
    )
	
/** @brief Wildcard dimension value for reinterpret (reshape as a world and representation (view)) **/	
#define NNL2_WILDCARD_DIM -1

///@} [tensor_macros]

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
NNL2_FORCE_INLINE static size_t get_dtype_size(TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();	
	    NNL2_DEBUG("Function get_dtype_size was called with dtype=%d", dtype);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if(dtype >= NUM_TENSOR_TYPES) {
			NNL2_ERROR("Invalid data type: %d", dtype);
		}
	#endif
		
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL    
		NNL2_FUNC_EXIT();	
	#endif
		
	#ifdef __GNUC__
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Warray-bounds"
		#define NNL2_SUPPRESS_ARRAY_BOUNDS
    #endif
	
	return (const size_t[]){sizeof(int), sizeof(float), sizeof(double)}[dtype]; 
	
	#ifdef __GNUC__
		#pragma GCC diagnostic pop
	    #undef NNL2_SUPPRESS_ARRAY_BOUNDS
    #endif
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
NNL2_FORCE_INLINE static size_t product(const int32_t* lst, int32_t len) { // todo rename from product to nnl2_product
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();	
		int32_t original_len = len;
	#endif
	
	// Additional checks when the debugging level is sufficient 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (lst == NULL) {
			NNL2_ERROR("product(): NULL pointer passed as shape array");
			return 0;
		}
	
		if (len <= 0) {
			NNL2_ERROR("product(): Invalid length %d", len);
			return 0;
		}
	#endif
	
	switch(len) {
		// Unrolling
		case 0: return 0;  
		case 1: return lst[0];
		case 2: return lst[0] * lst[1]; 
		case 3: return lst[0] * lst[1] * lst[2];
		case 4: return lst[0] * lst[1] * lst[2] * lst[3];
		
		default: {
			size_t acc = 1;
			while (len--) {
				#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE 
					if (*lst <= 0) {
						NNL2_ERROR("Invalid dimension value %d", *lst);
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
	}
}

/** @brief
 * Converts an arbitrary type value to double (float64)
 *
 ** @param value
 * Void pointer to the value to convert
 *
 ** @param dtype 
 * Source data type
 *
 ** @return 
 * Converted value of the float64 type
 *
 ** @note 
 * For unsupported types, it returns NAN and generates a fatal type error
 */
NNL2_FORCE_INLINE static double nnl2_convert_to_float64(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: return *((double*)value);
		case FLOAT32: return (double)(*((float*)value)); 
		case INT32:   return (double)(*((int32_t*)value)); 
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return NAN;
		}
	}
}

/** @brief
 * Converts an arbitrary type value to float (float32)
 *
 ** @param value
 * Void pointer to the value to convert
 *
 ** @param dtype 
 * Source data type
 *
 ** @return 
 * Converted value of the float32 type
 *
 ** @warning
 * When converting double->float, it checks for out-of-range float values
 *
 ** @note 
 * For unsupported types, it returns 0.0f and generates a fatal type error
 *
 */
NNL2_FORCE_INLINE static float nnl2_convert_to_float32(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: {
            double casted_double = *((double*)value);
            
			// Checking for overflow of the float range
            if (casted_double < FLT_MIN || casted_double > FLT_MAX) {
                NNL2_FATAL("FLOAT64 value out of FLOAT32 range (Point Overflow)");
                return INFINITY;
            }
			
            return (float)casted_double; 
        }
		
		case FLOAT32: return *((float*)value);
		case INT32:   return (float)(*((int32_t*)value)); 
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return 0.0f;
		}
	}
}

/** @brief
 * Converts an arbitrary type value to int (int32)
 *
 ** @param value
 * Void pointer to the value to convert
 *
 ** @param dtype 
 * Source data type
 *
 ** @return 
 * Converted value of the int32 type
 *
 ** @warning 
 * Checks that the value is within the range of int32_t
 *
 ** @warning
 * Checks that fractional numbers do not have a fractional part before conversion
 *
 ** @note 
 * For unsupported types, it returns 0 and generates a fatal error
 */
NNL2_FORCE_INLINE static int32_t nnl2_convert_to_int32(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: {
			double casted_double = *((double*)value);
			
			// Checking that the number does not have a fractional part
			if(casted_double != trunc(casted_double)) {
				NNL2_FATAL("Cannot convert FLOAT64 to INT32");
				return 0;
			}
			
			// Checking for overflow of the int32 range
			if (casted_double < INT32_MIN || casted_double > INT32_MAX) {
                NNL2_FATAL("FLOAT64 value out of INT32 range (Point Overflow)");
                return 0;
            }
			
			return (int32_t)casted_double;
		}
		
		case FLOAT32: {
			float casted_float = *((float*)value);
			
			// Checking that the number does not have a fractional part
			if(casted_float != truncf(casted_float)) {
				NNL2_FATAL("Cannot convert FLOAT32 to INT32");
				return 0;
			}
			
			// Checking for overflow of the int32 range
			if (casted_float < INT32_MIN || casted_float > INT32_MAX) {
                NNL2_FATAL("FLOAT32 value out of INT32 range");
                return 0;
            }
			
			return (int32_t)casted_float;
		}
		
		case INT32: return *((int32_t*)value);
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return 0;
		}
	}
}

/** @brief
 * Creates a new tensor without initializing the data
 *
 * This function allocates memory for the Tensor structure and its data,
 * using the provided shape and data type. The data memory is not (!) initialized
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

		if (rank <= 0) {
			NNL2_ERROR("Bad rank (%d). Rank must be positive", rank);
			return NULL;
		}
		
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
	tensor->is_view = false;
	
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
 * @see ESET_BACKEND_BY_NAME
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
	tensor->is_view = false;
	
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
 * @see ESET_BACKEND_BY_NAME
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
	
		if (!tensor->is_view && tensor->data != NULL) {
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
 * Fills the tensor with the specified value in-place
 *
 ** @details
 * The function fills all tensor elements with the specified value.
 * The operation is performed directly in the tensor memory without creating copies.
 * INT32, FLOAT32, and FLOAT64 data types are supported.
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled (the type must match the dtype)
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @example
 * // Filling a tensor with integers
 * int32_t fill_value = 42;
 * naive_inplace_fill(tensor, &fill_value, INT32);
 * 
 * // Filling a tensor with floating-point numbers
 * float float_value = 3.14f;
 * naive_inplace_fill(tensor, &float_value, FLOAT32);
 *
 ** @note
 * The function will return at the very beginning if the tensor is empty without doing anything
 *
 ** @note
 * In the NNL2_SAFETY_MODE_OFF safety mode, a regular pointer is used,
 * while in other modes, a volatile pointer is used to prevent compiler optimizations
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *	
 ** @exception NNL2Error
 * Throws error if tensor pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error  
 * Throws error if tensor data pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if value pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specified
 *
 */
bool nnl2_naive_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Validate input parameters in maximum safety mode
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (!tensor || !tensor->data || !value) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
		
			NNL2_ERROR("Incorrect tensor structure");
			return false;
		}
	#endif
	
	// Calculate total number of elements from tensor shape and rank
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return true; // Early return for empty tensors
	
	bool result = true;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value; // Extract integer fill value
			
			// Cast tensor data to integer pointer
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				int32_t* data = (int32_t*)tensor->data;
			#else
				volatile int32_t* data = (int32_t*)tensor->data;
			#endif 
			
			// Simple scalar loop for INT32 elements
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extract float fill value
			
			// Cast tensor data to float pointer
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				float* data = (float*)tensor->data;
			#else
				volatile float* data = (float*)tensor->data;
			#endif
			
			// Simple scalar loop for FLOAT32 elements
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extract double fill value
			
			// Cast tensor data to double pointer
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				double* data = (double*)tensor->data;
			#else 
				volatile double* data = (double*)tensor->data;
			#endif 
			
			// Simple scalar loop for FLOAT64 elements
			for(size_t i = 0; i < total_elems; ++i) data[i] = filler;
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype); // Error: unsupported data type
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT(); 
	#endif
	
	return result;
}

/** @brief
 * Fills the tensor with the specified value in-place using loop unrolling optimization
 *
 ** @details
 * The function fills all tensor elements with the specified value using loop unrolling
 * technique for improved performance. The operation is performed directly in the tensor
 * memory without creating copies. INT32, FLOAT32, and FLOAT64 data types are supported
 *
 * The function uses different unroll factors for different data types:
 * INT32 - 4 elements per iteration (32 bytes)
 * FLOAT32 - 4 elements per iteration (32 bytes)  
 * FLOAT64 - 8 elements per iteration (64 bytes)
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled (the type must match the dtype)]
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @note
 * In the NNL2_SAFETY_MODE_OFF safety mode, a regular pointer is used,
 * while in other modes, a volatile pointer is used to prevent compiler optimizations
 *
 ** @note
 * The function will return at the very beginning if the tensor is empty without doing anything
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *
 ** @example
 * // Filling a tensor with integers using optimized unrolled version
 * int32_t fill_value = 42;
 * unroll128_inplace_fill(tensor, &fill_value, INT32);
 * 
 * // Filling a tensor with floating-point numbers
 * float float_value = 3.14f;
 * unroll128_inplace_fill(tensor, &float_value, FLOAT32);
 *
 ** @exception NNL2Error
 * Throws error if tensor pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error  
 * Throws error if tensor data pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if value pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specifieds
 */
bool nnl2_unroll_128_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (!tensor || !tensor->data || !value) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			NNL2_ERROR("Incorrect tensor structure");
			return false;
		}
	#endif
	
	// Calculating the total number of elements in a tensor
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return true; // Exit if the tensor is empty
	
	bool result = true;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				int32_t* data = (int32_t*)tensor->data;	
			#else 
				volatile int32_t* data = (int32_t*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 4;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for int32 (4 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				float* data = (float*)tensor->data;	
			#else 
				volatile float* data = (float*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 4;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float32 (4 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				double* data = (double*)tensor->data;	
			#else 
				volatile double* data = (double*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 8;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float64 (8 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
                data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype); // Unsupported data type error
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/** @brief
 * Fills the tensor with the specified value in-place using 256-bit optimized loop unrolling
 *
 ** @details
 * The function fills all tensor elements with the specified value using loop unrolling
 * technique for improved performance. The operation is performed directly in the tensor
 * memory without creating copies. INT32, FLOAT32, and FLOAT64 data types are supported
 *
 * The function uses different unroll factors for different data types:
 * INT32 - 8 elements per iteration (32 bytes)
 * FLOAT32 - 8 elements per iteration (32 bytes)  
 * FLOAT64 - 16 elements per iteration (64 bytes)
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled (the type must match the dtype)
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @note
 * In the NNL2_SAFETY_MODE_OFF safety mode, a regular pointer is used,
 * while in other modes, a volatile pointer is used to prevent compiler optimizations
 *
 ** @note
 * The function will return at the very beginning if the tensor is empty without doing anything
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *
 ** @example
 * // Filling a tensor with integers using 256-bit optimized version
 * int32_t fill_value = 42;
 * unroll_256_inplace_fill(tensor, &fill_value, INT32);
 * 
 * // Filling a tensor with floating-point numbers
 * float float_value = 3.14f;
 * unroll_256_inplace_fill(tensor, &float_value, FLOAT32);
 *
 * // Filling a tensor with double precision numbers
 * double double_value = 2.71828;
 * unroll_256_inplace_fill(tensor, &double_value, FLOAT64);
 *
 ** @exception NNL2Error
 * Throws error if tensor pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error  
 * Throws error if tensor data pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if value pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specified
 *
 */
bool nnl2_unroll_256_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (!tensor || !tensor->data || !value) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			NNL2_ERROR("Incorrect tensor structure");
			return false;
		}
	#endif
	
	// Calculating the total number of elements in a tensor
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return true; // Exit if the tensor is empty
	
	bool result = true;
	
	switch(dtype) {
		case INT32: { 
			int32_t filler = *(int32_t*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				int32_t* data = (int32_t*)tensor->data;	
			#else 
				volatile int32_t* data = (int32_t*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 8;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for int32 (8 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				float* data = (float*)tensor->data;	
			#else 
				volatile float* data = (float*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 8;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float32 (8 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler;	data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				double* data = (double*)tensor->data;	
			#else 
				volatile double* data = (double*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 16;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float64 (16 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
				data[i + 8] = filler; data[i + 9] = filler; data[i + 10] = filler; data[i + 11] = filler;
				data[i + 12] = filler; data[i + 13] = filler; data[i + 14] = filler; data[i + 15] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler; // Unsupported data type error
            }
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype);
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/** @brief
 * Fills the tensor with the specified value in-place using 512-bit optimized loop unrolling
 *
 ** @details
 * The function fills all tensor elements with the specified value using loop unrolling
 * technique for improved performance. The operation is performed directly in the tensor
 * memory without creating copies. INT32, FLOAT32, and FLOAT64 data types are supported
 *
 * The function uses different unroll factors for different data types:
 * INT32 - 16 elements per iteration (32 bytes)
 * FLOAT32 - 16 elements per iteration (32 bytes)  
 * FLOAT64 - 32 elements per iteration (64 bytes)
 *
 ** @param tensor
 * Pointer to the tensor structure for filling
 *
 ** @param value 
 * Pointer to a value to be filled (the type must match the dtype)
 *
 ** @param dtype
 * The data type of the tensor value and elements
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *
 ** @note
 * In the NNL2_SAFETY_MODE_OFF safety mode, a regular pointer is used for maximum performance,
 * while in other modes, a volatile pointer is used to prevent compiler optimizations
 *
 ** @example
 * // Filling a tensor with integers using 512-bit optimized version
 * int32_t fill_value = 42;
 * unroll_512_inplace_fill(tensor, &fill_value, INT32);
 * 
 * // Filling a tensor with floating-point numbers
 * float float_value = 3.14f;
 * unroll_512_inplace_fill(tensor, &float_value, FLOAT32);
 *
 * // Filling a tensor with double precision numbers
 * double double_value = 2.71828;
 * unroll_512_inplace_fill(tensor, &double_value, FLOAT64);
 *
 ** @exception NNL2Error
 * Throws error if tensor pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error  
 * Throws error if tensor data pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if value pointer is NULL (only in NNL2_SAFETY_MODE_MAX)
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specified
 *
 */
bool nnl2_unroll_512_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (!tensor || !tensor->data || !value) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			NNL2_ERROR("Incorrect tensor structure");
			return false;
		}
	#endif
	
	// Calculating the total number of elements in a tensor
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if (total_elems == 0) return true; // Exit if the tensor is empty
	
	bool result = true;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				int32_t* data = (int32_t*)tensor->data;	
			#else 
				volatile int32_t* data = (int32_t*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 16;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for int32 (16 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
				data[i + 8] = filler; data[i + 9] = filler; data[i + 10] = filler; data[i + 11] = filler;
				data[i + 12] = filler; data[i + 13] = filler; data[i + 14] = filler; data[i + 15] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				float* data = (float*)tensor->data;	
			#else 
				volatile float* data = (float*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 16;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float32 (16 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler;
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
				data[i + 8] = filler; data[i + 9] = filler; data[i + 10] = filler; data[i + 11] = filler;
				data[i + 12] = filler; data[i + 13] = filler; data[i + 14] = filler; data[i + 15] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extracting the value to fill in
			
			#if NNL2_SAFETY_MODE == NNL2_SAFETY_MODE_OFF
				double* data = (double*)tensor->data;	
			#else 
				volatile double* data = (double*)tensor->data;	
			#endif
			
			size_t i = 0;
			size_t unroll_factor = 32;
			size_t main_elems = total_elems & ~(unroll_factor - 1);
			
			// Main loop with scanning for float64 (32 elements per iteration)
			for(; i < main_elems; i += unroll_factor) {
				data[i] = filler; data[i + 1] = filler; data[i + 2] = filler; data[i + 3] = filler; 
				data[i + 4] = filler; data[i + 5] = filler; data[i + 6] = filler; data[i + 7] = filler;
				data[i + 8] = filler; data[i + 9] = filler; data[i + 10] = filler; data[i + 11] = filler;
				data[i + 12] = filler; data[i + 13] = filler; data[i + 14] = filler; data[i + 15] = filler;
				data[i + 16] = filler; data[i + 17] = filler; data[i + 18] = filler; data[i + 19] = filler;
				data[i + 20] = filler; data[i + 21] = filler; data[i + 22] = filler; data[i + 23] = filler;
				data[i + 24] = filler; data[i + 25] = filler; data[i + 26] = filler; data[i + 27] = filler;
				data[i + 28] = filler; data[i + 29] = filler; data[i + 30] = filler; data[i + 31] = filler;
			}
			
			// Post-processing of the remaining elements
			for (; i < total_elems; i++) {
                data[i] = filler;
            }
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype); // Unsupported data type error
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

#ifdef __AVX__
/** @brief 
 * Fills the tensor with the specified value using AVX intrinsics
 *
 ** @details 
 * The function utilizes AVX (Advanced Vector Extensions) instructions to
 * efficiently fill tensor elements with the specified value. This implementation
 * provides significant performance improvements over scalar versions by processing
 * multiple elements simultaneously using 256-bit SIMD registers
 *
 * Memory alignment is automatically detected and appropriate instructions are used:
 * Aligned stores (_mm256_store_*) for 32-byte aligned memory
 *  Unaligned stores (_mm256_storeu_*) for unaligned memory
 *
 ** @param tensor 
 * Pointer to the tensor structure to be filled
 *
 ** @param value
 * Pointer to the fill value (must match tensor data type)
 *
 ** @param dtype 
 * Data type of the tensor elements and fill value
 *
 ** @return
 * Returns true (1) if the function is successful, false (0) if it is unsuccessful
 *
 ** @note 
 * This function requires AVX support and will only be compiled if __AVX__ is defined
 *
 ** @note 
 * For optimal performance, ensure tensor memory is 32-byte aligned
 *
 ** @example
 * // Filling aligned float tensor with AVX256
 * float fill_val = 1.0f;
 * avx_inplace_fill(tensor, &fill_val, FLOAT32);
 *
 ** @exception NNL2Error
 * Throws error if unsupported data type is specified
 *
 */
bool nnl2_avx256_inplace_fill(Tensor* tensor, void* value, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (tensor == NULL || value == NULL || tensor->data == NULL) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return false; // Invalid parameters
		}
	#endif
	
	// Calculate total elements from tensor shape and rank
	size_t total_elems = product(tensor->shape, tensor->rank);
	if (total_elems == 0) return true; // Exit if the tensor is empty
	
	// Check if tensor data is 32-byte aligned for optimal AVX performance
	bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
	
	// Warning for unaligned memory in safety modes (performance impact)
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MINI
		if(!is_aligned) {
			NNL2_WARN("In the avx256 implementation of inplace_fill, memory is not aligned to 32 bytes. Calculations may be slightly slower");
		}
	#endif
	
	bool result = true;
	
	switch(dtype) {
		case INT32: {
			int32_t filler = *(int32_t*)value; // Extract scalar fill value
			int32_t* data = (int32_t*)tensor->data; // Cast tensor data to appropriate type
			
			// Create AVX vector with 8 copies of the fill value
			__m256i avx_filler = _mm256_set1_epi32(filler);
				
			size_t avx_iters = total_elems / NNL2_INT32_ELEMENTS_PER_AVX256; // total_elems / 8
			size_t avx_processed_elems = avx_iters * NNL2_INT32_ELEMENTS_PER_AVX256; // avx_iters * 8
			
			if(is_aligned) {
				// Process aligned memory with optimized stores
				for (size_t i = 0; i < avx_iters; i++) { 
					_mm256_store_si256((__m256i*)(data + i * NNL2_INT32_ELEMENTS_PER_AVX256), avx_filler);
				}
			} else {
				// Process unaligned memory with unaligned stores
				for (size_t i = 0; i < avx_iters; i++) { 
					_mm256_storeu_si256((__m256i*)(data + i * NNL2_INT32_ELEMENTS_PER_AVX256), avx_filler);
				}
			}

			// Process remaining elements
			for (size_t j = avx_processed_elems; j < total_elems; j++) {
				data[j] = filler;
			}	
				
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value; // Extract scalar fill value
			float* data = (float*)tensor->data; // Cast tensor data to appropriate type
			
			// Create AVX vector with 8 copies of the fill value
			__m256 avx_filler = _mm256_set1_ps(filler);
			
			size_t avx_iters = total_elems / NNL2_FLOAT32_ELEMENTS_PER_AVX256; // total_elems / 8 
			size_t avx_processed_elems = avx_iters * NNL2_FLOAT32_ELEMENTS_PER_AVX256; // avx_iters * 8
			
			if(is_aligned) {
				// Process aligned memory with optimized stores
				for (size_t i = 0; i < avx_iters; i++) {
					_mm256_store_ps(data + i * NNL2_FLOAT32_ELEMENTS_PER_AVX256, avx_filler);
				}
			} else {
				// Process unaligned memory with unaligned stores
				for (size_t i = 0; i < avx_iters; i++) { 
					_mm256_storeu_ps(data + i * NNL2_FLOAT32_ELEMENTS_PER_AVX256, avx_filler);
				}
			}

			// Process remaining elements
			for (size_t j = avx_processed_elems; j < total_elems; j++) {
				data[j] = filler;
			}
			
			break;
		}
		
		case FLOAT64: {
			double filler = *(double*)value; // Extract scalar fill value
			double* data = (double*)tensor->data; // Cast tensor data to appropriate type
			
			// Create AVX vector with 4 copies of the fill value
			__m256d avx_filler = _mm256_set1_pd(filler);
			
			size_t it = 0;
			size_t avx_limit = total_elems - (NNL2_MIN_ELEMENTS_FOR_AVX256_DOUBLE - 1);
			
			if(is_aligned) {
				// Process aligned memory with optimized stores
				for(; it < avx_limit; it += NNL2_FLOAT64_ELEMENTS_PER_AVX256) {
					_mm256_store_pd(data + it, avx_filler);
				}	
			} else {
				// Process unaligned memory with unaligned stores
				for(; it < avx_limit; it += NNL2_FLOAT64_ELEMENTS_PER_AVX256) {
					_mm256_storeu_pd(data + it, avx_filler);
				}
			}
			
			// Process remaining elements
			for(size_t j = it; j < total_elems; j++) {
				data[j] = filler;
			}
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype); // Unsupported data type error
			result = false;
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}
#endif

/** @ingroup backend_system
 ** @brief Backend implementations for inplace_fill
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_unroll_128_inplace_fill: 128-bit optimized loop unrolling implementation
 *  - nnl2_unroll_256_inplace_fill: 256-bit optimized loop unrolling implementation  
 *  - nnl2_unroll_512_inplace_fill: 512-bit optimized loop unrolling implementation
 *  - nnl2_naive_inplace_fill: Basic reference implementation
 *  - nnl2_avx256_inplace_fill: AVX-256 SIMD optimized implementation (conditionally compiled)
 *
 ** @see REGISTER_BACKEND
 ** @see UNROLL_128_BACKEND_NAME
 ** @see UNROLL_256_BACKEND_NAME
 ** @see UNROLL_512_BACKEND_NAME
 ** @see AVX512_BACKEND_NAME
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_unroll_128_inplace_fill
 ** @see nnl2_unroll_256_inplace_fill
 ** @see nnl2_unroll_512_inplace_fill
 ** @see nnl2_avx_256_inplace_fill
 ** @see nnl2_naive_inplace_fillss
 ** @see nnl2_unroll_128
 ** @see nnl2_unroll_256
 ** @see nnl2_unroll_512
 ** @see nnl2_avx256
 ** @see nnl2_naive
 **/
Implementation inplace_fill_backends[] = {
	REGISTER_BACKEND(nnl2_unroll_128_inplace_fill, nnl2_unroll_128, UNROLL_128_BACKEND_NAME),	
	REGISTER_BACKEND(nnl2_unroll_256_inplace_fill, nnl2_unroll_256, UNROLL_256_BACKEND_NAME),
	REGISTER_BACKEND(nnl2_unroll_512_inplace_fill, nnl2_unroll_512, UNROLL_512_BACKEND_NAME),
	REGISTER_BACKEND(nnl2_naive_inplace_fill, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
		#if TENSOR_MEM_ALIGNMENT == 32
			REGISTER_BACKEND(nnl2_avx256_inplace_fill, nnl2_avx256, AVX256_BACKEND_NAME),
		#endif
	#endif
};

/**
 * @brief Function pointer for inplace_fill 
 * @ingroup backend_system 
 */
fn_inplace_fill inplace_fill;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(inplace_fill);

/** 
 * @brief Sets the backend for inplace_fill
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_inplace_fill_backend(const char* backend_name) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
	    NNL2_FUNC_ENTER();
		NNL2_DEBUG("Changed backend for inplace_fill from %s to %s", CURRENT_BACKEND(inplace_fill), backend_name);	
	#endif
	
    ESET_BACKEND_BY_NAME(inplace_fill_backends, inplace_fill, backend_name, CURRENT_BACKEND(inplace_fill));
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
}

/** 
 * @brief Gets the name of the active backend for inplace_all
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_inplace_fill_backend() {
	return current_backend(inplace_fill);
}

/** 
 * @brief Function declaration for getting all `inplace_fill` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(inplace_fill);

/**
 * @brief Function declaration for getting the number of all `inplace_fill` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(inplace_fill);

/** @brief 
 * Creates a new tensor filled with ones
 *
 ** @param shape
 * Pointer to integer array specifying the dimensions of the tensor
 *
 ** @param rank 
 * Number of dimensions (length of shape array)
 *
 ** @param dtype
 * Data type of the tensor elements
 *
 ** @return
 * Pointer to the newly allocated tensor filled with ones, or NULL on failure
 *
 ** @note 
 * The returned tensor must be freed using nnl2_free_tensor() to avoid memory leaks
 *
 ** @note 
 * For optimal performance with floating-point types, ensure proper memory alignment
 *
 ** @note
 * May conduct additional checks depending on the safety level
 *
 ** @example
 * // Create a 3x3 matrix of ones with float32 type
 * int shape[] = {3, 3};
 * Tensor* ones_matrix = nnl2_ones(shape, 2, FLOAT32); 
 *
 ** @exception NNL2Error
 * Shape is NULL or rank is invalid
 *
 ** @exception NNL2Error
 * New tensor allocation fails
 *
 ** @exception NNL2Error
 * Unsupported data type is specified
 *
 ** @see nnl2_empty()
 ** @see nnl2_zeros()
 ** @see inplace_fill()
 **/
Tensor* nnl2_ones(int32_t* shape, int32_t rank, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif
	
	// Additional validation of input parameters in maximal safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (shape == NULL || rank <= 0) {
			NNL2_ERROR("Invalid shape or rank in ones");
			return NULL;
		}
	#endif
	
	// Creating an empty tensor with a specified shape and data type
    Tensor* tensor_t = nnl2_empty(shape, rank, dtype);
	
	// Checking the success of tensor creation
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor_t == NULL) {
			NNL2_ERROR("Failed to allocate empty tensor");
			return NULL;
		}
	#endif

	bool success;

	// Filling the tensor with units depending on the data type
    switch(dtype) {
        case INT32:    success = inplace_fill(tensor_t, &(int32_t){1}, dtype);   break;
        case FLOAT32:  success = inplace_fill(tensor_t, &(float){1.0f}, dtype);  break;     
        case FLOAT64:  success = inplace_fill(tensor_t, &(double){1.0}, dtype);  break;

		// Processing unsupported data types
        default: {
            NNL2_TYPE_ERROR(dtype);
            nnl2_free_tensor(tensor_t);  
            return NULL;
        }
    }
	
	if(!success) {
		NNL2_ERROR("Function completed failed");
	}
    
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
	
    return tensor_t;
}

/** @brief 
 * Creates a new tensor filled with a specified value
 *
 ** @param shape
 * Pointer to integer array specifying the dimensions of the tensor
 *
 ** @param rank 
 * Number of dimensions (length of shape array)
 *
 ** @param dtype 
 * Data type of the tensor elements
 *
 ** @param filler
 * Pointer to the value used to fill the tensor. Must match the specified data type
 * For example: for FLOAT32, pass pointer to float; for INT32, pass pointer to int32_t
 *
 ** @return 
 * Pointer to the newly allocated tensor filled with specified value, or NULL on failure
 *
 ** @note
 * The returned tensor must be freed using nnl2_free_tensor() to avoid memory leaks
 *
 ** @note
 * The filler parameter must point to a valid value of the correct type matching dtype
 *
 ** @note
 * For optimal performance with floating-point types, ensure proper memory alignment
 *
 ** @note
 * May conduct additional checks depending on the safety level configuration
 *
 ** @example
 * // Create a 2x3 matrix filled with 5.5 (float32)
 * int shape[] = {2, 3};
 * float value = 5.5f;
 * Tensor* filled_tensor = nnl2_full(shape, 2, FLOAT32, &value);
 *
 * // Create a 1D vector filled with 42 (int32)
 * int shape1d[] = {10};
 * int32_t int_value = 42;
 * Tensor* int_tensor = nnl2_full(shape1d, 1, INT32, &int_value);
 *
 ** @exception NNL2_ERROR_SHAPE_INVALID
 * Shape is NULL or contains invalid dimensions
 *
 ** @exception NNL2_ERROR_RANK_INVALID
 * Rank is zero or negative
 *
 ** @exception NNL2_ERROR_FILLER_NULL
 * Filler pointer is NULL
 *
 ** @exception NNL2_ERROR_ALLOCATION_FAILED
 * Tensor allocation fails
 *
 ** @exception NNL2_ERROR_UNSUPPORTED_TYPE
 * Unsupported data type is specified
 *
 ** @see inplace_fill()
 **/
Tensor* nnl2_full(const int* shape, int rank, TensorType dtype, void* filler) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif
	
	// Additional validation of input parameters in maximal safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX 
		if (!shape || rank <= 0 || !filler) {
			NNL2_ERROR("Invalid tensor structure in full");
			return NULL;
		}
		
		// Validate each dimension of the shape
		for (int i = 0; i < rank; i++) {
			if (shape[i] <= 0) {
				NNL2_ERROR("Invalid shape dimension at index %d: %d", i, shape[i]);
				return NULL;
			}
		}
	#endif
	
	// Create empty tensor with specified shape and data type
	Tensor* tensor_t = nnl2_empty(shape, rank, dtype);
	
	// Check if tensor creation was successful
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (!tensor_t) {
			fprintf(stderr, "Failed to allocate tensor\n");
			return NULL;
		}
	#endif
	
	// Fill the tensor with the specified value
	if(!inplace_fill(tensor_t, filler, dtype)) {
		// Error handle
		NNL2_ERROR("Function completed failed");
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
	
	return tensor_t;
}

/** @brief
 * Naive implementation of single-precision general matrix multiplication (sgemm) in-place
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @param c
 * Pointer to tensor for storing result matrix C (modified in-place)
 *
 ** @param ldc
 * Leading dimension of matrix C (stride between rows/columns)
 *
 ** @return
 * None (result is stored in-place in matrix C)
 *
 ** @note
 * This is a naive triple-loop implementation
 * Not optimized for performance - use for reference only
 *
 ** @note
 * All input tensors must be of FLOAT32 type and properly allocated
 *
 ** @note
 * Matrix dimensions must satisfy: 
 * - A: [m x k] if transa == NoTrans, [k x m] if transa == Trans
 * - B: [k x n] if transb == NoTrans, [n x k] if transb == Trans  
 * - C: [m x n]
 *
 ** @note
 * Leading dimensions must be >= corresponding matrix dimensions
 *
 ** @example
 * // Multiply two matrices: C = alpha * A * B + beta * C
 * naive_sgemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, 
 *                   m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** @exception
 * Function returns early with error message if invalid parameters are detected
 *
 **/
void naive_sgemminplace(const nnl2_order order, const nnl2_transpose transa, 
                        const nnl2_transpose transb, const int m, const int n, 
                        const int k, const float alpha, const Tensor* a, const int lda,
                        const Tensor* b, const int ldb, const float beta, Tensor* c,
                        const int ldc) {
							
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	// Checking the input data for correctness

	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (!a || !b || !c || !a->data || !b->data || !c->data) {
			NNL2_ERROR("Null pointer passed as argument (gemm)");
			return;
		}
		
		if (m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
			NNL2_ERROR("Invalid dimensions provided (gemm)");
			return;
		}
		
		int a_cols = (transa == nnl2Trans) ? m : k;
		int b_cols = (transb == nnl2Trans) ? k : n;
		
		if (lda < a_cols) {
			NNL2_ERROR("lda is less than number of columns of a matrix! (gemm)");
			return;
		}
		
		if (ldb < b_cols) {  
			NNL2_ERROR("ldb is less than number of columns of b matrix! (gemm)");
			return;
		}

		if (ldc < n) {    
			NNL2_ERROR("ldc is less than n! (gemm)");
			return;
		}
	#endif
	
	// Casting tensor data to float with volatile to prevent compiler optimizations
    volatile float* data_a = (volatile float*)a->data;
    volatile float* data_b = (volatile float*)b->data;
    volatile float* data_c = (volatile float*)c->data;                          
    
    if(order == nnl2RowMajor) {
		// Implementation for RowMajor order (lowercase data organization)
		
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
					// C[i,j] = alpha * (A * B)[i,j]
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
					// C[i,j] = alpha * (A * B)[i,j] + beta * C[i,j]
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    } else {
		// Implementation for ColumnMajor order (column-based data organization)
		
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
					// C[i,j] = alpha * (A * B)[i,j]
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
					// C[i,j] = alpha * (A * B)[i,j] + beta * C[i,j]
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#ifdef OPENBLAS_AVAILABLE
/** @brief
 * BLAS-accelerated implementation of single-precision general matrix multiplication (SGEMM) in-place
 *
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @param c
 * Pointer to tensor for storing result matrix C (modified in-place)
 *
 ** @param ldc
 * Leading dimension of matrix C (stride between rows/columns)
 *
 ** @return
 * None (result is stored in-place in matrix C)
 *
 ** @note
 * Requires OpenBLAS library to be available and linked
 * Significantly faster than naive implementation
 *
 ** @note
 * All input tensors must be of FLOAT32 type and properly allocated
 *
 ** @note
 * Performs the operation: C = alpha * op(A) * op(B) + beta * C
 * where op(X) is either X or X^T depending on transpose flags
 *
 ** @example
 * // Multiply matrices using BLAS: C = alpha * A * B + beta * C
 * blas_sgemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** cblas_sgemm()
 **/
void blas_sgemminplace(const nnl2_order order, const nnl2_transpose transa, 
                       const nnl2_transpose transb, const int m, const int n, 
                       const int k, const float alpha, const Tensor* a, const int lda,
                       const Tensor* b, const int ldb, const float beta, Tensor* c,
                       const int ldc) {

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	// Casting from void*
	float* a_data = (float*)a->data;
	float* b_data = (float*)b->data;
	float* c_data = (float*)c->data;
	
	// Convert nnl2 order enum to CBLAS order enum
	CBLAS_ORDER cblas_order;
	
	switch(order) {
		case nnl2RowMajor:
			cblas_order = CblasRowMajor;
			break;
			
		case nnl2ColMajor:
			cblas_order = CblasColMajor;
			break;
			
		default: {
			NNL2_ORDER_ERROR(order);
			return;
		}
	}
	
	// Convert nnl2 transpose flags to CBLAS transpose enums
	CBLAS_TRANSPOSE cblas_transa;
	CBLAS_TRANSPOSE cblas_transb;
	
	switch(transa) {
		case nnl2NoTrans:
			cblas_transa = CblasNoTrans;  // Use matrix A as-is (no transposition)
			break;
			
		case nnl2Trans:
			cblas_transa = CblasTrans;    // Use transpose of matrix A
			break;
			
		default: {
			NNL2_ORDER_ERROR(transa);
			return;
		}
	}
	
	switch(transb) {
		case nnl2NoTrans:
			cblas_transb = CblasNoTrans;  // Use matrix B as-is (no transposition)
			break;
			
		case nnl2Trans:
			cblas_transb = CblasTrans;    // Use transpose of matrix B
			break;
			
		default: {
			NNL2_ORDER_ERROR(transb);
			return;
		}
	}
					
	// Call the actual BLAS SGEMM function
    // This is the highly optimized matrix multiplication routine from OpenBLAS
    // Performs: C = alpha * op(A) * op(B) + beta * C				
	cblas_sgemm(cblas_order,    // Memory ordering (RowMajor/ColMajor)
				cblas_transa,   // Transpose flag for matrix A
				cblas_transb,   // Transpose flag for matrix B
				m, 				// Number of rows in matrices A and C
				n,			    // Number of columns in matrices B and C
				k, 			    // Number of columns in A and rows in B
				alpha,		    // Scalar multiplier for A*B product
				a_data, 		// Pointer to matrix A data
				lda, 			// Leading dimension of matrix A
				b_data, 	    // Pointer to matrix B data
				ldb,  			// Leading dimension of matrix B
				beta,  			// Scalar multiplier for matrix C
				c_data,		    // Pointer to matrix C data (output, modified in-place)
				ldc);     	    // Leading dimension of matrix C
				
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif			
}
#endif

/** @ingroup backend_system
 ** @brief Backend implementations for sgemminplace
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - naive_sgemminplace: A simple, unremarkable naive implementation of matrix multiplication
 *  - blas_sgemminplace: BLAS version of matrix multiplication
 *
 ** @see naive_sgemminplace
 ** @see blas_sgemminplace
 **/
Implementation sgemminplace_backends[] = {	
	REGISTER_BACKEND(naive_sgemminplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef OPENBLAS_AVAILABLE
	REGISTER_BACKEND(blas_sgemminplace, nnl2_blas, BLAS_BACKEND_NAME),
	#endif
};

/**
 * @brief Function pointer for sgemm in-place
 * @ingroup backend_system 
 */
sgemminplacefn sgemminplace;

/** 
 * @brief Sets the backend for view
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_sgemminplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sgemminplace_backends, sgemminplace, backend_name);
}

/** @brief
 * Naive implementation of double-precision general matrix multiplication (DGEMM) in-place
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @param c
 * Pointer to tensor for storing result matrix C (modified in-place)
 *
 ** @param ldc
 * Leading dimension of matrix C (stride between rows/columns)
 *
 ** @return
 * None (result is stored in-place in matrix C)
 *
 ** @note
 * This is a naive triple-loop implementation
 * Not optimized for performance - use for reference only
 *
 ** @note
 * All input tensors must be of FLOAT64 type and properly allocated
 *
 ** @note
 * Matrix dimensions must satisfy: 
 * - A: [m x k] if transa == NoTrans, [k x m] if transa == Trans
 * - B: [k x n] if transb == NoTrans, [n x k] if transb == Trans  
 * - C: [m x n]
 *
 ** @note
 * Leading dimensions must be >= corresponding matrix dimensions
 *
 ** @example
 * // Multiply two matrices: C = alpha * A * B + beta * C
 * naive_dgemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** @exception
 * Function returns early with error message if invalid parameters are detected
 *
 **/
void naive_dgemminplace(const nnl2_order order, const nnl2_transpose transa,
                        const nnl2_transpose transb, const int m, const int n,
                        const int k, const double alpha, const Tensor* a, const int lda,
                        const Tensor* b, const int ldb, const double beta, Tensor* c,
                        const int ldc) {	

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	

	// Checking the input data for correctness
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (!a || !b || !c || !a->data || !b->data || !c->data) {
			NNL2_ERROR("Null pointer passed as argument (gemm)");
			return;
		}

		if (m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
			NNL2_ERROR("Invalid dimensions provided (gemm)");
			return;
		}

		int a_cols = (transa == nnl2Trans) ? m : k;
		int b_cols = (transb == nnl2Trans) ? k : n;

		if (lda < a_cols) {
			NNL2_ERROR("lda is less than number of columns of a matrix");
			return;
		}

		if (ldb < b_cols) {
			NNL2_ERROR("ldb is less than number of columns of b matrix (gemm)");
			return;
		}

		if (ldc < n) {
			NNL2_ERROR("ldc is less than n (gemm)");
			return;
		}
	#endif

	// Casting data from void*
    volatile double* data_a = (volatile double*)a->data;
    volatile double* data_b = (volatile double*)b->data;
    volatile double* data_c = (volatile double*)c->data;

    if(order == nnl2RowMajor){
		// Implementation for RowMajor order (row-wise data organization)
		
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
					// C[i,j] = alpha * (A * B)[i,j]
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
					// C[i,j] = alpha * (A * B)[i,j] + beta * C[i,j]
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    } else {
		// Implementation for ColumnMajor order (column-wise data organization)
		
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
					// C[i,j] = alpha * (A * B)[i,j]
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
					// C[i,j] = alpha * (A * B)[i,j] + beta * C[i,j]
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
}

#ifdef OPENBLAS_AVAILABLE
/** @brief
 * BLAS-accelerated implementation of double-precision general matrix multiplication (DGEMM) in-place
 * Uses highly optimized OpenBLAS library for matrix multiplication operations
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @param c
 * Pointer to tensor for storing result matrix C (modified in-place)
 *
 ** @param ldc
 * Leading dimension of matrix C (stride between rows/columns)
 *
 ** @return
 * None (result is stored in-place in matrix C)
 *
 ** @note
 * Requires OpenBLAS library to be available and linked
 * Significantly faster than naive implementation
 *
 ** @note
 * All input tensors must be of FLOAT64 type and properly allocated
 *
 ** @note
 * Performs the operation: C = alpha * op(A) * op(B) + beta * C
 * where op(X) is either X or X^T depending on transpose flags
 *
 ** @example
 * // Multiply matrices using BLAS: C = alpha * A * B + beta * C
 * blas_dgemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** @see cblas_dgemm()
 **/
void blas_dgemminplace(const nnl2_order order, const nnl2_transpose transa, 
                       const nnl2_transpose transb, const int m, const int n, 
                       const int k, const double alpha, const Tensor* a, const int lda,
                       const Tensor* b, const int ldb, const double beta, Tensor* c,
                       const int ldc) {
						   
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif						   

	// Cast data from void* to double*
	double* a_data = (double*)a->data;
	double* b_data = (double*)b->data;
	double* c_data = (double*)c->data;
	
	// Convert nnl2 order enum to CBLAS order enum
	CBLAS_ORDER cblas_order;
	
	switch(order) {
		case nnl2RowMajor:
			cblas_order = CblasRowMajor;
			break;
			
		case nnl2ColMajor:
			cblas_order = CblasColMajor;
			break;
			
		default: {
			NNL2_ORDER_ERROR(order);
			return;
		}
	}
	
	// Convert nnl2 transpose flags to CBLAS transpose enums
	CBLAS_TRANSPOSE cblas_transa;
	CBLAS_TRANSPOSE cblas_transb;
	
	switch(transa) {
		case nnl2NoTrans:
			cblas_transa = CblasNoTrans;  // Use matrix A as-is (no transposition)
			break;
			
		case nnl2Trans:
			cblas_transa = CblasTrans;    // Use transpose of matrix A
			break;
			
		default: {
			NNL2_TRANS_ERROR(transa);
			return;
		}
	}
	
	switch(transb) {
		case nnl2NoTrans:
			cblas_transb = CblasNoTrans;  // Use matrix B as-is (no transposition)
			break;
			
		case nnl2Trans:
			cblas_transb = CblasTrans;    // Use transpose of matrix B
			break;
			
		default: {
			NNL2_TRANS_ERROR(transb);
			return;
		}
	}
	
	// Call the actual BLAS DGEMM function for double-precision matrices
    // This is the highly optimized matrix multiplication routine from OpenBLAS
    // Performs: C = alpha * op(A) * op(B) + beta * Cы					   
	cblas_dgemm(cblas_order,     // Memory ordering (RowMajor/ColMajor)
                cblas_transa,    // Transpose flag for matrix A
                cblas_transb,    // Transpose flag for matrix B
                m,               // Number of rows in matrices A and C
                n,               // Number of columns in matrices B and C
                k,               // Number of columns in A and rows in B
                alpha,           // Scalar multiplier for A*B product
                a_data,          // Pointer to matrix A data
                lda,             // Leading dimension of matrix A
                b_data,          // Pointer to matrix B data
                ldb,             // Leading dimension of matrix B
                beta,            // Scalar multiplier for matrix C
                c_data,          // Pointer to matrix C data (output, modified in-place)
                ldc);            // Leading dimension of matrix C
				
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
}
#endif

/** @ingroup backend_system
 ** @brief Backend implementations for inplace_fill
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - naive_dgemminplace: Basic reference implementation
 *  - blas_dgemminplace: BLAS-accelerated implementation 
 *
 ** @see naive_dgemminplace
 ** @see blas_dgemminplace
 **/
Implementation dgemminplace_backends[] = {
	REGISTER_BACKEND(naive_dgemminplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef OPENBLAS_AVAILABLE
	REGISTER_BACKEND(blas_dgemminplace, nnl2_blas, BLAS_BACKEND_NAME),
	#endif
};

/**
 * @brief Function pointer for dgemm in-place
 * @ingroup backend_system 
 */
dgemminplacefn dgemminplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(gemm);

/** 
 * @brief Sets the backend for view
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_dgemminplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(dgemminplace_backends, dgemminplace, backend_name, current_backend(gemm));
}

/** 
 * @brief Gets the name of the active backend for inplace_all
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_gemm_backend() {
	return current_backend(gemm);
}

/** 
 * @brief Function declaration for getting all `dgemminplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(dgemminplace);

/**
 * @brief Function declaration for getting the number of all `dgemminplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(dgemminplace);

/** @brief
 * Single-precision general matrix multiplication with automatic output allocation
 * Creates a new tensor for the result and performs matrix multiplication
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A (must be FLOAT32)
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B (must be FLOAT32)
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @return
 * Pointer to newly allocated tensor containing result matrix C, or NULL on failure
 *
 ** @note
 * Automatically allocates output tensor
 * Result must be freed using nnl2_free_tensor() to avoid memory leaks
 *
 ** @note
 * Performs: C = alpha * op(A) * op(B) + beta * C
 * where C is initialized with ones
 *
 ** @example
 * // Multiply matrices and get new result tensor
 * Tensor* result = sgemm(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta);
 *
 ** @see sgemminplace()
 **/
Tensor* sgemm(const nnl2_order order, const nnl2_transpose transa, 
			  const nnl2_transpose transb, const int m, const int n, 
			  const int k, const float alpha, const Tensor* a, const int lda,
			  const Tensor* b, const int ldb, const float beta) {
				  
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif				  
	
	// Define shape and properties for result matrix C
	int shape_c[] = {m, n}; // Result matrix dimensions: m x n
	int rank_c = 2;		    // 2D matrix
	TensorType type_c = FLOAT32;
	
	// Create output tensor
	Tensor* c = nnl2_empty(shape_c, rank_c, type_c);
	
	// Perform in-place matrix multiplication on the created tensor
	sgemminplace(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, n);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
	
	return c; // Return the newly created result tensor
}

/** @brief
 * Double-precision general matrix multiplication with automatic output allocation
 * Creates a new tensor for the result and performs matrix multiplication
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A (must be FLOAT64)
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B (must be FLOAT64)
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @return
 * Pointer to newly allocated tensor containing result matrix C, or NULL on failure
 *
 ** @note
 * Automatically allocates output tensor filled with ones
 * Result must be freed using nnl2_free_tensor() to avoid memory leaks
 *
 ** @note
 * Performs: C = alpha * op(A) * op(B) + beta * C
 * where C is initialized with ones
 *
 ** @example
 * // Multiply matrices and get new result tensor
 * Tensor* result = dgemm(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta);
 *
 ** @see dgemminplace()
 **/
Tensor* dgemm(const nnl2_order order, const nnl2_transpose transa, 
			  const nnl2_transpose transb, const int m, const int n, 
			  const int k, const double alpha, const Tensor* a, const int lda,
			  const Tensor* b, const int ldb, const double beta) {
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	
	
	// Define shape and properties for result matrix C
	int shape_c[] = {m, n}; // Result matrix dimensions: m x n
	int rank_c = 2; 		// 2D matrix
	TensorType type_c = FLOAT64;
	
	// Create output tensor
	Tensor* c = nnl2_empty(shape_c, rank_c, type_c);
	
	// Perform in-place matrix multiplication on the created tensor
	dgemminplace(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, n);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
	
	return c; // Return the newly created result tensor
}

/** @brief
 * Type-agnostic general matrix multiplication with automatic output allocation
 * Automatically detects input data type and calls appropriate precision version
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A (FLOAT32 or FLOAT64)
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B (must match A's data type)
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @return
 * Pointer to newly allocated tensor containing result matrix C, or NULL on failure
 *
 ** @note
 * Automatically determines precision based on input tensor A data type
 * Supports both single (FLOAT32) and double (FLOAT64) precision
 *
 ** @note
 * Matrices A and B must have the same data type
 * Result tensor will have the same data type as input matrices
 *
 ** @example
 * // Multiply matrices with automatic type detection
 * Tensor* result = gemm(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, 
 *                      m, n, k, alpha, A, lda, B, ldb, beta);
 *
 ** @see sgemm()
 ** @see dgemm()
 **/
Tensor* gemm(const nnl2_order order, const nnl2_transpose transa, 
			 const nnl2_transpose transb, const int m, const int n, 
		     const int k, const double alpha, const Tensor* a, const int lda,
			 const Tensor* b, const int ldb, const double beta) {
		
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	
		
	// Determine data type from input tensor A			
	TensorType dtype = a->dtype;
	
	// Dispatch to appropriate precision implementation
	switch(dtype) {
		case FLOAT64: return dgemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta);
		case FLOAT32: return sgemm(order, transa, transb, m, n, k, (const float)alpha, a, lda, b, ldb, (const float)beta);
		
		default: {
			NNL2_TYPE_ERROR(dtype);
			return NULL;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
}

/** @brief
 * Type-agnostic in-place general matrix multiplication
 * Automatically detects input data type and calls appropriate precision version
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A (FLOAT32 or FLOAT64)
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B (must match A's data type)
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @param c
 * Pointer to output tensor for storing result matrix C (modified in-place)
 *
 ** @param ldc
 * Leading dimension of matrix C (stride between rows/columns)
 *
 ** @return
 * None (result is stored in-place in matrix C)
 *
 ** @note
 * Automatically determines precision based on input tensor A data type
 * Supports both single (FLOAT32) and double (FLOAT64) precision
 *
 ** @note
 * All input tensors must have the same data type
 * Matrix C is modified in-place and must be properly allocated
 *
 ** @example
 * // In-place matrix multiplication with automatic type detection
 * gemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** @see sgemminplace(), dgemminplace()
 **/
void gemminplace(const nnl2_order order, const nnl2_transpose transa, 
				 const nnl2_transpose transb, const int m, const int n, 
				 const int k, const double alpha, const Tensor* a, const int lda,
				 const Tensor* b, const int ldb, const double beta,
				 Tensor* c, const int ldc) {

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	

	// Determine data type from input tensor A
	TensorType dtype = a->dtype;
	
	// Dispatch to appropriate precision implementation
	switch(dtype) {
		case FLOAT64: dgemminplace(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);						    break;
		case FLOAT32: sgemminplace(order, transa, transb, m, n, k, (const float)alpha, a, lda, b, ldb, (const float)beta, c, ldc);  break;
		
		default: {
			NNL2_TYPE_ERROR(dtype);
			return;
		}
	}			
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
}

/** @brief 
 * Prints the contents of a 1D tensor to standard output
 *
 ** @param tensor
 * Pointer to the 1D tensor to be printed 
 *
 ** @param full_print
 * Flag controlling output truncation:
 ** true: Print all elements regardless of tensor size
 ** false: Truncate output for large tensors (show first and last elements)
 *
 ** @param max_rows
 * Maximum number of rows to display without truncation
 *
 ** @param show_rows
 * Number of elements to show from beginning and end when truncating
 *
 ** @note
 * In safety mode, performs extensive validation of input parameters
 *
 * @note
 * Output format includes tensor metadata (type, shape) and formatted data
 *
 ** @example
 * // Print full tensor contents
 * nnl2_print_1d_tensor(my_tensor, true, 10, 3);
 *
 * // Print truncated version for large tensors  
 * nnl2_print_1d_tensor(large_tensor, false, 10, 3);
 */
void nnl2_print_1d_tensor(Tensor* tensor, bool full_print, int32_t max_rows, int32_t show_rows) {		
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif	
	
	int rows = tensor->shape[0];

	// Comprehensive input validation in safety mode
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (tensor == NULL) {
            NNL2_ERROR("NULL tensor pointer");
            return;
        }
    
        if (tensor->data == NULL) {
            NNL2_ERROR("Tensor data is NULL");
            return;
        }
        
        if (tensor->shape == NULL) {
            NNL2_ERROR("Tensor shape is NULL");
            return;
        }
        
        if (tensor->rank != 1) {  
            NNL2_ERROR("Expected 1D tensor, got %dD", tensor->rank);
            return;
        }
		
        if (rows <= 0) {
            NNL2_ERROR("Invalid tensor shape [%d]", rows);
            return;
        }
    #endif
	
	if(!full_print) {
		if(2 * show_rows >= max_rows) {
			NNL2_ERROR("show_rows (%d) Is too large for max_rows (%d). Check the correctness of the tensor formatting settings you have specified", show_rows, max_rows);
			return;
		}
	}
    
	// Get data type information for formatting	
    TensorType dtype_tensor = tensor->dtype;
    char* type_name = get_tensortype_name(dtype_tensor);
    
    // Print tensor header with metadata
    printf("#<NNL2:TENSOR/%s [%d]:", type_name, rows);
    
	// Handle output truncation for large tensors
    if (rows > max_rows && !full_print) {    
		// Calculate number of elements to skip in the middle
		int skip = rows - 2 * show_rows;
	
        switch(dtype_tensor) {
            case FLOAT64: {
                double* data_t = (double*)tensor->data;
                for(int i = 0; i < show_rows; i++) printf("\n    " NNL2_FLOAT64_FORMAT, data_t[i]);
                printf("\n    ... (%d elements skipped) ...", skip);
                for(int i = rows - show_rows; i < rows; i++) printf("\n    " NNL2_FLOAT64_FORMAT, data_t[i]);
                break;
            }
            
            case FLOAT32: {
                float* data_t = (float*)tensor->data;
                for(int i = 0; i < show_rows; i++) printf("\n    " NNL2_FLOAT32_FORMAT, data_t[i]);
                printf("\n    ... (%d elements skipped) ...", skip);
                for(int i = rows - show_rows; i < rows; i++) printf("\n    " NNL2_FLOAT32_FORMAT, data_t[i]);
                break;
            }
            
            case INT32: {
                int32_t* data_t = (int32_t*)tensor->data;
                for(int i = 0; i < show_rows; i++) printf("\n    %d", data_t[i]);
                printf("\n    ... (%d elements skipped) ...", skip);
				for(int i = rows - show_rows; i < rows; i++) printf("\n    %d", data_t[i]);
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_tensor);
                return;
            }
        }
    } else {
		// Print all elements for small tensors or when full_print is requested
        switch(dtype_tensor) {
            case FLOAT64: {
                double* data_t = (double*)tensor->data;
                for(int i = 0; i < rows; i++) 
                    printf("\n    " NNL2_FLOAT64_FORMAT, data_t[i]);
                break;
            }
            
            case FLOAT32: {
                float* data_t = (float*)tensor->data;
                for(int i = 0; i < rows; i++) 
                    printf("\n    " NNL2_FLOAT32_FORMAT, data_t[i]);
                break;
            }
            
            case INT32: {
                int32_t* data_t = (int32_t*)tensor->data;
                for(int i = 0; i < rows; i++) 
                    printf("\n    %d", data_t[i]);
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_tensor);
                return;
            }
        }
    }
    
	// Close tensor output format
    printf(">\n");
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif	
}

/** @brief 
 * Prints the contents of a 2D tensor (matrix) to standard output
 *
 ** @param tensor
 * Pointer to the 2D tensor to be printed 
 *
 ** @param full_print
 * Flag controlling output truncation:
 ** true: Print all elements regardless of tensor size
 ** false: Truncate output for large tensors (show first and last rows/columns)
 *
 ** @note
 * In safety mode, performs extensive validation of input parameters
 *
 * @note
 * Output format includes tensor metadata (type, shape) and formatted data.
 * Truncation can be applied independently to rows and columns.
 *
 ** @example
 * // Print full matrix contents
 * nnl2_print_2d_tensor(my_matrix, true, 10, 10, 3, 5);
 *
 * // Print truncated version for large matrices  
 * nnl2_print_2d_tensor(large_matrix, false, 10, 10, 3, 5);
 *
 **/
void nnl2_print_2d_tensor(Tensor* tensor, bool full_print, int32_t max_rows, int32_t max_cols, int32_t quantity_show_rows, int32_t quantity_show_cols) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

	// Comprehensive input validation in maximal safety mode
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (tensor == NULL) {
            NNL2_ERROR("NULL tensor pointer");
            return;
        }
        
        if (tensor->data == NULL) {
            NNL2_ERROR("Tensor data is NULL");
            return;
        }
        
        if (tensor->shape == NULL) {
            NNL2_ERROR("Tensor shape is NULL");
            return;
        }
        
        if (tensor->rank != 2) {
            NNL2_ERROR("Expected 2D tensor, got %dD", tensor->rank);
            return;
        }
    #endif
    
    int rows = tensor->shape[0];
    int cols = tensor->shape[1];
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (rows <= 0 || cols <= 0) {
            NNL2_ERROR("Invalid tensor shape [%d, %d]", rows, cols);
            return;
        }
    #endif
	
	if (!full_print) {
        if (rows > max_rows && 2 * quantity_show_rows >= rows) {
            NNL2_ERROR("quantity_show_rows (%d) Is too large for tensor rows (%d). Check the correctness of the tensor formatting settings you have specified", quantity_show_rows, rows);
            return;
        }
        
        if (cols > max_cols && 2 * quantity_show_cols >= cols) {
            NNL2_ERROR("quantity_show_cols (%d) Is too large for tensor columns (%d). Check the correctness of the tensor formatting settings you have specified", quantity_show_cols, cols);
            return;
        }
    }
    
    TensorType dtype_tensor = tensor->dtype;
    char* type_name = get_tensortype_name(dtype_tensor);
    
	// Prefix
    printf("#<NNL2:TENSOR/%s [%dx%d]:", type_name, rows, cols);
    
    bool truncate_rows = (rows > max_rows) && !full_print;
    bool truncate_cols = (cols > max_cols) && !full_print;
    
	// Number of rows/columns displayed before and after skipping
    int show_rows = truncate_rows ? quantity_show_rows : rows;
    int show_cols = truncate_cols ? quantity_show_cols : cols;

    switch(dtype_tensor) {
        case FLOAT64: {
            double* data_t = (double*)tensor->data;
            
            for (int i = 0; i < show_rows; i++) {
                printf("\n");
                for (int j = 0; j < show_cols; j++) {
					// Concatenate "    " with NNL2_FLOAT64_FORMAT
                    printf("    " NNL2_FLOAT64_FORMAT, data_t[i * cols + j]);
                }
				
                if (truncate_cols) {
                    printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                    for (int j = cols - show_cols; j < cols; j++) {
						// Concatenate "    " with NNL2_FLOAT64_FORMAT
                        printf("    " NNL2_FLOAT64_FORMAT, data_t[i * cols + j]);
                    }
                }
            }
            
            if (truncate_rows) {
                printf("\n    ... (%d rows skipped) ...", rows - 2 * show_rows);
                
                for (int i = rows - show_rows; i < rows; i++) {
                    printf("\n");
                    for (int j = 0; j < show_cols; j++) {
						// Concatenate "    " with NNL2_FLOAT64_FORMAT
                        printf("    " NNL2_FLOAT64_FORMAT, data_t[i * cols + j]);
                    }
					
                    if (truncate_cols) {
                        printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                        for (int j = cols - show_cols; j < cols; j++) {
							// Concatenate "    " with NNL2_FLOAT64_FORMAT
                            printf("    " NNL2_FLOAT64_FORMAT, data_t[i * cols + j]);
                        }
                    }
                }
            }
            break;
        }
        
        case FLOAT32: {
            float* data_t = (float*)tensor->data;
            
            for (int i = 0; i < show_rows; i++) {
                printf("\n");
                for (int j = 0; j < show_cols; j++) {
					// Concatenate "    " with NNL2_FLOAT32_FORMAT
                    printf("    " NNL2_FLOAT32_FORMAT, data_t[i * cols + j]);
                }
				
                if (truncate_cols) {
                    printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                    for (int j = cols - show_cols; j < cols; j++) {
						// Concatenate "    " with NNL2_FLOAT32_FORMAT
                        printf("    " NNL2_FLOAT32_FORMAT, data_t[i * cols + j]);
                    }
                }
            }
            
            if (truncate_rows) {
                printf("\n    ... (%d rows skipped) ...", rows - 2 * show_rows);
                
                for (int i = rows - show_rows; i < rows; i++) {
                    printf("\n");
                    for (int j = 0; j < show_cols; j++) {
						// Concatenate "    " with NNL2_FLOAT32_FORMAT
                        printf("    " NNL2_FLOAT32_FORMAT, data_t[i * cols + j]);
                    }
					
                    if (truncate_cols) {
                        printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                        for (int j = cols - show_cols; j < cols; j++) {
							// Concatenate "    " with NNL2_FLOAT32_FORMAT
                            printf("    " NNL2_FLOAT32_FORMAT, data_t[i * cols + j]);
                        }
                    }
                }
            }
            break;
        }
        
        case INT32: {
            int32_t* data_t = (int32_t*)tensor->data;
            
            for (int i = 0; i < show_rows; i++) {
                printf("\n");
                for (int j = 0; j < show_cols; j++) {
                    printf("    %d", data_t[i * cols + j]);
                }
				
                if (truncate_cols) {
                    printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                    for (int j = cols - show_cols; j < cols; j++) {
                        printf("    %d", data_t[i * cols + j]);
                    }
                }
            }
            
            if (truncate_rows) {
                printf("\n    ... (%d rows skipped) ...", rows - 2 * show_rows);
                
                for (int i = rows - show_rows; i < rows; i++) {
                    printf("\n");
                    for (int j = 0; j < show_cols; j++) {
                        printf("    %d", data_t[i * cols + j]);
                    }
					
                    if (truncate_cols) {
                        printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                        for (int j = cols - show_cols; j < cols; j++) {
                            printf("    %d", data_t[i * cols + j]);
                        }
                    }
                }
            }
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(dtype_tensor);
            return;
        }
    }
    
	// Closing format
    printf(">\n");
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Prints a compact representation of a tensor with its metadata
 *
 ** @param tensor
 * Pointer to the tensor to be printed (any rank)
 *
 ** @note
 * Output includes tensor data type and shape information
 * Does not print actual tensor data values, only metadata
 *
 ** @note
 * In safety mode, performs validation of input parameters and tensor structure
 * Handles tensors of any rank including scalars (rank 0)
 *
 ** @example
 * // Print tensor metadata
 * nnl2_print_huge_tensor(my_tensor);
 *
 * // Output format: #<NNL2:TENSOR/FLOAT32 [3x4x5]>
 *
 ** @see get_tensortype_name
 **/
void nnl2_print_huge_tensor(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Comprehensive input validation in safety mode
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (tensor == NULL) {
			NNL2_ERROR("Provided tensor is NULL");
			return;
		}
		
		if (tensor->rank <= 0) {
            NNL2_ERROR("Invalid tensor rank: %d", tensor->rank);
            return;
        }
		
		if (tensor->rank > 0 && tensor->shape == NULL) {
            NNL2_ERROR("Tensor shape array is NULL");
            return;
        }
	#endif
	
	// Print tensor header with library identifier
	printf("#<NNL2:TENSOR/");
	
	TensorType dtype_tensor = tensor->dtype;
    char* type_name = get_tensortype_name(dtype_tensor);
	
	printf("%s [", type_name);
	
    // Format shape dimensions as "dim1xdim2xdim3..."
    printf("%d", tensor->shape[0]);
    for (int i = 1; i < tensor->rank; i++) {
        printf("x%d", tensor->shape[i]);
    }
	
	// Close tensor output format
    printf("]>");
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Universal tensor printing function that routes to appropriate specialized printer
 *
 ** @param tensor
 * Pointer to the tensor to be printed
 *
 ** @param full_print
 * Flag controlling output detail level:
 ** true: Print all elements regardless of tensor size
 ** false: Truncate output for large tensors
 *
 ** @note
 * Automatically selects the appropriate printing function based on tensor rank:
 ** Rank 1: Uses print_1d_tensor() for vector printing
 ** Rank 2: Uses print_2d_tensor() for matrix printing  
 ** Rank 3+: Uses print_huge_tensor() for metadata-only display
 ** Invalid rank: Handles error or returns silently based on safety mode
 *
 ** @note
 * In safety mode, performs validation of input parameters and tensor structure
 *
 ** @example
 * // Print full tensor contents
 * print_tensor(my_tensor, true);
 *
 * // Print truncated version for large tensors
 * print_tensor(large_tensor, false);
 *
 ** @see nnl2_print_1d_tensor
 ** @see nnl2_print_2d_tensor  
 ** @see nnl2_print_huge_tensor
 **/
void nnl2_print_tensor(Tensor* tensor, bool full_print, int32_t max_rows, int32_t max_cols, int32_t show_rows, int32_t show_cols) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	int32_t rank = tensor->rank;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if(rank <= 0) {
			NNL2_ERROR("Invalid rank: %d. Rank should be non-negative", rank);
			return;
		}
	#else 
		 if(rank <= 0) {return;}
	#endif
	else if(rank == 1) {nnl2_print_1d_tensor(tensor, full_print, max_rows, show_rows);}
	else if(rank == 2) {nnl2_print_2d_tensor(tensor, full_print, max_rows, max_cols, show_rows, show_cols);}
	else 			   {nnl2_print_huge_tensor(tensor);}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief
 * Function for quick debugging in C without specifying 
 * all the arguments in nnl2_print_tensor
 *
 ** @param tensor
 * Tensor to print
 */
void nnl2_quick_print_tensor(Tensor* tensor) {
	nnl2_print_tensor(tensor, 
					  false, // Full print?
					  20, // Max rows to print
					  10, // Max cols to print
					  10, // Show rows non in skip 
					  5); // Show cols non in skip
}

/** @brief
 * Get rank (ndim) of  tensor
 *
 ** @param tensor 
 * Pointer to the tensor structure
 *
 ** @return 
 * The rank of the tensor (number of dimensions)
 *
 ** @note 
 * In safety mode, validates the tensor pointer before access
 *
 ** @note 
 * Returns NNL2_TENSOR_TYPE_INVALID (-1) if tensor is invalid in safety mode
 *
 * @example
 * Tensor* t = nnl2_zeros(...);
 * int32_t rank = nnl2_get_tensor_rank(t);
 * printf("Tensor rank: %d\n", rank);
 */
int32_t nnl2_get_tensor_rank(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_TENSOR(tensor, NNL2_TENSOR_TYPE_INVALID)
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor->rank;
}

/** @brief 
 * Get the data type of a tensor
 *
 ** @param tensor 
 * Pointer to the tensor structure
 *
 ** @return TensorType 
 * Data type of the tensor elements
 *
 ** @note 
 * In safety mode, validates the tensor pointer before access
 *
 ** @note Returns 
 * NNL2_TENSOR_TYPE_INVALID if tensor is invalid in safety mode
 *
 * @example
 * Tensor* t = nnl2_zeros(...);
 * TensorType dtype = nnl2_get_tensor_dtype(t);
 * if (dtype == FLOAT32) {
 *     printf("Tensor is float32 type\n");
 * }
 */
TensorType nnl2_get_tensor_dtype(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_TENSOR(tensor, NNL2_TENSOR_TYPE_INVALID)
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor->dtype;
}

/** @brief 
 * Get the shape array of a tensor
 *
 ** @param tensor 
 * Pointer to the tensor structure
 *
 ** @return 
 * Pointer to the shape array containing dimension sizes
 *
 ** @note 
 * The returned array has length equal to the tensor's rank
 *
 ** @note 
 * In safety mode, validates the tensor pointer before access
 *
 ** @warning 
 * Modifying the returned array may corrupt the tensor structure
 *
 * @example
 * Tensor* t = nnl2_zeros(...);
 * int32_t* shape = nnl2_get_tensor_shape(t);
 * printf("Tensor shape: [");
 * for (int i = 0; i < nnl2_get_tensor_rank(t); i++) {
 *     printf("%d%s", shape[i], i < get_tensor_rank(t)-1 ? ", " : "");
 * }
 * printf("]\n");
 */
int32_t* nnl2_get_tensor_shape(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_TENSOR(tensor, NNL2_TENSOR_TYPE_INVALID_RET_PNTR);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor->shape;
}

/** @brief 
 * Performs element-wise addition of two tensors (naive implementation)
 * 
 * Adds the elements of the addend tensor to the corresponding elements 
 * of the summand tensor, modifying the summand tensor in place
 *
 ** @param summand 
 * Pointer to the tensor that will be modified (receives the addition result)
 *
 ** @param addend 
 * Pointer to the tensor whose values will be added to the summand
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The addend elements are converted to the summand's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the summand tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Add b to a (a becomes a + b)
 * naive_addinplace(a, b);
 * 
 * // Now a contains 2.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_addinplace(Tensor* summand, const Tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks at the maximum safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
		
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "Addend tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "Addend tensor's data is NULL");
	#endif
	
	// Calculating the total number of elements in the summand tensor
	size_t len_summand = product(summand->shape, summand->rank);
	
	// If the tensor is empty, exit the function
	if(len_summand == 0) return;
	
	TensorType dtype_summand = summand->dtype;
	TensorType dtype_addend = addend->dtype;
	
	if(dtype_summand == dtype_addend) {
		// Handling case when the tensors have the same type
		
		switch(dtype_summand) {
			case FLOAT64: {
				volatile double* data_summand = (double*)summand->data;
				volatile double* data_addend = (double*)addend->data;
				
				// Element-wise addition
				for(size_t i = 0; i < len_summand; i++) data_summand[i] += data_addend[i];
				break;
			}
			
			case FLOAT32: {
				volatile float* data_summand = (float*)summand->data;
				volatile float* data_addend = (float*)addend->data;
				
				// Element-wise addition
				for(size_t i = 0; i < len_summand; i++) data_summand[i] += data_addend[i];	
				break;
			}
			
			case INT32: {
				volatile int32_t* data_summand = (int32_t*)summand->data;
				volatile int32_t* data_addend = (int32_t*)addend->data;
				
				// Element-wise addition
				for(size_t i = 0; i < len_summand; i++) data_summand[i] += data_addend[i];		
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return;
			}
		}
	} else {
		// Handling the case when tensors have different data types
		// Calculating the step size for accessing addend tensor elements
		size_t addend_step = get_dtype_size(dtype_addend);
		
		// Casting addend data to char* for byte access
		char* addend_data = (char*)addend->data;
		
		switch(dtype_summand) {
			case FLOAT64: {
				volatile double* data_summand = (double*)summand->data;
				
				// For each element, convert the addend element to FLOAT64 and add it
				for(size_t i = 0; i < len_summand; i++) {
					void* addend_elem = addend_data + i * addend_step;
					data_summand[i] += nnl2_convert_to_float64(addend_elem, dtype_addend);
				}
				
				break; 
			}
			
			case FLOAT32: {
				volatile float* data_summand = (float*)summand->data;
				
				// For each element, convert the addend element to FLOAT32 and add it
				for(size_t i = 0; i < len_summand; i++) {
					void* addend_elem = addend_data + i * addend_step;
					data_summand[i] += nnl2_convert_to_float32(addend_elem, dtype_addend);
				}
				
				break; 
			}
			
			case INT32: {
				volatile int32_t* data_summand = (int32_t*)summand->data;
				
				// For each element, convert the addend element to INT32 and add it
				for(size_t i = 0; i < len_summand; i++) {
					void* addend_elem = addend_data + i * addend_step;
					data_summand[i] += nnl2_convert_to_int32(addend_elem, dtype_addend);
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

#ifdef __AVX__

// Declarations

/** @brief 
 * AVX256 optimized addition for double with the same types
 *
 ** @param summand 
 * Pointer to the summand data (mutable)
 *
 ** @param addend 
 * Pointer to the addend data
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_summand 
 * Flag for aligning the summand data
 *
 ** @param aligned_addend 
 * Flag for aligning the addend data
 */
static inline void nnl2_avx_add_float64_same_type(double* summand, double* addend, size_t len, bool aligned_summand, bool aligned_addend);

/** @brief
 * AVX256 optimized addition for float with the same types
 * Documentation is identical to that of nnl2_avx_add_float64_same_type
 *
 ** @see nnl2_avx_add_float64_same_type
 **/
static inline void nnl2_avx_add_float32_same_type(float* summand, float* addend, size_t len, bool aligned_summand, bool aligned_addend);

/** @brief
 * AVX256 optimized addition for int32 with the same types
 * Documentation is identical to that of nnl2_avx_add_float64_same_type
 *
 ** @see nnl2_avx_add_float64_same_type
 **/
static inline void nnl2_avx_add_int32_same_type(int32_t* summand, int32_t* addend, size_t len, bool aligned_summand, bool aligned_addend);

/** @brief
 * AVX256 optimized addition for double with different types
 *
 ** @param summand 
 * Pointer to the summand tensor data (mutable)
 * 
 ** @param addend 
 * Pointer to the addend tensor data (may be of a different type)
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_summand 
 * Flag for aligning the summand data
 */
static inline void nnl2_avx_add_float64_diff_type(double* summand, const Tensor* addend, size_t len, bool aligned_summand);

/** @brief
 * AVX256 optimized addition for float with different types
 * Documentation is identical to that of nnl2_avx_add_float64_diff_type
 *
 ** @see nnl2_avx_add_float64_diff_type
 **/
static inline void nnl2_avx_add_float32_diff_type(float* summand, const Tensor* addend, size_t len, bool aligned_summand);

/** @brief
 * AVX256 optimized addition for int32 with different types
 * Documentation is identical to that of nnl2_avx_add_float64_diff_type
 *
 ** @see nnl2_avx_add_float64_diff_type
 **/
static inline void nnl2_avx_add_int32_diff_type(int32_t* summand, const Tensor* addend, size_t len, bool aligned_summand);

// Main function

/** @brief
 * AVX256-optimized in-place addition operation 
 *
 ** @param summand
 * A tensor to which values are added (mutable)
 *
 ** @param addend
 * The tensor whose values are being added
 *
 ** @note
 * Additional checks may be performed depending on the safety level
 *
 ** @note
 * Supports type conversion
 *
 ** @note
 * Tensors can be either memory-aligned or non-memory-aligned
 *
 ** @warning
 * if the tensors are not memory-aligned, the calculations may be slightly slower
 *
 ** @see nnl2_avx_add_float64_same_type
 ** @see nnl2_avx_add_float32_same_type
 ** @see nnl2_avx_add_int32_same_type
 **
 ** @see nnl2_avx_add_float64_diff_type
 ** @see nnl2_avx_add_float32_diff_type
 ** @see nnl2_avx_add_int32_diff_type
 **/
void nnl2_avx256_addinplace(Tensor* summand, const Tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks at the maximum safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX  
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "Addend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "Addend tensor's data is NULL");
    #endif
	
	// Calculating the total number of elements in the summand tensor
    size_t len_summand = product(summand->shape, summand->rank);	
	
	// If the tensor is empty, exit the function
    if(len_summand == 0) return;
    
    TensorType dtype_summand = summand->dtype;
	TensorType dtype_addend = addend->dtype;
	
	bool is_aligned_summand = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32);
	bool is_aligned_addend = NNL2_IS_ALIGNED(addend->data, NNL2_TENSOR_ALIGNMENT_32);
	
	// Warning for unaligned memory in safety modes (performance impact)
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MINI
		if(!is_aligned_summand) {
			NNL2_WARN("In the avx256 implementation of add in-place, summand memory is not aligned to 32 bytes. Calculations may be slightly slower");
		}
		
		if(!is_aligned_addend && dtype_summand == dtype_addend) {
            NNL2_WARN("In the avx256 implementation of add in-place, addend memory is not aligned to 32 bytes. Calculations may be slightly slower");
        }
	#endif
	
	if(dtype_summand == dtype_addend) {
		// Handling case when the tensors have the same type
		
		switch (dtype_summand) {
			case FLOAT64: nnl2_avx_add_float64_same_type((double*)summand->data, (double*)addend->data, len_summand, is_aligned_summand, is_aligned_addend);  break;
			case FLOAT32: nnl2_avx_add_float32_same_type((float*)summand->data, (float*)addend->data, len_summand, is_aligned_summand, is_aligned_addend);    break;	
			case INT32:   nnl2_avx_add_int32_same_type((int32_t*)summand->data, (int32_t*)addend->data, len_summand, is_aligned_summand, is_aligned_addend);  break;
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return;
			}
		}
	} else {
		// Handling the case when tensors have different data types

        switch(dtype_summand) {
            case FLOAT64: nnl2_avx_add_float64_diff_type((double*)summand->data, addend, len_summand, is_aligned_summand);  break;
            case FLOAT32: nnl2_avx_add_float32_diff_type((float*)summand->data, addend, len_summand, is_aligned_summand);   break;
            case INT32:   nnl2_avx_add_int32_diff_type((int32_t*)summand->data, addend, len_summand, is_aligned_summand);   break;
            
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

// Implementations of auxiliary functions for the same type

/** @brief 
 * Implementation of double addition with the same types
 *
 ** @details 
 * Handles 4 combinations of memory alignment:
 * - The summand and addend tensors are aligned in memory
 * - The summand tensor is aligned in memory, but the addend is not
 * - The addend tensor is aligned in memory, but the summand is not
 * - The summand and addend tensors are not aligned in memory
 *
 ** @see nnl2_avx256_addinplace
 **/
static inline void nnl2_avx_add_float64_same_type(double* summand, double* addend, size_t len, bool aligned_summand, bool aligned_addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t i = 0;
    
	// Case 1: Both tensors are aligned 
    if(aligned_summand && aligned_addend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_load_pd(&summand[i]); 	    // Fast loading of aligned data
            __m256d v_addend = _mm256_load_pd(&addend[i]);		    // Fast loading of aligned data
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);  // Vector addition
            _mm256_store_pd(&summand[i], v_result);				    // Fast saving to aligned memory
        }
    } 
	
	// Case 2: Only the summand is aligned
	else if(aligned_summand) {
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_load_pd(&summand[i]);	    // Fast loading of aligned data
            __m256d v_addend = _mm256_loadu_pd(&addend[i]);			// Slow loading of unaligned data
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);  // Vector addition
            _mm256_store_pd(&summand[i], v_result);					// Fast saving to aligned memory
        }
    } 
	
	// Case 3: Only the addend is aligned
	else if(aligned_addend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_loadu_pd(&summand[i]);       // Slow loading of unaligned data
            __m256d v_addend = _mm256_load_pd(&addend[i]);			// Fast loading of aligned data
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);  // Vector addition
            _mm256_storeu_pd(&summand[i], v_result);				// Slow saving to unaligned memoty
        }
    } 
	
	// Case 4: Both tensors are not aligned
	else {
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_loadu_pd(&summand[i]);	    // Slow loading of unaligned data
            __m256d v_addend = _mm256_loadu_pd(&addend[i]);		    // Slow loading of unaligned data
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);  // Vector addition
            _mm256_storeu_pd(&summand[i], v_result);				// Slow saving to unaligned memoty
        }
    }
    
	// Processing the remainder
    for(; i < len; i++) summand[i] += addend[i];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of float addition with the same types
 * Similar to double, but processes 8 elements per iteration
 *
 ** @see nnl2_avx256_addinplace
 ** @see nnl2_avx_add_float64_same_type
 **/
static inline void nnl2_avx_add_float32_same_type(float* summand, float* addend, size_t len, bool aligned_summand, bool aligned_addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t i = 0;
    
	// Case 1: Both tensors are aligned 
    if(aligned_summand && aligned_addend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_load_ps(&summand[i]);        // Fast loading of aligned data
            __m256 v_addend = _mm256_load_ps(&addend[i]);		   // Fast loading of aligned data
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);  // Vector addition
            _mm256_store_ps(&summand[i], v_result);				   // Fast saving to aligned memory
        }
    } 
	
	// Case 2: Only the summand is aligned
	else if(aligned_summand) {
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_load_ps(&summand[i]);		   // Fast loading of aligned data
            __m256 v_addend = _mm256_loadu_ps(&addend[i]);		   // Slow loading of unaligned data
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);  // Vector addition
            _mm256_store_ps(&summand[i], v_result);			       // Fast saving to aligned memory
        }
    } 
	
	// Case 3: Only the addend is aligned
	else if(aligned_addend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_loadu_ps(&summand[i]);	   // Slow loading of unaligned data
            __m256 v_addend = _mm256_load_ps(&addend[i]);		   // Fast loading of aligned data
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);  // Vector addition
            _mm256_storeu_ps(&summand[i], v_result);			   // Slow saving to unaligned memoty
        }
    } 
	
	// Case 4: Both tensors are not aligned
	else {
        for(; i + 7 < len; i += 8) {	
            __m256 v_summand = _mm256_loadu_ps(&summand[i]);	   // Slow loading of unaligned data
            __m256 v_addend = _mm256_loadu_ps(&addend[i]);		   // Slow loading of unaligned data
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);  // Vector addition
            _mm256_storeu_ps(&summand[i], v_result);			   // Slow saving to unaligned memoty
        }
    }
    
	// Processing the remainder
    for(; i < len; i++) summand[i] += addend[i];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of int32 addition with the same types
 *
 ** @see nnl2_avx256_addinplace
 ** @see nnl2_avx_add_float64_same_type
 ** @see nnl2_avx_add_float32_same_type
 **/
static inline void nnl2_avx_add_int32_same_type(int32_t* summand, int32_t* addend, size_t len, bool aligned_summand, bool aligned_addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t i = 0;
    
	// Case 1: Both tensors are aligned 
    if(aligned_summand && aligned_addend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand[i]);  // Fast loading of aligned data
            __m256i v_addend = _mm256_load_si256((__m256i*)&addend[i]);	   // Fast loading of aligned data
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);	   // Vector addition
            _mm256_store_si256((__m256i*)&summand[i], v_result);		   // Fast saving to aligned memory
        }
    } 
	
	// Case 2: Only the summand is aligned
	else if(aligned_summand) {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand[i]);  // Fast loading of aligned data
            __m256i v_addend = _mm256_loadu_si256((__m256i*)&addend[i]);   // Slow loading of unaligned data
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);	   // Vector addition
            _mm256_store_si256((__m256i*)&summand[i], v_result);		   // Fast saving to aligned memory
        }
    } 
	
	// Case 3: Only the addend is aligned
	else if(aligned_addend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand[i]);  // Slow loading of unaligned data
            __m256i v_addend = _mm256_load_si256((__m256i*)&addend[i]);     // Fast loading of aligned data
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);	    // Vector addition
            _mm256_storeu_si256((__m256i*)&summand[i], v_result);			// Slow saving to unaligned memoty
        }
    } 
	
	// Case 4: Both tensors are not aligned
	else {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand[i]);  // Slow loading of unaligned data
            __m256i v_addend = _mm256_loadu_si256((__m256i*)&addend[i]);    // Slow loading of unaligned data
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);		// Vector addition
            _mm256_storeu_si256((__m256i*)&summand[i], v_result);		    // Slow saving to unaligned memoty
        }
    }
    
	// Processing the remainder
    for(; i < len; i++) summand[i] += addend[i];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

// implementations of auxiliary functions for different types

/** @brief 
 * Implementation of double addition with conversion from other types
 *
 ** @details 
 * Converts addend elements to double before addition
 *
 ** @see nnl2_avx256_addinplace
 **/
static inline void nnl2_avx_add_float64_diff_type(double* summand, const Tensor* addend, size_t len, bool aligned_summand) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	// Calculating the step between elements in bytes (for accessing raw data)
    size_t addend_step = get_dtype_size(addend->dtype);
    char* addend_data = (char*)addend->data;
    
    size_t i = 0;
    
	// Vector processing of 4 elements per iteration
    if(aligned_summand) {
        for(; i + 3 < len; i += 4) {
			// Loading 4 double from summand
            __m256d v_summand = _mm256_load_pd(&summand[i]);
			
			// Conversion and creation of a vector of 4 doubles
			// _mm256_set_pd fills the vector in reverse order (from oldest to youngest)
            __m256d v_addend = _mm256_set_pd(
                nnl2_convert_to_float64(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
			// Vector addition and saving the result
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);
            _mm256_store_pd(&summand[i], v_result);
        }
    } else {
		// Similarly, but with unaligned memory
		
        for(; i + 3 < len; i += 4) {
            __m256d v_summand = _mm256_loadu_pd(&summand[i]);
            __m256d v_addend = _mm256_set_pd(
                nnl2_convert_to_float64(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_float64(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256d v_result = _mm256_add_pd(v_summand, v_addend);
            _mm256_storeu_pd(&summand[i], v_result);
        }
    }
    
	// Scalar processing of the remaining elements
    for(; i < len; i++) {
        void* addend_elem = addend_data + i * addend_step;
        summand[i] += nnl2_convert_to_float64(addend_elem, addend->dtype);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of float addition with conversion from other types
 *
 ** @details 
 * Converts addend elements to float before addition
 *
 ** @see nnl2_avx256_addinplace
 **/
static inline void nnl2_avx_add_float32_diff_type(float* summand, const Tensor* addend, size_t len, bool aligned_summand) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t addend_step = get_dtype_size(addend->dtype);
    char* addend_data = (char*)addend->data;
    
    size_t i = 0;
    
	// Vector processing of 8 elements per iteration
    if(aligned_summand) {
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_load_ps(&summand[i]);
			
			// Creating a vector of 8 floats with conversion
			// _mm256_set_ps fills in reverse order
            __m256 v_addend = _mm256_set_ps(
                nnl2_convert_to_float32(addend_data + (i + 7) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 6) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 5) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 4) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);
            _mm256_store_ps(&summand[i], v_result);
        }
    } else {
		// Similarly for unaligned memory
        for(; i + 7 < len; i += 8) {
            __m256 v_summand = _mm256_loadu_ps(&summand[i]);
            __m256 v_addend = _mm256_set_ps(
                nnl2_convert_to_float32(addend_data + (i + 7) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 6) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 5) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 4) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_float32(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256 v_result = _mm256_add_ps(v_summand, v_addend);
            _mm256_storeu_ps(&summand[i], v_result);
        }
    }
    
	// Scalar processing of the remainder
    for(; i < len; i++) {
        void* addend_elem = addend_data + i * addend_step;
        summand[i] += nnl2_convert_to_float32(addend_elem, addend->dtype);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}	

/** @brief 
 * Implementation of int32 addition with conversion from other types
 *
 ** @details 
 * Converts addend elements to int32 before addition
 *
 ** @see nnl2_avx256_addinplace
 **/
static inline void nnl2_avx_add_int32_diff_type(int32_t* summand, const Tensor* addend, size_t len, bool aligned_summand) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    size_t addend_step = get_dtype_size(addend->dtype);
    char* addend_data = (char*)addend->data;
    
    size_t i = 0;
    
	// Vector processing of 8 elements per iteration
    if(aligned_summand) {
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_load_si256((__m256i*)&summand[i]);
			
			// Creating a vector of 8 int32s with conversion
            __m256i v_addend = _mm256_set_epi32(
                nnl2_convert_to_int32(addend_data + (i + 7) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 6) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 5) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 4) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);
            _mm256_store_si256((__m256i*)&summand[i], v_result);
        }
    } else {
		// Similarly for unaligned memory
        for(; i + 7 < len; i += 8) {
            __m256i v_summand = _mm256_loadu_si256((__m256i*)&summand[i]);
            __m256i v_addend = _mm256_set_epi32(
                nnl2_convert_to_int32(addend_data + (i + 7) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 6) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 5) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 4) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 3) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 2) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 1) * addend_step, addend->dtype),
                nnl2_convert_to_int32(addend_data + (i + 0) * addend_step, addend->dtype)
            );
			
            __m256i v_result = _mm256_add_epi32(v_summand, v_addend);
            _mm256_storeu_si256((__m256i*)&summand[i], v_result);
        }
    }
    
	// Scalar processing of the remainder
    for(; i < len; i++) {
        void* addend_elem = addend_data + i * addend_step;
        summand[i] += nnl2_convert_to_int32(addend_elem, addend->dtype);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif

#ifdef OPENBLAS_AVAILABLE
void nnl2_blas_addinplace(Tensor* summand, Tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "Addend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "Addend tensor's data is NULL");
    #endif
	
	// Calculating the total number of elements in the summand tensor
    size_t len_summand = product(summand->shape, summand->rank);
	
	// If the tensor is empty, exit the function
    if(len_summand == 0) return;
	
	TensorType dtype_summand = summand->dtype;
    TensorType dtype_addend = addend->dtype;
	
	if(dtype_summand == dtype_addend) {
        // Handling case when the tensors have the same type
		
		switch(dtype_summand) {
            case FLOAT64: {
                double* data_summand = (double*)summand->data;
                double* data_addend = (double*)addend->data;
				
				cblas_daxpy(len_summand, 1.0, data_addend, 1, data_summand, 1);
				break;
			}
			
			case FLOAT32: {
                float* data_summand = (float*)summand->data;
                float* data_addend = (float*)addend->data;
				
				cblas_saxpy(len_summand, 1.0, data_addend, 1, data_summand, 1);
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return;
			}
		}
	}
	
	else {
		// No yet...
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}
#endif

/** @ingroup backend_system
 ** @brief Backend implementations for add in-place
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_addinplace: Basic reference implementation
 *  - nnl2_avx256_addinplace: AVX256 implementation 
 *
 ** @see nnl2_naive_addinplace
 ** @see nnl2_avx256_addinplace
 **/
Implementation addinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_addinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(__AVX__) && TENSOR_MEM_ALIGNMENT == 32
		REGISTER_BACKEND(nnl2_avx256_addinplace, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
	
	#ifdef OPENBLAS_AVAILABLE
		REGISTER_BACKEND(nnl2_blas_addinplace, nnl2_blas, BLAS_BACKEND_NAME),
	#endif	
};

/**
 * @brief Function pointer for add in-place
 * @ingroup backend_systsem 
 */
addinplacefn addinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(addinplace);

/** 
 * @brief Sets the backend for add in-place
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_addinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(addinplace_backends, addinplace, backend_name, current_backend(addinplace));
}

/** 
 * @brief Gets the name of the active backend for add in-place
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_addinplace_backend() {
	return current_backend(addinplace);
}

/** 
 * @brief Function declaration for getting all `addinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(addinplace);

/**
 * @brief Function declaration for getting the number of all `dgemminplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(addinplace);

/** @brief 
 * Performs element-wise subtraction of two tensors (naive implementation)
 * 
 * Subtracts the elements of the subtrahend tensor from the corresponding elements 
 * of the minuend tensor, modifying the minuend tensor in place
 *
 ** @param minuend 
 * Pointer to the tensor that will be modified (receives the subtraction result)
 *
 ** @param subtrahend 
 * Pointer to the tensor whose values will be subtracted from the minuend
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The subtrahend elements are converted to the minuend's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the minuend tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Subtract b from a (a becomes a - b)
 * naive_subinplace(a, b);
 * 
 * // Now a contains 0.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_subinplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data, "Minuend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data, "Subtrahend tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the minuend tensor
    size_t len_minuend = product(minuend->shape, minuend->rank);
    
    // If the tensor is empty, exit the function
    if(len_minuend == 0) return;
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
    
    if(dtype_minuend == dtype_subtrahend) {
        // Handling case when the tensors have the same type
        
        switch(dtype_minuend) {
            case FLOAT64: {
                volatile double* data_minuend = (double*)minuend->data;
                volatile double* data_subtrahend = (double*)subtrahend->data;
                
                // Element-wise subtraction
                for(size_t i = 0; i < len_minuend; i++) data_minuend[i] -= data_subtrahend[i];
                break;
            }
            
            case FLOAT32: {
                volatile float* data_minuend = (float*)minuend->data;
                volatile float* data_subtrahend = (float*)subtrahend->data;
                
                // Element-wise subtraction
                for(size_t i = 0; i < len_minuend; i++) data_minuend[i] -= data_subtrahend[i];    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_minuend = (int32_t*)minuend->data;
                volatile int32_t* data_subtrahend = (int32_t*)subtrahend->data;
                
                // Element-wise subtraction
                for(size_t i = 0; i < len_minuend; i++) data_minuend[i] -= data_subtrahend[i];        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_minuend);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing subtrahend tensor elements
        size_t subtrahend_step = get_dtype_size(dtype_subtrahend);
        
        // Casting subtrahend data to char* for byte access
        char* subtrahend_data = (char*)subtrahend->data;
        
        switch(dtype_minuend) {
            case FLOAT64: {
                volatile double* data_minuend = (double*)minuend->data;
                
                // For each element, convert the subtrahend element to FLOAT64 and subtract it
                for(size_t i = 0; i < len_minuend; i++) {
                    void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
                    data_minuend[i] -= nnl2_convert_to_float64(subtrahend_elem, dtype_subtrahend);
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_minuend = (float*)minuend->data;
                
                // For each element, convert the subtrahend element to FLOAT32 and subtract it
                for(size_t i = 0; i < len_minuend; i++) {
                    void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
                    data_minuend[i] -= nnl2_convert_to_float32(subtrahend_elem, dtype_subtrahend);
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_minuend = (int32_t*)minuend->data;
                
                // For each element, convert the subtrahend element to INT32 and subtract it
                for(size_t i = 0; i < len_minuend; i++) {
                    void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
                    data_minuend[i] -= nnl2_convert_to_int32(subtrahend_elem, dtype_subtrahend);
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_minuend);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#ifdef __AVX__

// Declarations

/** @brief 
 * AVX256 optimized subtraction for double with the same types
 *
 ** @param minuend 
 * Pointer to the minuend data (mutable)
 *
 ** @param subtrahend 
 * Pointer to the subtrahend data
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_minuend 
 * Flag for aligning the minuend data
 *
 ** @param aligned_subtrahend 
 * Flag for aligning the subtrahend data
 */
static inline void nnl2_avx_sub_float64_same_type(double* minuend, double* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend);

/** @brief
 * AVX256 optimized subtraction for float with the same types
 * Documentation is identical to that of nnl2_avx_sub_float64_same_type
 *
 ** @see nnl2_avx_sub_float64_same_type
 **/
static inline void nnl2_avx_sub_float32_same_type(float* minuend, float* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend);

/** @brief
 * AVX256 optimized subtraction for int32 with the same types
 * Documentation is identical to that of nnl2_avx_sub_float64_same_type
 *
 ** @see nnl2_avx_sub_float64_same_type
 **/
static inline void nnl2_avx_sub_int32_same_type(int32_t* minuend, int32_t* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend);

/** @brief
 * AVX256 optimized subtraction for double with different types
 *
 ** @param minuend 
 * Pointer to the minuend tensor data (mutable)
 * 
 ** @param subtrahend 
 * Pointer to the subtrahend tensor data (may be of a different type)
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_minuend 
 * Flag for aligning the minuend data
 */
static inline void nnl2_avx_sub_float64_diff_type(double* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend);

/** @brief
 * AVX256 optimized subtraction for float with different types
 * Documentation is identical to that of nnl2_avx_sub_float64_diff_type
 *
 ** @see nnl2_avx_sub_float64_diff_type
 **/
static inline void nnl2_avx_sub_float32_diff_type(float* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend);

/** @brief
 * AVX256 optimized subtraction for int32 with different types
 * Documentation is identical to that of nnl2_avx_sub_float64_diff_type
 *
 ** @see nnl2_avx_sub_float64_diff_type
 **/
static inline void nnl2_avx_sub_int32_diff_type(int32_t* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend);

// Main function

/** @brief
 * AVX256-optimized in-place subtraction operation 
 *
 ** @param minuend
 * A tensor from which values are subtracted (mutable)
 *
 ** @param subtrahend
 * The tensor whose values are being subtracted
 *
 ** @note
 * Additional checks may be performed depending on the safety level
 *
 ** @note
 * Supports type conversion
 *
 ** @note
 * Tensors can be either memory-aligned or non-memory-aligned
 *
 ** @warning
 * if the tensors are not memory-aligned, the calculations may be slightly slower
 *
 ** @see nnl2_avx_sub_float64_same_type
 ** @see nnl2_avx_sub_float32_same_type
 ** @see nnl2_avx_sub_int32_same_type
 **
 ** @see nnl2_avx_sub_float64_diff_type
 ** @see nnl2_avx_sub_float32_diff_type
 ** @see nnl2_avx_sub_int32_diff_type
 **/
void nnl2_avx256_subinplace(Tensor* minuend, const Tensor* subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "Minuend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data, "Minuend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "Subtrahend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data, "Subtrahend tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the minuend tensor
    size_t len_minuend = product(minuend->shape, minuend->rank);    
    
    // If the tensor is empty, exit the function
    if(len_minuend == 0) return;
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
    
    bool is_aligned_minuend = NNL2_IS_ALIGNED(minuend->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_subtrahend = NNL2_IS_ALIGNED(subtrahend->data, NNL2_TENSOR_ALIGNMENT_32);
    
    // Warning for unaligned memory in safety modes (performance impact)
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MINI
        if(!is_aligned_minuend) {
            NNL2_WARN("In the avx256 implementation of sub in-place, minuend memory is not aligned to 32 bytes. Calculations may be slightly slower");
        }
        
        if(!is_aligned_subtrahend && dtype_minuend == dtype_subtrahend) {
            NNL2_WARN("In the avx256 implementation of sub in-place, subtrahend memory is not aligned to 32 bytes. Calculations may be slightly slower");
        }
    #endif
    
    if(dtype_minuend == dtype_subtrahend) {
        // Handling case when the tensors have the same type
        
        switch (dtype_minuend) {
            case FLOAT64: nnl2_avx_sub_float64_same_type((double*)minuend->data, (double*)subtrahend->data, len_minuend, is_aligned_minuend, is_aligned_subtrahend);  break;
            case FLOAT32: nnl2_avx_sub_float32_same_type((float*)minuend->data, (float*)subtrahend->data, len_minuend, is_aligned_minuend, is_aligned_subtrahend);    break;    
            case INT32:   nnl2_avx_sub_int32_same_type((int32_t*)minuend->data, (int32_t*)subtrahend->data, len_minuend, is_aligned_minuend, is_aligned_subtrahend);  break;
            
            default: {
                NNL2_TYPE_ERROR(dtype_minuend);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types

        switch(dtype_minuend) {
            case FLOAT64: nnl2_avx_sub_float64_diff_type((double*)minuend->data, subtrahend, len_minuend, is_aligned_minuend);  break;
            case FLOAT32: nnl2_avx_sub_float32_diff_type((float*)minuend->data, subtrahend, len_minuend, is_aligned_minuend);   break;
            case INT32:   nnl2_avx_sub_int32_diff_type((int32_t*)minuend->data, subtrahend, len_minuend, is_aligned_minuend);   break;
            
            default: {
                NNL2_TYPE_ERROR(dtype_minuend);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

// Implementations of auxiliary functions for the same type

/** @brief 
 * Implementation of double subtraction with the same types
 *
 ** @details 
 * Handles 4 combinations of memory alignment:
 * - The minuend and subtrahend tensors are aligned in memory
 * - The minuend tensor is aligned in memory, but the subtrahend is not
 * - The subtrahend tensor is aligned in memory, but the minuend is not
 * - The minuend and subtrahend tensors are not aligned in memory
 *
 ** @see nnl2_avx256_subinplace
 **/
static inline void nnl2_avx_sub_float64_same_type(double* minuend, double* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_minuend && aligned_subtrahend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);        // Fast loading of aligned data
            __m256d v_subtrahend = _mm256_load_pd(&subtrahend[i]);  // Fast loading of aligned data
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_store_pd(&minuend[i], v_result);                 // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only the minuend is aligned
    else if(aligned_minuend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);            // Fast loading of aligned data
            __m256d v_subtrahend = _mm256_loadu_pd(&subtrahend[i]);     // Slow loading of unaligned data
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_store_pd(&minuend[i], v_result);                     // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only the subtrahend is aligned
    else if(aligned_subtrahend) {
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);           // Slow loading of unaligned data
            __m256d v_subtrahend = _mm256_load_pd(&subtrahend[i]);      // Fast loading of aligned data
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_storeu_pd(&minuend[i], v_result);                    // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);           // Slow loading of unaligned data
            __m256d v_subtrahend = _mm256_loadu_pd(&subtrahend[i]);     // Slow loading of unaligned data
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_storeu_pd(&minuend[i], v_result);                    // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) minuend[i] -= subtrahend[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of float subtraction with the same types
 * Similar to double, but processes 8 elements per iteration
 *
 ** @see nnl2_avx256_subinplace
 ** @see nnl2_avx_sub_float64_same_type
 **/
static inline void nnl2_avx_sub_float32_same_type(float* minuend, float* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_minuend && aligned_subtrahend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);         // Fast loading of aligned data
            __m256 v_subtrahend = _mm256_load_ps(&subtrahend[i]);   // Fast loading of aligned data
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);  // Vector subtraction
            _mm256_store_ps(&minuend[i], v_result);                 // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only the minuend is aligned
    else if(aligned_minuend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);             // Fast loading of aligned data
            __m256 v_subtrahend = _mm256_loadu_ps(&subtrahend[i]);      // Slow loading of unaligned data
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);   // Vector subtraction
            _mm256_store_ps(&minuend[i], v_result);                     // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only the subtrahend is aligned
    else if(aligned_subtrahend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);            // Slow loading of unaligned data
            __m256 v_subtrahend = _mm256_load_ps(&subtrahend[i]);       // Fast loading of aligned data
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);   // Vector subtraction
            _mm256_storeu_ps(&minuend[i], v_result);                    // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {    
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);            // Slow loading of unaligned data
            __m256 v_subtrahend = _mm256_loadu_ps(&subtrahend[i]);      // Slow loading of unaligned data
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);   // Vector subtraction
            _mm256_storeu_ps(&minuend[i], v_result);                    // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) minuend[i] -= subtrahend[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of int32 subtraction with the same types
 *
 ** @see nnl2_avx256_subinplace
 ** @see nnl2_avx_sub_float64_same_type
 ** @see nnl2_avx_sub_float32_same_type
 **/
static inline void nnl2_avx_sub_int32_same_type(int32_t* minuend, int32_t* subtrahend, size_t len, bool aligned_minuend, bool aligned_subtrahend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_minuend && aligned_subtrahend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);       // Fast loading of aligned data
            __m256i v_subtrahend = _mm256_load_si256((__m256i*)&subtrahend[i]); // Fast loading of aligned data
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);       // Vector subtraction
            _mm256_store_si256((__m256i*)&minuend[i], v_result);                // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only the minuend is aligned
    else if(aligned_minuend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);           // Fast loading of aligned data
            __m256i v_subtrahend = _mm256_loadu_si256((__m256i*)&subtrahend[i]);    // Slow loading of unaligned data
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);           // Vector subtraction
            _mm256_store_si256((__m256i*)&minuend[i], v_result);                    // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only the subtrahend is aligned
    else if(aligned_subtrahend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);          // Slow loading of unaligned data
            __m256i v_subtrahend = _mm256_load_si256((__m256i*)&subtrahend[i]);     // Fast loading of aligned data
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);           // Vector subtraction
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);                   // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);          // Slow loading of unaligned data
            __m256i v_subtrahend = _mm256_loadu_si256((__m256i*)&subtrahend[i]);    // Slow loading of unaligned data
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);           // Vector subtraction
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);                   // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) minuend[i] -= subtrahend[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

// implementations of auxiliary functions for different types

/** @brief 
 * Implementation of double subtraction with conversion from other types
 *
 ** @details 
 * Converts subtrahend elements to double before subtraction
 *
 ** @see nnl2_avx256_subinplace
 **/
static inline void nnl2_avx_sub_float64_diff_type(double* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculating the step between elements in bytes (for accessing raw data)
    size_t subtrahend_step = get_dtype_size(subtrahend->dtype);
    char* subtrahend_data = (char*)subtrahend->data;
    
    size_t i = 0;
    
    // Vector processing of 4 elements per iteration
    if(aligned_minuend) {
        for(; i + 3 < len; i += 4) {
            // Loading 4 double from minuend
            __m256d v_minuend = _mm256_load_pd(&minuend[i]);
            
            // Conversion and creation of a vector of 4 doubles
            // _mm256_set_pd fills the vector in reverse order (from oldest to youngest)
            __m256d v_subtrahend = _mm256_set_pd(
                nnl2_convert_to_float64(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            // Vector subtraction and saving the result
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_store_pd(&minuend[i], v_result);
        }
    } else {
        // Similarly, but with unaligned memory
        
        for(; i + 3 < len; i += 4) {
            __m256d v_minuend = _mm256_loadu_pd(&minuend[i]);
            __m256d v_subtrahend = _mm256_set_pd(
                nnl2_convert_to_float64(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float64(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256d v_result = _mm256_sub_pd(v_minuend, v_subtrahend);
            _mm256_storeu_pd(&minuend[i], v_result);
        }
    }
    
    // Scalar processing of the remaining elements
    for(; i < len; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_float64(subtrahend_elem, subtrahend->dtype);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Implementation of float subtraction with conversion from other types
 *
 ** @details 
 * Converts subtrahend elements to float before subtraction
 *
 ** @see nnl2_avx256_subinplace
 **/
static inline void nnl2_avx_sub_float32_diff_type(float* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t subtrahend_step = get_dtype_size(subtrahend->dtype);
    char* subtrahend_data = (char*)subtrahend->data;
    
    size_t i = 0;
    
    // Vector processing of 8 elements per iteration
    if(aligned_minuend) {
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_load_ps(&minuend[i]);
            
            // Creating a vector of 8 floats with conversion
            // _mm256_set_ps fills in reverse order
            __m256 v_subtrahend = _mm256_set_ps(
                nnl2_convert_to_float32(subtrahend_data + (i + 7) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 6) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 5) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 4) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_store_ps(&minuend[i], v_result);
        }
    } else {
        // Similarly for unaligned memory
        for(; i + 7 < len; i += 8) {
            __m256 v_minuend = _mm256_loadu_ps(&minuend[i]);
            __m256 v_subtrahend = _mm256_set_ps(
                nnl2_convert_to_float32(subtrahend_data + (i + 7) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 6) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 5) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 4) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_float32(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256 v_result = _mm256_sub_ps(v_minuend, v_subtrahend);
            _mm256_storeu_ps(&minuend[i], v_result);
        }
    }
    
    // Scalar processing of the remainder
    for(; i < len; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_float32(subtrahend_elem, subtrahend->dtype);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}    

/** @brief 
 * Implementation of int32 subtraction with conversion from other types
 *
 ** @details 
 * Converts subtrahend elements to int32 before subtraction
 *
 ** @see nnl2_avx256_subinplace
 **/
static inline void nnl2_avx_sub_int32_diff_type(int32_t* minuend, const Tensor* subtrahend, size_t len, bool aligned_minuend) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t subtrahend_step = get_dtype_size(subtrahend->dtype);
    char* subtrahend_data = (char*)subtrahend->data;
    
    size_t i = 0;
    
    // Vector processing of 8 elements per iteration
    if(aligned_minuend) {
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_load_si256((__m256i*)&minuend[i]);
            
            // Creating a vector of 8 int32s with conversion
            __m256i v_subtrahend = _mm256_set_epi32(
                nnl2_convert_to_int32(subtrahend_data + (i + 7) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 6) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 5) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 4) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_store_si256((__m256i*)&minuend[i], v_result);
        }
    } else {
        // Similarly for unaligned memory
        for(; i + 7 < len; i += 8) {
            __m256i v_minuend = _mm256_loadu_si256((__m256i*)&minuend[i]);
            __m256i v_subtrahend = _mm256_set_epi32(
                nnl2_convert_to_int32(subtrahend_data + (i + 7) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 6) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 5) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 4) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 3) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 2) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 1) * subtrahend_step, subtrahend->dtype),
                nnl2_convert_to_int32(subtrahend_data + (i + 0) * subtrahend_step, subtrahend->dtype)
            );
            
            __m256i v_result = _mm256_sub_epi32(v_minuend, v_subtrahend);
            _mm256_storeu_si256((__m256i*)&minuend[i], v_result);
        }
    }
    
    // Scalar processing of the remainder
    for(; i < len; i++) {
        void* subtrahend_elem = subtrahend_data + i * subtrahend_step;
        minuend[i] -= nnl2_convert_to_int32(subtrahend_elem, subtrahend->dtype);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif

/** @ingroup backend_system
 ** @brief Backend implementations for subtract in-place
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_subinplace: Basic reference implementation
 *  - nnl2_avx256_subinplace: AVX256 implementation 
 *
 ** @see nnl2_naive_subinplace
 ** @see nnl2_avx256_subinplace
 **/
Implementation subinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_subinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
		#if TENSOR_MEM_ALIGNMENT == 32
			REGISTER_BACKEND(nnl2_avx256_subinplace, nnl2_avx256, AVX256_BACKEND_NAME),
		#endif
	#endif
};

/**
 * @brief Function pointer for subtract in-place
 * @ingroup backend_system 
 */
subinplacefn subinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(subinplace);

/** 
 * @brief Sets the backend for subtract in-place
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_subinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(subinplace_backends, subinplace, backend_name, current_backend(subinplace));
}

/** 
 * @brief Gets the name of the active backend for subtract in-place
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_subinplace_backend() {
	return current_backend(subinplace);
}

/** 
 * @brief Function declaration for getting all `subinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(subinplace);

/**
 * @brief Function declaration for getting the number of all `subinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(subinplace);

/**
 * @brief Gets the memory alignment requirement for tensors
 * @ingroup backend_system
 * @return Memory alignment in bytes
 */
int get_mem_alignment() {
	return TENSOR_MEM_ALIGNMENT;
}

/**
 * @brief Gets the total number of elements in a tensor
 * @ingroup tensor_utils
 * @param tensor Pointer to the tensor
 * @return Total number of elements (product of all dimensions)
 */
int get_size(Tensor* tensor) {
	return product(tensor->shape, tensor->rank);
}

/**
 * @brief Gets the total memory size of a tensor in bytes
 * @ingroup tensor_utils
 * @param tensor Pointer to the tensor
 * @return Total memory size in bytes
 */
int get_size_in_bytes(Tensor* tensor) {
	return product(tensor->shape, tensor->rank) * get_dtype_size(tensor->dtype);
}

/** @brief
 * Performs element-wise addition of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the sum of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param summand
 * Pointer to the summand tensor
 *
 ** @param addend
 * Pointer to the addend tensor
 *
 ** @return 
 * Pointer to a new tensor with the addition result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations (sometimes, with -O2/-O3, naive loops cause errors without volatile. based on experience, adding volatile does not affect speed)
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
Tensor* nnl2_naive_add(Tensor* summand, Tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate the total number of elements in the tensors
	size_t len = product(summand->shape, summand->rank);
	
	TensorType dtype_summand = summand->dtype;
	TensorType dtype_addend = addend->dtype;
	
	// Selecting the winning type (higher in the hierarchy)
	TensorType winner_in_the_type_hierarchy = MAX(dtype_summand, dtype_addend);

	// Create an output tensor with the same shape and data type
	Tensor* amount = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
	
	if(len == 0) return amount;
	
	if(dtype_summand == dtype_addend) {
		// Handling the case if the data types match
		
		switch(dtype_summand) {
			case FLOAT64: {
				volatile double* data_summand = (double*)summand->data;
				volatile double* data_addend = (double*)addend->data;
				volatile double* data_amount = (double*)amount->data;
			
				// Element-wise addition
				for(size_t i = 0; i < len; i++) {
					data_amount[i] = data_summand[i] + data_addend[i];
				}
				
				break;
			}
			
			case FLOAT32: {
				volatile float* data_summand = (float*)summand->data;
				volatile float* data_addend = (float*)addend->data;
				volatile float* data_amount = (float*)amount->data;
		
				// Element-wise addition
				for(size_t i = 0; i < len; i++) {
					data_amount[i] = data_summand[i] + data_addend[i];
				}
				
				break;
			}
			
			case INT32: {
				volatile int32_t* data_summand = (int32_t*)summand->data;
				volatile int32_t* data_addend = (int32_t*)addend->data;
				volatile int32_t* data_amount = (int32_t*)amount->data;
		
				// Element-wise addition
				for(size_t i = 0; i < len; i++) {
					data_amount[i] = data_summand[i] + data_addend[i];
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return NULL;
			}
		}
	} else {
		// Handling the case if the data types are NOT match
		switch(winner_in_the_type_hierarchy) {
			case FLOAT64: {
				volatile double* data_amount = (double*)amount->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
					void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
					
					data_amount[i] = nnl2_convert_to_float64(elem_summand, dtype_summand) + nnl2_convert_to_float64(elem_addend, dtype_addend);
				}
				
				break;
			}
			
			case FLOAT32: {
				volatile float* data_amount = (float*)amount->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
					void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
					
					data_amount[i] = nnl2_convert_to_float32(elem_summand, dtype_summand) + nnl2_convert_to_float32(elem_addend, dtype_addend);
				}
				
				break;
			}
        
			case INT32: {
				volatile int32_t* data_amount = (int32_t*)amount->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
					void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
					
					data_amount[i] = nnl2_convert_to_int32(elem_summand, dtype_summand) + nnl2_convert_to_int32(elem_addend, dtype_addend);
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
				return NULL;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return amount;
}

#ifdef __AVX__

/** @brief 
 * AVX256 optimized element-wise addition for int32 tensors (non-in-place)
 *
 ** @details
 * Performs vectorized addition of two int32 tensors using AVX256 instructions
 * Handles four different alignment scenarios for optimal performance
 *
 ** @param a 
 * Pointer to destination tensor data (will store result)
 *
 ** @param b 
 * Pointer to source tensor data (will not be modified)
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_a 
 * Whether tensor a is aligned to 32-byte boundary
 *
 ** @param aligned_b 
 * Whether tensor b is aligned to 32-byte boundary
 *
 ** @see nnl2_avx256_add
 **/
static inline void nnl2_avx_add_non_in_place_int32_same_type(int32_t* a, const int32_t* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * AVX256 optimized element-wise addition for float32 tensors (non-in-place)
 *
 * Вocumentation is identical to the documentation of the 
 * nnl2_avx_add_non_in_place_int32_same_type declaration
 *
 ** @see nnl2_avx256_add
 ** @see nnl2_avx_add_non_in_place_int32_same_type
 **/
static inline void nnl2_avx_add_non_in_place_float32_same_type(float* a, const float* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * AVX256 optimized element-wise addition for float64 tensors (non-in-place)
 *
 * Вocumentation is identical to the documentation of the 
 * nnl2_avx_add_non_in_place_int32_same_type declaration
 *
 ** @see nnl2_avx256_add
 ** @see nnl2_avx_add_non_in_place_float32_same_type
 ** @see nnl2_avx_add_non_in_place_int32_same_type
 **/
static inline void nnl2_avx_add_non_in_place_float64_same_type(double* a, const double* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * Performs element-wise addition of two tensors using AVX256 instructions
 * 
 ** @details
 * The function creates a new tensor containing the sum of corresponding elements
 * from two input tensors. It supports various data types with automatic type
 * promotion to the highest type in the hierarchy. For same data types, it uses
 * optimized AVX256 vector instructions. For mixed types, it falls back to scalar
 * operations with type conversion
 * 
 ** @param summand 
 * Pointer to the summand tensor
 *
 ** @param addend 
 * Pointer to the addend tensor
 * 
 ** @return 
 * Pointer to a new tensor containing the element-wise sum
 * 
 ** @note 
 * For mixed types, scalar operations are used due to AVX limitations
 * in handling type conversions within vector instructions
 *
 ** @note  
 * Includes proper handling of empty tensors (len == 0)
 * 
 ** @see nnl2_empty()
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()  
 ** @see nnl2_convert_to_int32()
 **/
Tensor* nnl2_avx256_add(const Tensor* summand, const Tensor* addend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    size_t len = product(summand->shape, summand->rank);
    
    TensorType dtype_summand = summand->dtype;
    TensorType dtype_addend = addend->dtype;
	
	// Selecting the winning type (higher in the hierarchy)
	TensorType winner_in_the_type_hierarchy = MAX(dtype_summand, dtype_addend);
    
    Tensor* sum = nnl2_empty(summand->shape, summand->rank, winner_in_the_type_hierarchy);
	
	if(len == 0) return sum; 
    
	if(dtype_summand == dtype_addend) {
	    // Check alignment for both tensors
		bool aligned_summand = NNL2_IS_ALIGNED(summand->data, NNL2_TENSOR_ALIGNMENT_32);
		bool aligned_addend = NNL2_IS_ALIGNED(addend->data, NNL2_TENSOR_ALIGNMENT_32);

		// Handling the case when the data types are the same
		switch(dtype_summand) {
			case FLOAT64: {
                double* data_summand = (double*)summand->data;
                double* data_addend = (double*)addend->data;
                double* data_sum = (double*)sum->data;
                
                // Copy data from summand to result first
                memcpy(data_sum, data_summand, len * sizeof(double));
                
                // Use optimized addition
                nnl2_avx_add_non_in_place_float64_same_type(data_sum, data_addend, len, aligned_summand, aligned_addend);
                break;
            }
            
            case FLOAT32: {
                float* data_summand = (float*)summand->data;
                float* data_addend = (float*)addend->data;
                float* data_sum = (float*)sum->data;
                
                // Copy data from summand to result first
                memcpy(data_sum, data_summand, len * sizeof(float));
                
                // Use optimized addition
                nnl2_avx_add_non_in_place_float32_same_type(data_sum, data_addend, len, aligned_summand, aligned_addend);
                break;
            }
            
            case INT32: {
                int32_t* data_summand = (int32_t*)summand->data;
                int32_t* data_addend = (int32_t*)addend->data;
                int32_t* data_sum = (int32_t*)sum->data;
                
                // Copy data from summand to result first
                memcpy(data_sum, data_summand, len * sizeof(int32_t));
                
                // Use optimized addition
                nnl2_avx_add_non_in_place_int32_same_type(data_sum, data_addend, len, aligned_summand, aligned_addend);
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(dtype_summand);
				return NULL;
			}
		} 
	} else {
		// Handling the case when the data types are NOT the same
		// For mixed types, using scalar operations since AVX doesn't easily handle
        // type conversions within the same instruction
		
		switch(winner_in_the_type_hierarchy) {
			case FLOAT64: {
                double* data_sum = (double*)sum->data;
                
				// Element-wise addition
                for(size_t i = 0; i < len; i++) {
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
                    
                    data_sum[i] = nnl2_convert_to_float64(elem_summand, dtype_summand) + nnl2_convert_to_float64(elem_addend, dtype_addend);
                }
				
                break;
            }
            
            case FLOAT32: {
                float* data_sum = (float*)sum->data;
				
				// Element-wise addition
                for(size_t i = 0; i < len; i++) {
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
                    
                    data_sum[i] = nnl2_convert_to_float32(elem_summand, dtype_summand) + nnl2_convert_to_float32(elem_addend, dtype_addend);
                }
                
                break;
            }
            
            case INT32: {
                int32_t* data_sum = (int32_t*)sum->data;

				// Element-wise addition
                for(size_t i = 0; i < len; i++) {
                    void* elem_summand = (char*)summand->data + i * get_dtype_size(dtype_summand);
                    void* elem_addend = (char*)addend->data + i * get_dtype_size(dtype_addend);
                    
                    data_sum[i] = nnl2_convert_to_int32(elem_summand, dtype_summand) + nnl2_convert_to_int32(elem_addend, dtype_addend);
                }
                
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
				return NULL;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
    
    return sum;
}

/** @brief 
 * AVX-optimized element-wise addition for int32 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_add_non_in_place_int32_same_type (declaration)
 **/
static inline void nnl2_avx_add_non_in_place_int32_same_type(int32_t* a, const int32_t* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_load_si256((__m256i*)&a[i]);       // Fast loading of aligned data
            __m256i v_b = _mm256_load_si256((__m256i*)&b[i]);       // Fast loading of aligned data
            __m256i v_result = _mm256_add_epi32(v_a, v_b);          // Vector addition
            _mm256_store_si256((__m256i*)&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_load_si256((__m256i*)&a[i]);       // Fast loading of aligned data
            __m256i v_b = _mm256_loadu_si256((__m256i*)&b[i]);      // Slow loading of unaligned data
            __m256i v_result = _mm256_add_epi32(v_a, v_b);          // Vector addition
            _mm256_store_si256((__m256i*)&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_loadu_si256((__m256i*)&a[i]);      // Slow loading of unaligned data
            __m256i v_b = _mm256_load_si256((__m256i*)&b[i]);       // Fast loading of aligned data
            __m256i v_result = _mm256_add_epi32(v_a, v_b);          // Vector addition
            _mm256_storeu_si256((__m256i*)&a[i], v_result);         // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_loadu_si256((__m256i*)&a[i]);      // Slow loading of unaligned data
            __m256i v_b = _mm256_loadu_si256((__m256i*)&b[i]);      // Slow loading of unaligned data
            __m256i v_result = _mm256_add_epi32(v_a, v_b);          // Vector addition
            _mm256_storeu_si256((__m256i*)&a[i], v_result);         // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] += b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * AVX-optimized element-wise addition for float32 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_add_non_in_place_float32_same_type (declaration)
 **/
static inline void nnl2_avx_add_non_in_place_float32_same_type(float* a, const float* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_load_ps(&a[i]);        // Fast loading of aligned data
            __m256 v_b = _mm256_load_ps(&b[i]);        // Fast loading of aligned data
            __m256 v_result = _mm256_add_ps(v_a, v_b); // Vector addition
            _mm256_store_ps(&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_load_ps(&a[i]);        // Fast loading of aligned data
            __m256 v_b = _mm256_loadu_ps(&b[i]);       // Slow loading of unaligned data
            __m256 v_result = _mm256_add_ps(v_a, v_b); // Vector addition
            _mm256_store_ps(&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_loadu_ps(&a[i]);       // Slow loading of unaligned data
            __m256 v_b = _mm256_load_ps(&b[i]);        // Fast loading of aligned data
            __m256 v_result = _mm256_add_ps(v_a, v_b); // Vector addition
            _mm256_storeu_ps(&a[i], v_result);         // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_loadu_ps(&a[i]);       // Slow loading of unaligned data
            __m256 v_b = _mm256_loadu_ps(&b[i]);       // Slow loading of unaligned data
            __m256 v_result = _mm256_add_ps(v_a, v_b); // Vector addition
            _mm256_storeu_ps(&a[i], v_result);         // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] += b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * AVX-optimized element-wise addition for float64 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_add_non_in_place_float64_same_type (declaration)
 **/
static inline void nnl2_avx_add_non_in_place_float64_same_type(double* a, const double* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_load_pd(&a[i]);        // Fast loading of aligned data
            __m256d v_b = _mm256_load_pd(&b[i]);        // Fast loading of aligned data
            __m256d v_result = _mm256_add_pd(v_a, v_b); // Vector addition
            _mm256_store_pd(&a[i], v_result);           // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_load_pd(&a[i]);        // Fast loading of aligned data
            __m256d v_b = _mm256_loadu_pd(&b[i]);       // Slow loading of unaligned data
            __m256d v_result = _mm256_add_pd(v_a, v_b); // Vector addition
            _mm256_store_pd(&a[i], v_result);           // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_loadu_pd(&a[i]);       // Slow loading of unaligned data
            __m256d v_b = _mm256_load_pd(&b[i]);        // Fast loading of aligned data
            __m256d v_result = _mm256_add_pd(v_a, v_b); // Vector addition
            _mm256_storeu_pd(&a[i], v_result);          // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_loadu_pd(&a[i]);       // Slow loading of unaligned data
            __m256d v_b = _mm256_loadu_pd(&b[i]);       // Slow loading of unaligned data
            __m256d v_result = _mm256_add_pd(v_a, v_b); // Vector addition
            _mm256_storeu_pd(&a[i], v_result);          // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] += b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif

/** 
 * @ingroup backend_system
 * @brief Backend implementations for addition operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - naive_add: Basic reference implementation
 *  - nnl2_avx256_add: AVX256 implementation (if available)
 * 
 * @see naive_add
 * @see nnl2_avx256_add
 */
Implementation add_backends[] = {
	REGISTER_BACKEND(nnl2_naive_add, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
		#if TENSOR_MEM_ALIGNMENT == 32
			REGISTER_BACKEND(nnl2_avx256_add, nnl2_avx256, AVX256_BACKEND_NAME),
		#endif
	#endif
};

/**
 * @brief Function pointer for addition operation
 * @ingroup backend_system 
 */
addfn add;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(add);

/** 
 * @brief Sets the backend for addition operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_add_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(add_backends, add, backend_name, CURRENT_BACKEND(add));
}

/** 
 * @brief Gets the name of the active backend for addition operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_add_backend() {
	return current_backend(add);
}

/** 
 * @brief Function declaration for getting all `add` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(add);

/**
 * @brief Function declaration for getting the number of all `add` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(add);

/** @brief
 * Performs element-wise subtraction of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the difference of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param minuend
 * Pointer to the minuend tensor (number from which to subtract)
 *
 ** @param subtrahend
 * Pointer to the subtrahend tensor (number to subtract)
 *
 ** @return 
 * Pointer to a new tensor with the subtraction result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations (sometimes, with -O2/-O3, naive loops cause errors without volatile. based on experience, adding volatile does not affect speed)
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
Tensor* nnl2_naive_sub(const Tensor* minuend, const Tensor* subtrahend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate the total number of elements in the tensors
	size_t len = product(minuend->shape, minuend->rank);
	
	TensorType dtype_minuend = minuend->dtype;
	TensorType dtype_subtrahend = subtrahend->dtype;
	
	// Selecting the winning type (higher in the hierarchy)
	TensorType winner_in_the_type_hierarchy = MAX(dtype_minuend, dtype_subtrahend);

	// Create an output tensor with the same shape and data type
	Tensor* difference = nnl2_empty(minuend->shape, minuend->rank, winner_in_the_type_hierarchy);
	
	if(len == 0) return difference;
	
	if(dtype_minuend == dtype_subtrahend) {
		// Handling the case if the data types match
		
		switch(dtype_minuend) {
			case FLOAT64: {
				volatile double* data_minuend = (double*)minuend->data;
				volatile double* data_subtrahend = (double*)subtrahend->data;
				volatile double* data_difference = (double*)difference->data;
			
				// Element-wise subtraction
				for(size_t i = 0; i < len; i++) {
					data_difference[i] = data_minuend[i] - data_subtrahend[i];
				}
				
				break;
			}
			
			case FLOAT32: {
				volatile float* data_minuend = (float*)minuend->data;
				volatile float* data_subtrahend = (float*)subtrahend->data;
				volatile float* data_difference = (float*)difference->data;
		
				// Element-wise subtraction
				for(size_t i = 0; i < len; i++) {
					data_difference[i] = data_minuend[i] - data_subtrahend[i];
				}
				
				break;
			}
			
			case INT32: {
				volatile int32_t* data_minuend = (int32_t*)minuend->data;
				volatile int32_t* data_subtrahend = (int32_t*)subtrahend->data;
				volatile int32_t* data_difference = (int32_t*)difference->data;
		
				// Element-wise subtraction
				for(size_t i = 0; i < len; i++) {
					data_difference[i] = data_minuend[i] - data_subtrahend[i];
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(dtype_minuend);
				return NULL;
			}
		}
	} else {
		// Handling the case if the data types are NOT match
		switch(winner_in_the_type_hierarchy) {
			case FLOAT64: {
				volatile double* data_difference = (double*)difference->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
					void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
					
					data_difference[i] = nnl2_convert_to_float64(elem_minuend, dtype_minuend) - nnl2_convert_to_float64(elem_subtrahend, dtype_subtrahend);
				}
				
				break;
			}
			
			case FLOAT32: {
				volatile float* data_difference = (float*)difference->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
					void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
					
					data_difference[i] = nnl2_convert_to_float32(elem_minuend, dtype_minuend) - nnl2_convert_to_float32(elem_subtrahend, dtype_subtrahend);
				}
				
				break;
			}
        
			case INT32: {
				volatile int32_t* data_difference = (int32_t*)difference->data;
				
				for(size_t i = 0; i < len; i++) {
					// Calculate the pointers to the current elements, taking into account the size of the type
					void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
					void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
					
					data_difference[i] = nnl2_convert_to_int32(elem_minuend, dtype_minuend) - nnl2_convert_to_int32(elem_subtrahend, dtype_subtrahend);
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
				return NULL;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return difference;
}

#ifdef __AVX__

/** @brief 
 * AVX256 optimized element-wise subtraction for int32 tensors (non-in-place)
 *
 ** @details
 * Performs vectorized subtraction of two int32 tensors using AVX256 instructions
 * Handles four different alignment scenarios for optimal performance
 *
 ** @param a 
 * Pointer to destination tensor data (will store result)
 *
 ** @param b 
 * Pointer to source tensor data (will not be modified)
 *
 ** @param len 
 * Number of elements to process
 *
 ** @param aligned_a 
 * Whether tensor a is aligned to 32-byte boundary
 *
 ** @param aligned_b 
 * Whether tensor b is aligned to 32-byte boundary
 *
 ** @see nnl2_avx256_sub
 **/
static inline void nnl2_avx_sub_non_in_place_int32_same_type(int32_t* a, const int32_t* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * AVX256 optimized element-wise subtraction for float32 tensors (non-in-place)
 *
 * Вocumentation is identical to the documentation of the 
 * nnl2_avx_sub_non_in_place_int32_same_type declaration
 *
 ** @see nnl2_avx256_sub
 ** @see nnl2_avx_sub_non_in_place_int32_same_type
 **/
static inline void nnl2_avx_sub_non_in_place_float32_same_type(float* a, const float* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * AVX256 optimized element-wise subtraction for float64 tensors (non-in-place)
 *
 * Вocumentation is identical to the documentation of the 
 * nnl2_avx_sub_non_in_place_int32_same_type declaration
 *
 ** @see nnl2_avx256_sub
 ** @see nnl2_avx_sub_non_in_place_float32_same_type
 ** @see nnl2_avx_sub_non_in_place_int32_same_type
 **/
static inline void nnl2_avx_sub_non_in_place_float64_same_type(double* a, const double* b, size_t len, bool aligned_a, bool aligned_b);

/** @brief 
 * Performs element-wise subtraction of two tensors using AVX256 instructions
 * 
 ** @details
 * The function creates a new tensor containing the difference of corresponding elements
 * from two input tensors. It supports various data types with automatic type
 * promotion to the highest type in the hierarchy. For same data types, it uses
 * optimized AVX256 vector instructions. For mixed types, it falls back to scalar
 * operations with type conversion
 * 
 ** @param minuend 
 * Pointer to the minuend tensor (number from which to subtract)
 *
 ** @param subtrahend 
 * Pointer to the subtrahend tensor (number to subtract)
 * 
 ** @return 
 * Pointer to a new tensor containing the element-wise difference
 * 
 ** @note 
 * For mixed types, scalar operations are used due to AVX limitations
 * in handling type conversions within vector instructions
 *
 ** @note  
 * Includes proper handling of empty tensors (len == 0)
 * 
 ** @see nnl2_empty()
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()  
 ** @see nnl2_convert_to_int32()
 **/
Tensor* nnl2_avx256_sub(const Tensor* minuend, const Tensor* subtrahend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    size_t len = product(minuend->shape, minuend->rank);
    
    TensorType dtype_minuend = minuend->dtype;
    TensorType dtype_subtrahend = subtrahend->dtype;
	
	// Selecting the winning type (higher in the hierarchy)
	TensorType winner_in_the_type_hierarchy = MAX(dtype_minuend, dtype_subtrahend);
    
    Tensor* difference = nnl2_empty(minuend->shape, minuend->rank, winner_in_the_type_hierarchy);
	
	if(len == 0) return difference; 
    
	if(dtype_minuend == dtype_subtrahend) {
	    // Check alignment for both tensors
		bool aligned_minuend = NNL2_IS_ALIGNED(minuend->data, NNL2_TENSOR_ALIGNMENT_32);
		bool aligned_subtrahend = NNL2_IS_ALIGNED(subtrahend->data, NNL2_TENSOR_ALIGNMENT_32);

		// Handling the case when the data types are the same
		switch(dtype_minuend) {
			case FLOAT64: {
                double* data_minuend = (double*)minuend->data;
                double* data_subtrahend = (double*)subtrahend->data;
                double* data_difference = (double*)difference->data;
                
                // Copy data from minuend to result first
                memcpy(data_difference, data_minuend, len * sizeof(double));
                
                // Use optimized subtraction
                nnl2_avx_sub_non_in_place_float64_same_type(data_difference, data_subtrahend, len, aligned_minuend, aligned_subtrahend);
                break;
            }
            
            case FLOAT32: {
                float* data_minuend = (float*)minuend->data;
                float* data_subtrahend = (float*)subtrahend->data;
                float* data_difference = (float*)difference->data;
                
                // Copy data from minuend to result first
                memcpy(data_difference, data_minuend, len * sizeof(float));
                
                // Use optimized subtraction
                nnl2_avx_sub_non_in_place_float32_same_type(data_difference, data_subtrahend, len, aligned_minuend, aligned_subtrahend);
                break;
            }
            
            case INT32: {
                int32_t* data_minuend = (int32_t*)minuend->data;
                int32_t* data_subtrahend = (int32_t*)subtrahend->data;
                int32_t* data_difference = (int32_t*)difference->data;
                
                // Copy data from minuend to result first
                memcpy(data_difference, data_minuend, len * sizeof(int32_t));
                
                // Use optimized subtraction
                nnl2_avx_sub_non_in_place_int32_same_type(data_difference, data_subtrahend, len, aligned_minuend, aligned_subtrahend);
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(dtype_minuend);
				return NULL;
			}
		} 
	} else {
		// Handling the case when the data types are NOT the same
		// For mixed types, using scalar operations since AVX doesn't easily handle
        // type conversions within the same instruction
		
		switch(winner_in_the_type_hierarchy) {
			case FLOAT64: {
                double* data_difference = (double*)difference->data;
                
				// Element-wise subtraction
                for(size_t i = 0; i < len; i++) {
                    void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
                    void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
                    
                    data_difference[i] = nnl2_convert_to_float64(elem_minuend, dtype_minuend) - nnl2_convert_to_float64(elem_subtrahend, dtype_subtrahend);
                }
				
                break;
            }
            
            case FLOAT32: {
                float* data_difference = (float*)difference->data;
				
				// Element-wise subtraction
                for(size_t i = 0; i < len; i++) {
                    void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
                    void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
                    
                    data_difference[i] = nnl2_convert_to_float32(elem_minuend, dtype_minuend) - nnl2_convert_to_float32(elem_subtrahend, dtype_subtrahend);
                }
                
                break;
            }
            
            case INT32: {
                int32_t* data_difference = (int32_t*)difference->data;

				// Element-wise subtraction
                for(size_t i = 0; i < len; i++) {
                    void* elem_minuend = (char*)minuend->data + i * get_dtype_size(dtype_minuend);
                    void* elem_subtrahend = (char*)subtrahend->data + i * get_dtype_size(dtype_subtrahend);
                    
                    data_difference[i] = nnl2_convert_to_int32(elem_minuend, dtype_minuend) - nnl2_convert_to_int32(elem_subtrahend, dtype_subtrahend);
                }
                
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
				return NULL;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
    
    return difference;
}

/** @brief 
 * AVX-optimized element-wise subtraction for int32 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_sub_non_in_place_int32_same_type (declaration)
 **/
static inline void nnl2_avx_sub_non_in_place_int32_same_type(int32_t* a, const int32_t* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_load_si256((__m256i*)&a[i]);       // Fast loading of aligned data
            __m256i v_b = _mm256_load_si256((__m256i*)&b[i]);       // Fast loading of aligned data
            __m256i v_result = _mm256_sub_epi32(v_a, v_b);          // Vector subtraction
            _mm256_store_si256((__m256i*)&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_load_si256((__m256i*)&a[i]);       // Fast loading of aligned data
            __m256i v_b = _mm256_loadu_si256((__m256i*)&b[i]);      // Slow loading of unaligned data
            __m256i v_result = _mm256_sub_epi32(v_a, v_b);          // Vector subtraction
            _mm256_store_si256((__m256i*)&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_loadu_si256((__m256i*)&a[i]);      // Slow loading of unaligned data
            __m256i v_b = _mm256_load_si256((__m256i*)&b[i]);       // Fast loading of aligned data
            __m256i v_result = _mm256_sub_epi32(v_a, v_b);          // Vector subtraction
            _mm256_storeu_si256((__m256i*)&a[i], v_result);         // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256i v_a = _mm256_loadu_si256((__m256i*)&a[i]);      // Slow loading of unaligned data
            __m256i v_b = _mm256_loadu_si256((__m256i*)&b[i]);      // Slow loading of unaligned data
            __m256i v_result = _mm256_sub_epi32(v_a, v_b);          // Vector subtraction
            _mm256_storeu_si256((__m256i*)&a[i], v_result);         // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] -= b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * AVX-optimized element-wise subtraction for float32 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_sub_non_in_place_float32_same_type (declaration)
 **/
static inline void nnl2_avx_sub_non_in_place_float32_same_type(float* a, const float* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_load_ps(&a[i]);        // Fast loading of aligned data
            __m256 v_b = _mm256_load_ps(&b[i]);        // Fast loading of aligned data
            __m256 v_result = _mm256_sub_ps(v_a, v_b); // Vector subtraction
            _mm256_store_ps(&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_load_ps(&a[i]);        // Fast loading of aligned data
            __m256 v_b = _mm256_loadu_ps(&b[i]);       // Slow loading of unaligned data
            __m256 v_result = _mm256_sub_ps(v_a, v_b); // Vector subtraction
            _mm256_store_ps(&a[i], v_result);          // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_loadu_ps(&a[i]);       // Slow loading of unaligned data
            __m256 v_b = _mm256_load_ps(&b[i]);        // Fast loading of aligned data
            __m256 v_result = _mm256_sub_ps(v_a, v_b); // Vector subtraction
            _mm256_storeu_ps(&a[i], v_result);         // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 7 < len; i += 8) {
            __m256 v_a = _mm256_loadu_ps(&a[i]);       // Slow loading of unaligned data
            __m256 v_b = _mm256_loadu_ps(&b[i]);       // Slow loading of unaligned data
            __m256 v_result = _mm256_sub_ps(v_a, v_b); // Vector subtraction
            _mm256_storeu_ps(&a[i], v_result);         // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] -= b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * AVX-optimized element-wise subtraction for float64 tensors (non-in-place)
 *
 ** @details
 * See docs at declaration
 *
 ** @see nnl2_avx_sub_non_in_place_float64_same_type (declaration)
 **/
static inline void nnl2_avx_sub_non_in_place_float64_same_type(double* a, const double* b, size_t len, bool aligned_a, bool aligned_b) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    size_t i = 0;
    
    // Case 1: Both tensors are aligned 
    if(aligned_a && aligned_b) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_load_pd(&a[i]);        // Fast loading of aligned data
            __m256d v_b = _mm256_load_pd(&b[i]);        // Fast loading of aligned data
            __m256d v_result = _mm256_sub_pd(v_a, v_b); // Vector subtraction
            _mm256_store_pd(&a[i], v_result);           // Fast saving to aligned memory
        }
    } 
    
    // Case 2: Only tensor a is aligned
    else if(aligned_a) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_load_pd(&a[i]);        // Fast loading of aligned data
            __m256d v_b = _mm256_loadu_pd(&b[i]);       // Slow loading of unaligned data
            __m256d v_result = _mm256_sub_pd(v_a, v_b); // Vector subtraction
            _mm256_store_pd(&a[i], v_result);           // Fast saving to aligned memory
        }
    } 
    
    // Case 3: Only tensor b is aligned
    else if(aligned_b) {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_loadu_pd(&a[i]);       // Slow loading of unaligned data
            __m256d v_b = _mm256_load_pd(&b[i]);        // Fast loading of aligned data
            __m256d v_result = _mm256_sub_pd(v_a, v_b); // Vector subtraction
            _mm256_storeu_pd(&a[i], v_result);          // Slow saving to unaligned memory
        }
    } 
    
    // Case 4: Both tensors are not aligned
    else {
        for(; i + 3 < len; i += 4) {
            __m256d v_a = _mm256_loadu_pd(&a[i]);       // Slow loading of unaligned data
            __m256d v_b = _mm256_loadu_pd(&b[i]);       // Slow loading of unaligned data
            __m256d v_result = _mm256_sub_pd(v_a, v_b); // Vector subtraction
            _mm256_storeu_pd(&a[i], v_result);          // Slow saving to unaligned memory
        }
    }
    
    // Processing the remainder
    for(; i < len; i++) a[i] -= b[i];
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif

/** 
 * @ingroup backend_system
 * @brief Backend implementations for subtraction operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_sub: Basic reference implementation
 *  - nnl2_avx256_sub: AVX256 implementation (if available)
 * 
 * @see nnl2_naive_sub
 * @see nnl2_avx256_sub
 */
Implementation sub_backends[] = {
	REGISTER_BACKEND(nnl2_naive_sub, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef __AVX__
	REGISTER_BACKEND(nnl2_avx256_sub, nnl2_avx256, AVX256_BACKEND_NAME),
	#endif
};

/**
 * @brief Function pointer for subtraction operation
 * @ingroup backend_system 
 */
subfn sub;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sub);

/** 
 * @brief Sets the backend for subtraction operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_sub_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sub_backends, sub, backend_name, CURRENT_BACKEND(sub));
}

/** 
 * @brief Gets the name of the active backend for subtraction operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_sub_backend() {
	return current_backend(sub);
}

/** 
 * @brief Function declaration for getting all `sub` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sub);

/**
 * @brief Function declaration for getting the number of all `sub` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sub);

/** @brief 
 * Performs element-wise multiplication of two tensors (naive implementation)
 * 
 * Multiplies the elements of the multiplicand tensor by the corresponding elements 
 * of the multiplier tensor, modifying the multiplicand tensor in place
 *
 ** @param multiplicand 
 * Pointer to the tensor that will be modified (receives the multiplication result)
 *
 ** @param multiplier 
 * Pointer to the tensor whose values will multiply the multiplicand
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The multiplier elements are converted to the multiplicand's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the multiplicand tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Multiply a by b (a becomes a * b)
 * naive_mulinplace(a, b);
 * 
 * // Now a contains 1.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_mulinplace(Tensor* multiplicand, const Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX   
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "Multiplicand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->data, "Multiplicand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->data, "Multiplier tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the multiplicand tensor
    size_t len_multiplicand = product(multiplicand->shape, multiplicand->rank);
    
    // If the tensor is empty, exit the function
    if(len_multiplicand == 0) return;
    
    TensorType dtype_multiplicand = multiplicand->dtype;
    TensorType dtype_multiplier = multiplier->dtype;
    
    if(dtype_multiplicand == dtype_multiplier) {
        // Handling case when the tensors have the same type
        
        switch(dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                volatile double* data_multiplier = (double*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];
                break;
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                volatile float* data_multiplier = (float*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                volatile int32_t* data_multiplier = (int32_t*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing multiplier tensor elements
        size_t multiplier_step = get_dtype_size(dtype_multiplier);
        
        // Casting multiplier data to char* for byte access
        char* multiplier_data = (char*)multiplier->data;
        
        switch(dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                
                // For each element, convert the multiplier element to FLOAT64 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_float64(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                
                // For each element, convert the multiplier element to FLOAT32 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_float32(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                
                // For each element, convert the multiplier element to INT32 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_int32(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
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
 * @brief Backend implementations for multiplication operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_mulinplace: Basic reference implementation
 * 
 * @see nnl2_naive_mulinplace
 */
Implementation mulinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_mulinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for multiplication operation
 * @ingroup backend_system 
 */
mulinplacefn mulinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(mulinplace);

/** 
 * @brief Sets the backend for multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_mulinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mulinplace_backends, mulinplace, backend_name, CURRENT_BACKEND(mulinplace));
}

/** 
 * @brief Gets the name of the active backend for multiplication operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_mulinplace_backend() {
	return CURRENT_BACKEND(mulinplace);
}

/** 
 * @brief Function declaration for getting all `mulinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mulinplace);

/**
 * @brief Function declaration for getting the number of all `mulinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mulinplace);

/** @brief 
 * Performs element-wise division of two tensors (naive implementation)
 * 
 * Divides the elements of the dividend tensor by the corresponding elements 
 * of the divisor tensor, modifying the dividend tensor in place
 *
 ** @param dividend 
 * Pointer to the tensor that will be modified (receives the division result)
 *
 ** @param divisor 
 * Pointer to the tensor whose values will divide the dividend
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The divisor elements are converted to the dividend's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the dividend tensor directly
 * Division by zero may result in undefined behavior depending on data type
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Divide a by b (a becomes a / b)
 * nnl2_naive_divinplace(a, b);
 * 
 * // Now a contains 1.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_divinplace(Tensor* dividend, const Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "Dividend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data, "Dividend tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data, "Divisor tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the dividend tensor
    size_t len_dividend = product(dividend->shape, dividend->rank);
    
    // If the tensor is empty, exit the function
    if(len_dividend == 0) return;
    
    TensorType dtype_dividend = dividend->dtype;
    TensorType dtype_divisor = divisor->dtype;
    
    if(dtype_dividend == dtype_divisor) {
        // Handling case when the tensors have the same type
        
        switch(dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                volatile double* data_divisor = (double*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];
                break;
            }
            
            case FLOAT32: {
                volatile float* data_dividend = (float*)dividend->data;
                volatile float* data_divisor = (float*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_dividend = (int32_t*)dividend->data;
                volatile int32_t* data_divisor = (int32_t*)divisor->data;
                
                // Element-wise division
                for(size_t i = 0; i < len_dividend; i++) data_dividend[i] /= data_divisor[i];        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing divisor tensor elements
        size_t divisor_step = get_dtype_size(dtype_divisor);
        
        // Casting divisor data to char* for byte access
        char* divisor_data = (char*)divisor->data;
        
        switch(dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                
                // For each element, convert the divisor element to float64 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_float64(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_dividend = (float*)dividend->data;
                
                // For each element, convert the divisor element to float32 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_float32(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_dividend = (int32_t*)dividend->data;
                
                // For each element, convert the divisor element to int32 and divide by it
                for(size_t i = 0; i < len_dividend; i++) {
                    void* divisor_elem = divisor_data + i * divisor_step;
                    data_dividend[i] /= nnl2_convert_to_int32(divisor_elem, dtype_divisor);
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
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
 * @brief Backend implementations for division operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_divinplace: Basic reference implementation
 * 
 * @see nnl2_naive_divinplace
 */
Implementation divinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_divinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for division operation
 * @ingroup backend_system 
 */
divinplacefn divinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(divinplace);

/** 
 * @brief Sets the backend for division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_divinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(divinplace_backends, divinplace, backend_name, CURRENT_BACKEND(divinplace));
}

/** 
 * @brief Gets the name of the active backend for division operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_divinplace_backend() {
	return CURRENT_BACKEND(divinplace);
}

/** 
 * @brief Function declaration for getting all `divinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(divinplace);

/**
 * @brief Function declaration for getting the number of all `divinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(divinplace);

/** @brief
 * Performs element-wise multiplication of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the product of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param multiplicand
 * Pointer to the multiplicand tensor
 *
 ** @param multiplier
 * Pointer to the multiplier tensor
 *
 ** @return 
 * Pointer to a new tensor with the multiplication result
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
Tensor* nnl2_naive_mul(const Tensor* multiplicand, const Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(multiplicand->shape, multiplicand->rank);
    
    TensorType dtype_multiplicand = multiplicand->dtype;
    TensorType dtype_multiplier = multiplier->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_multiplicand, dtype_multiplier);

    // Create an output tensor with the same shape and winning data type
    Tensor* product = nnl2_empty(multiplicand->shape, multiplicand->rank, winner_in_the_type_hierarchy);
    
    if (product == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return product;
    }
    
    if (dtype_multiplicand == dtype_multiplier) {
        // Handling the case if the data types match
        
        switch (dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                volatile double* data_multiplier = (double*)multiplier->data;
                volatile double* data_product = (double*)product->data;
            
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                volatile float* data_multiplier = (float*)multiplier->data;
                volatile float* data_product = (float*)product->data;
        
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                volatile int32_t* data_multiplier = (int32_t*)multiplier->data;
                volatile int32_t* data_product = (int32_t*)product->data;
        
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_product = (double*)product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_float64(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_float64(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_product = (float*)product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_float32(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_float32(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_product = (int32_t*)product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_int32(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_int32(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                return NULL;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return product;
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for multiplication operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_mul: Basic reference implementation
 * 
 * @see nnl2_naive_mul
 */
Implementation mul_backends[] = {
	REGISTER_BACKEND(nnl2_naive_mul, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for multiplication operation
 * @ingroup backend_system 
 */
mulfn mul;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(mul);

/** 
 * @brief Sets the backend for multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_mul_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mul_backends, mul, backend_name, CURRENT_BACKEND(mul));
}

/** 
 * @brief Gets the name of the active backend for multiplication operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_mul_backend() {
	return CURRENT_BACKEND(mul);
}

/** 
 * @brief Function declaration for getting all `mul` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mul);

/**
 * @brief Function declaration for getting the number of all `mul` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mul);

/** @brief
 * Creates a copy of a tensor with possible data type conversion
 *
 ** @param tensor
 * Pointer to the source tensor for copying
 *
 ** @param copy_type
 * The target data type for the copy
 *
 ** @return 
 * A pointer to a new tensor copy, or NULL if an error occurs
 *
 ** @note
 * Can perform additional checks depending on the safety level
 *
 ** @see nnl2_empty
 ** @see nnl2_convert_to_float64
 ** @see nnl2_convert_to_float32
 ** @see nnl2_convert_to_int32
 **/
Tensor* naive_copy(Tensor* tensor, TensorType copy_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Additional checks depending on the safety level
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		// NULL checks
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "Tensor shape is NULL", NULL);
	#endif
	
	TensorType dtype = tensor->dtype;
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	Tensor* result;
	
	if(dtype == copy_type) {
		result = nnl2_empty(tensor->shape, tensor->rank, dtype);
		
		// Element-by-element copying based on data type
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
				NNL2_TYPE_ERROR(dtype);
				return NULL;
			}
		}
	} else {
		// Create a tensor with the target data type
		result = nnl2_empty(tensor->shape, tensor->rank, copy_type);
		
		// Data conversion and copying
		switch(copy_type) {
			case FLOAT64: {
				double* cast_data_copy = (double*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_float64(original_elem, dtype);
				}
				
				break;
			}
			
			case FLOAT32: {
				float* cast_data_copy = (float*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_float32(original_elem, dtype);
				}
				
				break;
			}
			
			case INT32: {
				int32_t* cast_data_copy = (int32_t*)result->data;
				
				for(size_t it = 0; it < total_elems; it++) {
					// Getting a pointer to the current element of the source tensor
					void* original_elem = (char*)tensor->data + it * get_dtype_size(dtype);
					
					// Convert and copy the element
					cast_data_copy[it] = nnl2_convert_to_int32(original_elem, dtype);
				}
				
				break;
			}
			
			default: {
				NNL2_TYPE_ERROR(copy_type);
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
 * @brief Backend implementations for copy operation
 * @details
 * Array follows the common backend registration pattern for copy operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation copy_backends[] = {
	REGISTER_BACKEND(naive_copy, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for copy operation
 * @ingroup backend_system 
 */
copyfn nnl2_copy;

/** 
 * @brief Makes the copy backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(copy);

/** 
 * @brief Sets the backend for copy operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_copy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(copy_backends, nnl2_copy, backend_name, current_backend(copy));
}

/** 
 * @brief Gets the name of the active backend for copy operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_copy_backend() {
	return current_backend(copy);
}

/** 
 * @brief Function declaration for getting all available copy backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(copy);

/**
 * @brief Function declaration for getting the number of available copy backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(copy);

/** @brief
 * Performs element-wise division of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the quotient of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy. Checks for division by zero.
 *
 ** @param dividend
 * Pointer to the dividend tensor
 *
 ** @param divisor
 * Pointer to the divisor tensor
 *
 ** @return 
 * Pointer to a new tensor with the division result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or division by zero
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* nnl2_naive_div(const Tensor* dividend, const Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(dividend->shape, dividend->rank);
    
    TensorType dtype_dividend = dividend->dtype;
    TensorType dtype_divisor = divisor->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_dividend, dtype_divisor);

    // Create an output tensor with the same shape and winning data type
    Tensor* quotient = nnl2_empty(dividend->shape, dividend->rank, winner_in_the_type_hierarchy);
    
    if (quotient == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return quotient;
    }
    
    if (dtype_dividend == dtype_divisor) {
        // Handling the case if the data types match
        
        switch (dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                volatile double* data_divisor = (double*)divisor->data;
                volatile double* data_quotient = (double*)quotient->data;
            
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0.0) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
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
        
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0.0f) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
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
        
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    data_quotient[i] = data_dividend[i] / data_divisor[i];
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
                nnl2_free_tensor(quotient);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_quotient = (double*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    double divisor_val = nnl2_convert_to_float64(elem_divisor, dtype_divisor);
                    if (divisor_val == 0.0) {
                        fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_float64(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_quotient = (float*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    float divisor_val = nnl2_convert_to_float32(elem_divisor, dtype_divisor);
                    if (divisor_val == 0.0f) {
                        fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_float32(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_quotient = (int32_t*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    int32_t divisor_val = nnl2_convert_to_int32(elem_divisor, dtype_divisor);
                    if (divisor_val == 0) {
                        fprintf(stderr, "Error (Hello from C!): Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_int32(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                nnl2_free_tensor(quotient);
                return NULL;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return quotient;
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for division operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_div: Basic reference implementation
 * 
 * @see nnl2_naive_div
 */
Implementation div_backends[] = {
	REGISTER_BACKEND(nnl2_naive_div, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for division operation
 * @ingroup backend_system 
 */
divfn nnl2_div;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(div);

/** 
 * @brief Sets the backend for division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_div_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(div_backends, div, backend_name, CURRENT_BACKEND(div));
}

/** 
 * @brief Gets the name of the active backend for division operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_div_backend() {
	return CURRENT_BACKEND(div);
}

/** 
 * @brief Function declaration for getting all `div` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(div);

/**
 * @brief Function declaration for getting the number of all `div` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(div);

/** @brief 
 * Performs element-wise exponentiation of two tensors (naive implementation)
 * 
 * Raises the elements of the base tensor to the power of the corresponding elements 
 * of the exponent tensor, modifying the base tensor in place
 *
 ** @param base 
 * Pointer to the tensor that will be modified (receives the exponentiation result)
 *
 ** @param exponent 
 * Pointer to the tensor whose values will be used as exponents
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The exponent elements are converted to the base's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the base tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_tensor((float[]){2.0, 3.0, 4.0}, (int[]){3}, 1, FLOAT32);
 * Tensor* b = nnl2_tensor((float[]){2.0, 1.0, 0.5}, (int[]){3}, 1, FLOAT32);
 * 
 * // Raise a to the power of b (a becomes a^b)
 * naive_powinplace(a, b);
 * 
 * // Now a contains [4.0, 3.0, 2.0]
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_powinplace(Tensor* base, const Tensor* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(base, "Base tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(base->data, "Base tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(exponent, "Exponent tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(exponent->data, "Exponent tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the base tensor
    size_t len_base = product(base->shape, base->rank);
    
    // If the tensor is empty, exit the function
    if(len_base == 0) return;
    
    TensorType dtype_base = base->dtype;
    TensorType dtype_exponent = exponent->dtype;
    
    if(dtype_base == dtype_exponent) {
        // Handling case when the tensors have the same type
        
        switch(dtype_base) {
            case FLOAT64: {
                volatile double* data_base = (double*)base->data;
                volatile double* data_exponent = (double*)exponent->data;
                
                // Element-wise exponentiation
                for(size_t i = 0; i < len_base; i++) {
                    data_base[i] = pow(data_base[i], data_exponent[i]);
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_base = (float*)base->data;
                volatile float* data_exponent = (float*)exponent->data;
                
                // Element-wise exponentiation
                for(size_t i = 0; i < len_base; i++) {
                    data_base[i] = powf(data_base[i], data_exponent[i]);
                }    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_base = (int32_t*)base->data;
                volatile int32_t* data_exponent = (int32_t*)exponent->data;
                
                // Element-wise exponentiation
                for(size_t i = 0; i < len_base; i++) {
                    data_base[i] = (int32_t)pow(data_base[i], data_exponent[i]);
                }        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_base);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing exponent tensor elements
        size_t exponent_step = get_dtype_size(dtype_exponent);
        
        // Casting exponent data to char* for byte access
        char* exponent_data = (char*)exponent->data;
        
        switch(dtype_base) {
            case FLOAT64: {
                volatile double* data_base = (double*)base->data;
                
                // For each element, convert the exponent element to FLOAT64 and use it as exponent
                for(size_t i = 0; i < len_base; i++) {
                    void* exponent_elem = exponent_data + i * exponent_step;
                    data_base[i] = pow(data_base[i], nnl2_convert_to_float64(exponent_elem, dtype_exponent));
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_base = (float*)base->data;
                
                // For each element, convert the exponent element to FLOAT32 and use it as exponent
                for(size_t i = 0; i < len_base; i++) {
                    void* exponent_elem = exponent_data + i * exponent_step;
                    data_base[i] = powf(data_base[i], nnl2_convert_to_float32(exponent_elem, dtype_exponent));
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_base = (int32_t*)base->data;
                
                // For each element, convert the exponent element to INT32 and use it as exponent
                for(size_t i = 0; i < len_base; i++) {
                    void* exponent_elem = exponent_data + i * exponent_step;
                    data_base[i] = (int32_t)pow(data_base[i], nnl2_convert_to_int32(exponent_elem, dtype_exponent));
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_base);
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
 * @brief Backend implementations for in-place power operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - naive_powinplace: Basic reference implementation
 * 
 * @see naive_powinplace
 */
Implementation powinplace_backends[] = {
	REGISTER_BACKEND(naive_powinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for in-place power operation
 * @ingroup backend_system 
 */
powinplacefn powinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(powinplace);

/** 
 * @brief Sets the backend for in-place power operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_powinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(powinplace_backends, powinplace, backend_name, CURRENT_BACKEND(powinplace));
}

/** 
 * @brief Gets the name of the active backend for in-place power operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_powinplace_backend() {
	return CURRENT_BACKEND(powinplace);
}

/** 
 * @brief Function declaration for getting all `powinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(powinplace);

/**
 * @brief Function declaration for getting the number of all `powinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(powinplace);

/** @brief
 * Performs element-wise exponentiation of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the result of raising each element
 * of the base tensor to the power of the corresponding element in the exponent tensor.
 * It supports various data types with automatic casting to a higher type in the hierarchy.
 *
 ** @param base
 * Pointer to the base tensor
 *
 ** @param exponent
 * Pointer to the exponent tensor
 *
 ** @return 
 * Pointer to a new tensor with the exponentiation result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or unsupported data type
 *
 ** @note
 * For integer types, the result is cast back to integer which may truncate values
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_pow(const Tensor* base, const Tensor* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks on maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base, "Base tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base, "Exponent tensor is NULL", NULL);
	#endif
	
    // Calculate the total number of elements in the tensors
    size_t len = product(base->shape, base->rank);
    
    TensorType dtype_base = base->dtype;
    TensorType dtype_exponent = exponent->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_base, dtype_exponent);

    // Create an output tensor with the same shape and winning data type
    Tensor* result = nnl2_empty(base->shape, base->rank, winner_in_the_type_hierarchy);
    
    if (result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    if (dtype_base == dtype_exponent) {
        // Handling the case if the data types match
        
        switch (dtype_base) {
            case FLOAT64: {
                volatile double* data_base = (double*)base->data;
                volatile double* data_exponent = (double*)exponent->data;
                volatile double* data_result = (double*)result->data;
            
                // Element-wise exponentiation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = pow(data_base[i], data_exponent[i]);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_base = (float*)base->data;
                volatile float* data_exponent = (float*)exponent->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise exponentiation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = powf(data_base[i], data_exponent[i]);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_base = (int32_t*)base->data;
                volatile int32_t* data_exponent = (int32_t*)exponent->data;
                volatile int32_t* data_result = (int32_t*)result->data;
        
                // Element-wise exponentiation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = (int32_t)pow(data_base[i], data_exponent[i]);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_base);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_result = (double*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_base = (char*)base->data + i * get_dtype_size(dtype_base);
                    void* elem_exponent = (char*)exponent->data + i * get_dtype_size(dtype_exponent);
                    
                    data_result[i] = pow(nnl2_convert_to_float64(elem_base, dtype_base), nnl2_convert_to_float64(elem_exponent, dtype_exponent));
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_base = (char*)base->data + i * get_dtype_size(dtype_base);
                    void* elem_exponent = (char*)exponent->data + i * get_dtype_size(dtype_exponent);
                    
                    data_result[i] = powf(nnl2_convert_to_float32(elem_base, dtype_base), nnl2_convert_to_float32(elem_exponent, dtype_exponent));
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_base = (char*)base->data + i * get_dtype_size(dtype_base);
                    void* elem_exponent = (char*)exponent->data + i * get_dtype_size(dtype_exponent);
                    
                    data_result[i] = (int32_t)pow(nnl2_convert_to_int32(elem_base, dtype_base), nnl2_convert_to_int32(elem_exponent, dtype_exponent));
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
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
 * @brief Backend implementations for power operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - naive_pow: Basic reference implementation
 * 
 * @see naive_pow
 */
Implementation pow_backends[] = {
	REGISTER_BACKEND(naive_pow, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for power operation
 * @ingroup backend_system 
 */
powfn nnl2_pow;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(pow);

/** 
 * @brief Sets the backend for power operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_pow_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(pow_backends, pow, backend_name, CURRENT_BACKEND(pow));
}

/** 
 * @brief Gets the name of the active backend for power operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_pow_backend() {
	return CURRENT_BACKEND(pow);
}

/** 
 * @brief Function declaration for getting all `pow` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(pow);

/**
 * @brief Function declaration for getting the number of all `pow` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(pow);

/** @brief
 * Calculates the exponent of each tensor element in place
 *
 ** @details
 * The function applies the exponential function (e^x) to each element of the tensor,
 * replacing the original values with the calculated results
 *
 ** @param tensor
 * Pointer to a tensor for processing. The function modifies
 * the tensor data in place
 *
 ** @see exp
 ** @see expf
 **/
void nnl2_naive_expinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = product(tensor->shape, tensor->rank);
	
	// If tensor empty, exiting the function
	if(len == 0) return;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "A NULL tensor was passed to .exp!");
	#endif
	
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
			
			// 0 is the only integer for which exp gives an integer
			for (size_t it = 0; it < len; it++) {
				if (tensor_data[it] != 0) {
					NNL2_FATAL("Can't apply .exp! to a passed tensor");
				}
			}
			
			for (size_t it = 0; it < len; it++) {
				tensor_data[it] = 1;
			}
			
			break;	
		}
			
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}	

/** 
 * @ingroup backend_system
 * @brief Backend implementations for exponential in-place operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_expinplace: Basic reference implementation
 * 
 * @see nnl2_naive_expinplace
 */
Implementation expinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_expinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for exponential in-place operation
 * @ingroup backend_system 
 */
expinplacefn expinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(expinplace);

/** 
 * @brief Sets the backend for exponential in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_expinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(expinplace_backends, expinplace, backend_name, CURRENT_BACKEND(expinplace));
}

/** 
 * @brief Gets the name of the active backend for exponential in-place operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_expinplace_backend() {
	return CURRENT_BACKEND(expinplace);
}

/** 
 * @brief Function declaration for getting all `expinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(expinplace);

/**
 * @brief Function declaration for getting the number of all `expinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(expinplace);

/** @brief 
 * Naive implementation of exponential operation
 *
 ** @details
 * Computes element-wise exponential function e^x for each element in the input tensor
 *
 ** @param tensor 
 * Input tensor for exponential operation
 *
 ** @param save_type 
 * Flag to save data type for special case (all elements = 0)
 * true - return INT32 tensor with ones
 * false - return FLOAT64 tensor with ones
 *
 ** @return 
 * New tensor with exponential values applied element-wise
 */
Tensor* naive_exp(Tensor* tensor, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
		
	size_t len = product(tensor->shape, tensor->rank);
	
	// Processing a tensor with an integer data type of INT32.
	// Calculates the exponent for each element, but only if the tensor
	// has at least one non-zero element
	if (tensor->dtype == INT32) {
		int32_t* tensor_data = (int32_t*)tensor->data;
		bool has_non_zero = false;
		
		// Check whether the tensor has at least one non-zero element
		for (size_t it = 0; it < len; it++) {
			if (tensor_data[it] != 0) {
				has_non_zero = true;
				break;
			}
		}
		
		if (has_non_zero) {
			Tensor* result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
			double* result_data = (double*)result->data;
			
			for (size_t it = 0; it < len; it++) {
				result_data[it] = exp((double)tensor_data[it]);
			}
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
			return result;
		} else {
			if(save_type) {
				return nnl2_ones(tensor->shape, tensor->rank, INT32);
			} else {
				return nnl2_ones(tensor->shape, tensor->rank, FLOAT64);
			}
		}
	}
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	if(len == 0) return result;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			volatile double* tensor_data = (double*)tensor->data;
			volatile double* result_data = (double*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = exp(tensor_data[it]);
			break;
		}
		
		case FLOAT32: {
			volatile float* tensor_data = (float*)tensor->data;
			volatile float* result_data = (float*)result->data;
			for(size_t it = 0; it < len; it++) result_data[it] = expf(tensor_data[it]);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for exponential operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - naive_exp: Basic reference implementation
 * 
 * @see naive_exp
 */
Implementation exp_backends[] = {
	REGISTER_BACKEND(naive_exp, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for exponential operation
 * @ingroup backend_system 
 */
expfn nnl2_exp;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(exp);

/** 
 * @brief Sets the backend for exponential operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_exp_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(exp_backends, exp, backend_name, current_backend(exp));
}

/** 
 * @brief Gets the name of the active backend for exponential operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_exp_backend() {
	return current_backend(exp);
}

/** 
 * @brief Function declaration for getting all `exp` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(exp);

/**
 * @brief Function declaration for getting the number of all `exp` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(exp);

/** @brief
 * Calculates the natural logarithm of each tensor element in place
 *
 ** @details
 * The function applies the natural logarithm function (ln(x)) to each element of the tensor,
 * replacing the original values with the calculated results
 *
 ** @param tensor
 * Pointer to a tensor for processing. The function modifies
 * the tensor data in place
 *
 ** @see log
 ** @see logf
 **/
void nnl2_naive_loginplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = product(tensor->shape, tensor->rank);
	
	// If tensor empty, exiting the function
	if(len == 0) return;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "A NULL tensor was passed to .log!");
	#endif
	
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
			
			// 1 is the only integer for which log gives an integer
			for(size_t it = 0; it < len; it++) {
				if (tensor_data[it] != 1) {
					NNL2_FATAL("Can't apply .log! to passed tensor");
				}
			}
			
			for(size_t it = 0; it < len; it++) {
				tensor_data[it] = 0;
			}
			
			break;	
		}
			
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}	

/** 
 * @ingroup backend_system
 * @brief Backend implementations for logarithm in-place operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_loginplace: Basic reference implementation
 * 
 * @see nnl2_naive_loginplace
 */
Implementation loginplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_loginplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for logarithm in-place operation
 * @ingroup backend_system 
 */
loginplacefn loginplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(loginplace);

/** 
 * @brief Sets the backend for logarithm in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_loginplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(loginplace_backends, loginplace, backend_name, CURRENT_BACKEND(loginplace));
}

/** 
 * @brief Gets the name of the active backend for logarithm in-place operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_loginplace_backend() {
	return CURRENT_BACKEND(loginplace);
}

/** 
 * @brief Function declaration for getting all `loginplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(loginplace);

/**
 * @brief Function declaration for getting the number of all `loginplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(loginplace);

/** @brief
 * Calculates the natural logarithm of the elements of the input tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param save_type 
 * Flag to save data type for special case (all elements = 1)
 * true - return INT32 tensor with zeros
 * false - return FLOAT64 tensor with zeros
 *
 ** @return
 * A pointer to a new tensor with the result of calculating the logarithm
 */
Tensor* naive_log(const Tensor* tensor, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t len = product(tensor->shape, tensor->rank);
	
    if (tensor->dtype == INT32) {
        int32_t* tensor_data = (int32_t*)tensor->data;
        int has_non_ones = 0;
        
        // Checking if there are elements other than 1
        for (size_t it = 0; it < len; it++) {
            if (tensor_data[it] != 1) {
                has_non_ones = 1;
                break;
            }
        }
        
        if (has_non_ones) {
            Tensor* result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
            double* result_data = (double*)result->data;
            
            for (size_t it = 0; it < len; it++) {
                result_data[it] = log((double)tensor_data[it]);
            }
			
            return result;
        } else {
			if(save_type) {
				return nnl2_zeros(tensor->shape, tensor->rank, INT32);
			} else {
				return nnl2_zeros(tensor->shape, tensor->rank, FLOAT64);
			}
        }
    }
	
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
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for logarithm operation
 * @details
 * Array follows the common backend registration pattern for logarithm operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation log_backends[] = {
	REGISTER_BACKEND(naive_log, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for logarithm operation
 * @ingroup backend_system 
 */
logfn nnl2_logarithm;

/** 
 * @brief Makes the logarithm backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(log);

/** 
 * @brief Sets the backend for logarithm operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_log_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(log_backends, log, backend_name, current_backend(log));
}

/** 
 * @brief Gets the name of the active backend for logarithm operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_log_backend() {
	return current_backend(log);
}

/** 
 * @brief Function declaration for getting all available logarithm backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(log);

/**
 * @brief Function declaration for getting the number of available logarithm backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(log);

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
 * Scales the tensor in-place using a multiplier
 * The function multiplies each element of the tensor by a specified multiplier
 *
 ** @param tensor 	
 * Pointer to a tensor for scaling
 *
 ** @param multiplier
 * Multiplier for scaling
 *
 ** @note
 * For integer types, the multiplier must be an integer
 *
 ** @throws NNL2_ERROR
 * If you try to multiply an INT32 tensor by a fractional number
 */
void naive_scaleinplace(Tensor* tensor, float multiplier) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	void* data = tensor->data;
	
	// Calculate the total number of elements in the tensor
	int num_elems = product(tensor->shape, tensor->rank);
	if(num_elems == 0) return; // If the tensor is empty, exit
	
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
			
			// Check that the multiplier is integer for the INT32 tensor
			if(round(multiplier) != multiplier) {
				NNL2_ERROR("Can't multiply an int32 tensor by a fractional number");
				return;
			}
			
			for(int i = 0; i < num_elems; i++) data_t[i] = (int32_t)round(data_t[i] * multiplier);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for scaleinplace operation
 * @details
 * Array follows the common backend registration pattern for scaleinplace operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation scaleinplace_backends[] = {
	REGISTER_BACKEND(naive_scaleinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for scaleinplace operation
 * @ingroup backend_system 
 */
scaleinplacefn scaleinplace;

/** 
 * @brief Makes the scaleinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(scaleinplace);

/** 
 * @brief Sets the backend for scaleinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_scaleinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(scaleinplace_backends, scaleinplace, backend_name, current_backend(scaleinplace));
}

/** 
 * @brief Gets the name of the active backend for scaleinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_scaleinplace_backend() {
	return current_backend(scaleinplace);
}

/** 
 * @brief Function declaration for getting all available scaleinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(scaleinplace);

/**
 * @brief Function declaration for getting the number of available scaleinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(scaleinplace);

/** @brief
 * Multiplies a tensor by a scalar factor
 *
 * Special cases (multiplication by 0 and 1) are handled optimally by creating
 * a tensor with zeros or copying the original tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param multiplier 
 * A scalar multiplier for multiplying tensor elements
 *
 ** @param save_type 
 * The flag for saving the data type (for example, by default, int32 
 * is cast to float64. save_type attempts to preserve the type if possible)
 *
 ** @see nnl2_copy
 ** @see nnl2_zeros
 ** @see nnl2_empty
 ** @see nnl2_free_tensor
 **/
Tensor* nnl2_naive_scale(const Tensor* tensor, float multiplier, bool save_type) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "Passed tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "Tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->shape, "Tensor shape is NULL", NULL);
	#endif
	
	Tensor* result;
	void* data_original = tensor->data;
	
	// Calculate the total number of elements in the tensor
	size_t num_elems = product(tensor->shape, tensor->rank);
	TensorType dtype = tensor->dtype;
	
	if (multiplier == 1.0f) {
		// For multiplication by 1, just return a copy
		TensorType output_dtype = (dtype == INT32 && !save_type) ? FLOAT64 : dtype;
		return nnl2_copy(tensor, output_dtype);
	}
	
	if (multiplier == 0.0f) {
		// For multiplication by 0, return a tensor of zeros
		TensorType output_dtype = (dtype == INT32 && !save_type) ? FLOAT64 : dtype;
		return nnl2_zeros(tensor->shape, tensor->rank, output_dtype);
	}
	
	if(dtype == INT32) {
		int32_t* data_t = (int32_t*)data_original;
		
		if(save_type) {
			// Scale integer data, round, and keep as INT32
			result = nnl2_empty(tensor->shape, tensor->rank, INT32);
			int32_t* data_o = (int32_t*)result->data;
			for(size_t i = 0; i < num_elems; i++) {
				data_o[i] = (int32_t)lround(data_t[i] * multiplier);
			}
			return result;
		} else {
			// Scale integer data and promote to FLOAT64
			result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
			double* data_o = (double*)result->data;
			for(size_t i = 0; i < num_elems; i++) {
				data_o[i] = data_t[i] * multiplier;
			}
		}
		
		return result;
	}
	
	result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	void* data_result = result->data;
	
	switch(dtype) {
		case FLOAT64: {
			double* data_t = (double*)data_original;
			double* data_o = (double*)data_result;
			for(size_t i = 0; i < num_elems; i++) data_o[i] = data_t[i] * (double)multiplier;
			break;
		}
		
		case FLOAT32: {
			float* data_t = (float*)data_original;
			float* data_o = (float*)data_result;
			for(size_t i = 0; i < num_elems; i++) data_o[i] = data_t[i] * multiplier;
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(dtype);
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
 * @brief Backend implementations for scale operation
 * @details
 * Array follows the common backend registration pattern for scale operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation scale_backends[] = {
	REGISTER_BACKEND(nnl2_naive_scale, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for scale operation
 * @ingroup backend_system 
 */
scalefn scale;

/** 
 * @brief Makes the scale backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(scale);

/** 
 * @brief Sets the backend for scale operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_scale_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(scale_backends, scale, backend_name, current_backend(scale));
}

/** 
 * @brief Gets the name of the active backend for scale operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_scale_backend() {
	return current_backend(scale);
}

/** 
 * @brief Function declaration for getting all available scale backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(scale);

/**
 * @brief Function declaration for getting the number of available scale backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(scale);

/** @brief
 * Creates a new tensor of the same shape and type, but with uninitialized data
 *
 ** @param tensor
 * The initial tensor from which a new tensor is created
 *
 ** @return
 * Tensor* Pointer to a new tensor with uninitialized data
 *
 ** @note
 * The data in the returned tensor is not initialized and may contain garbage
 *
 ** @see nnl2_empty
 **/
Tensor* nnl2_empty_like(Tensor* tensor) {
	return nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
}

/** @brief
 * Creates a new tensor of the same shape and type, filled with zeros
 *
 ** @param tensor
 * The initial tensor from which a new tensor is created
 *
 ** @return
 * Tensor* A pointer to a new tensor filled with zeros
 *
 ** @note
 * All elements of the returned tensor are initialized with zero values
 *
 ** @see nnl2_zeros
 **/
Tensor* nnl2_zeros_like(Tensor* tensor) {
	return nnl2_zeros(tensor->shape, tensor->rank, tensor->dtype);
}

/** @brief 
 * Creates a new tensor of the same shape and type, filled with units
 *
 ** @param tensor
 * The initial tensor from which a new tensor is created
 *
 ** @return
 * Tensor* A pointer to a new tensor filled with ones
 *
 ** @note 
 * All elements of the returned tensor are initialized with ones
 *
 ** @see nnl2_ones
 **/
Tensor* nnl2_ones_like(Tensor* tensor) {
	return nnl2_ones(tensor->shape, tensor->rank, tensor->dtype);
}

/** @brief
 * Creates a new tensor of the same shape and type, filled with the specified value
 *
 ** @param tensor
 * The initial tensor from which a new tensor is created
 *
 ** @param filler
 * A pointer to the value that will be filled into the tensor
 *
 ** @return 
 * Tensor* Pointer to a new tensor filled with the specified value
 * 
 ** @note
 * The filler value must be of the correct dtype tensor type
 *
 ** @warning
 * It is necessary to ensure that the filler value and tensor type match
 *
 ** @see nnl2_full
 **/
Tensor* nnl2_full_like(Tensor* tensor, void* filler) {
	return nnl2_full(tensor->shape, tensor->rank, tensor->dtype, filler);
}

/** @brief 
 * Performs element-wise maximum of two tensors (naive implementation)
 * 
 * Compares elements of the first tensor with corresponding elements 
 * of the second tensor, storing the maximum value in the first tensor
 *
 ** @param tensora 
 * Pointer to the tensor that will be modified 
 *
 ** @param tensorb 
 * Pointer to the tensor whose values will be used for comparison
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The second tensor's elements are converted to the first tensor's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the first tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_tensor((float[]){2.0, 3.0, 4.0}, (int[]){3}, 1, FLOAT32);
 * Tensor* b = nnl2_tensor((float[]){3.0, 1.0, 5.0}, (int[]){3}, 1, FLOAT32);
 * 
 * // Store element-wise maximum in tensor a
 * naive_maxinplace(a, b);
 * 
 * // Now a contains [3.0, 3.0, 5.0]
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_maxinplace(Tensor* tensora, Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora, "First tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora->data, "First tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb, "Second tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb->data, "Second tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the first tensor
    size_t total_elems = product(tensora->shape, tensora->rank);
    
    // If the tensor is empty, exit the function
    if(total_elems == 0) return;
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    if(typea == typeb) {
        // Handling case when the tensors have the same type
        
        switch(typea) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                volatile double* data_b = (double*)tensorb->data;
                
                // Element-wise maximum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MAX(data_a[i], data_b[i]);
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                volatile float* data_b = (float*)tensorb->data;
                
                // Element-wise maximum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MAX(data_a[i], data_b[i]);
                }
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                volatile int32_t* data_b = (int32_t*)tensorb->data;
                
                // Element-wise maximum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MAX(data_a[i], data_b[i]);
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(typea);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing second tensor elements
        size_t typeb_step = get_dtype_size(typeb);
        
        // Casting second tensor data to char* for byte access
        char* data_b = (char*)tensorb->data;
        
        switch(typea) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                
                // For each element, convert the second tensor element to FLOAT64 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    data_a[i] = MAX(data_a[i], nnl2_convert_to_float64(elem_b, typeb));
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                
                // For each element, convert the second tensor element to FLOAT32 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    data_a[i] = MAX(data_a[i], nnl2_convert_to_float32(elem_b, typeb));
                }
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                
                // For each element, convert the second tensor element to INT32 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    data_a[i] = MAX(data_a[i], nnl2_convert_to_int32(elem_b, typeb));
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(typea);
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
 * @brief Backend implementations for maxinplace operation
 * @details
 * Array follows the common backend registration pattern for element-wise maximum operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation maxinplace_backends[] = {
	REGISTER_BACKEND(naive_maxinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for maxinplace operation
 * @ingroup backend_system 
 */
maxinplacefn maxinplace;

/** 
 * @brief Makes the maxinplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(maxinplace);

/** 
 * @brief Sets the backend for maxinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_maxinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(maxinplace_backends, maxinplace, backend_name, CURRENT_BACKEND(maxinplace));
}

/** 
 * @brief Gets the name of the active backend for maxinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_maxinplace_backend() {
	return CURRENT_BACKEND(maxinplace);
}

/** 
 * @brief Function declaration for getting all available maxinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(maxinplace);

/**
 * @brief Function declaration for getting the number of available maxinplace backends
 * @ingroup backend_system
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(maxinplace);

/** @brief 
 * Performs element-wise minimum of two tensors (naive implementation)
 * 
 * Compares elements of the first tensor with corresponding elements 
 * of the second tensor, storing the minimum value in the first tensor
 *
 ** @param tensora 
 * Pointer to the tensor that will be modified (receives the minimum values)
 *
 ** @param tensorb 
 * Pointer to the tensor whose values will be used for comparison
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The second tensor's elements are converted to the first tensor's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the first tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_tensor((float[]){2.0, 3.0, 4.0}, (int[]){3}, 1, FLOAT32);
 * Tensor* b = nnl2_tensor((float[]){3.0, 1.0, 5.0}, (int[]){3}, 1, FLOAT32);
 * 
 * // Store element-wise minimum in tensor a
 * naive_mininplace(a, b);
 * 
 * // Now a contains [2.0, 1.0, 4.0]
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void naive_mininplace(Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora, "First tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensora->data, "First tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb, "Second tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensorb->data, "Second tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the first tensor
    size_t total_elems = product(tensora->shape, tensora->rank);
    
    // If the tensor is empty, exit the function
    if(total_elems == 0) return;
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    if(typea == typeb) {
        // Handling case when the tensors have the same type
        
        switch(typea) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                volatile double* data_b = (double*)tensorb->data;
                
                // Element-wise minimum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MIN(data_a[i], data_b[i]);
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                volatile float* data_b = (float*)tensorb->data;
                
                // Element-wise minimum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MIN(data_a[i], data_b[i]);
                }
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                volatile int32_t* data_b = (int32_t*)tensorb->data;
                
                // Element-wise minimum calculation
                for(size_t i = 0; i < total_elems; i++) {
                    data_a[i] = MIN(data_a[i], data_b[i]);
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(typea);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing second tensor elements
        size_t typeb_step = get_dtype_size(typeb);
        
        // Casting second tensor data to char* for byte access
        char* data_b = (char*)tensorb->data;
        
        switch(typea) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                
                // For each element, convert the second tensor element to FLOAT64 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    double converted_b = nnl2_convert_to_float64(elem_b, typeb);
                    data_a[i] = MIN(data_a[i], converted_b);
                }
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                
                // For each element, convert the second tensor element to FLOAT32 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    float converted_b = nnl2_convert_to_float32(elem_b, typeb);
                    data_a[i] = MIN(data_a[i], converted_b);
                }
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                
                // For each element, convert the second tensor element to INT32 and compare
                for(size_t i = 0; i < total_elems; i++) {
                    void* elem_b = data_b + i * typeb_step;
                    int32_t converted_b = nnl2_convert_to_int32(elem_b, typeb);
                    data_a[i] = MIN(data_a[i], converted_b);
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(typea);
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
 * @brief Backend implementations for mininplace operation
 * @details
 * Array follows the common backend registration pattern for element-wise minimum operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation mininplace_backends[] = {
	REGISTER_BACKEND(naive_mininplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for mininplace operation
 * @ingroup backend_system 
 */
mininplacefn mininplace;

/** 
 * @brief Makes the mininplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(mininplace);

/** 
 * @brief Sets the backend for mininplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_mininplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mininplace_backends, mininplace, backend_name, current_backend(mininplace));
}

/** 
 * @brief Gets the name of the active backend for mininplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_mininplace_backend() {
	return current_backend(mininplace);
}

/** 
 * @brief Function declaration for getting all available mininplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mininplace);

/**
 * @brief Function declaration for getting the number of available mininplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mininplace);

// why am I doing this

/** @brief
 * Performs element-wise maximum of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the result of taking the maximum
 * of each element from the first tensor and the corresponding element in the second tensor.
 * It supports various data types with automatic casting to a higher type in the hierarchy.
 *
 ** @param tensora
 * Pointer to the first tensor
 *
 ** @param tensorb
 * Pointer to the second tensor
 *
 ** @return 
 * Pointer to a new tensor with the maximum values
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or unsupported data type
 *
 ** @note
 * The result tensor has the same shape as input tensors and the highest data type
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_max(const Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks on maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora, "First tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora->data, "First tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb, "Second tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb->data, "Second tensor data is NULL", NULL);
	#endif
	
    // Calculate the total number of elements in the tensors
    size_t len = product(tensora->shape, tensora->rank);
    
    TensorType dtype_a = tensora->dtype;
    TensorType dtype_b = tensorb->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_a, dtype_b);

    // Create an output tensor with the same shape and winning data type
    Tensor* result = nnl2_empty(tensora->shape, tensora->rank, winner_in_the_type_hierarchy);
    
    if (result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    if (dtype_a == dtype_b) {
        // Handling the case if the data types match
        
        switch (dtype_a) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                volatile double* data_b = (double*)tensorb->data;
                volatile double* data_result = (double*)result->data;
            
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                volatile float* data_b = (float*)tensorb->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                volatile int32_t* data_b = (int32_t*)tensorb->data;
                volatile int32_t* data_result = (int32_t*)result->data;
        
                // Element-wise maximum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MAX(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_a);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_result = (double*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_float64(elem_a, dtype_a), nnl2_convert_to_float64(elem_b, dtype_b));
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_float32(elem_a, dtype_a), nnl2_convert_to_float32(elem_b, dtype_b));
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MAX(nnl2_convert_to_int32(elem_a, dtype_a), nnl2_convert_to_int32(elem_b, dtype_b));
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
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
 * @brief Backend implementations for max operation
 * @details
 * Array follows the common backend registration pattern for element-wise maximum operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation max_backends[] = {
	REGISTER_BACKEND(naive_max, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for max operation
 * @ingroup backend_system 
 */
maxfn nnl2_max;

/** 
 * @brief Makes the max backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(max);

/** 
 * @brief Sets the backend for max operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_max_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(max_backends, nnl2_max, backend_name, current_backend(max));
}

/** 
 * @brief Gets the name of the active backend for max operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_max_backend() {
	return current_backend(max);
}

/** 
 * @brief Function declaration for getting all available max backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(max);

/**
 * @brief Function declaration for getting the number of available max backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(max);

/** @brief
 * Performs element-wise minimum of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the result of taking the minimum
 * of each element from the first tensor and the corresponding element in the second tensor.
 * It supports various data types with automatic casting to a higher type in the hierarchy.
 *
 ** @param tensora
 * Pointer to the first tensor
 *
 ** @param tensorb
 * Pointer to the second tensor
 *
 ** @return 
 * Pointer to a new tensor with the minimum values
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or unsupported data type
 *
 ** @note
 * The result tensor has the same shape as input tensors and the highest data type
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_min(const Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Additional checks on maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora, "First tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora->data, "First tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb, "Second tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb->data, "Second tensor data is NULL", NULL);
	#endif
	
    // Calculate the total number of elements in the tensors
    size_t len = product(tensora->shape, tensora->rank);
    
    TensorType dtype_a = tensora->dtype;
    TensorType dtype_b = tensorb->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_a, dtype_b);

    // Create an output tensor with the same shape and winning data type
    Tensor* result = nnl2_empty(tensora->shape, tensora->rank, winner_in_the_type_hierarchy);
    
    if (result == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    if (dtype_a == dtype_b) {
        // Handling the case if the data types match
        
        switch (dtype_a) {
            case FLOAT64: {
                volatile double* data_a = (double*)tensora->data;
                volatile double* data_b = (double*)tensorb->data;
                volatile double* data_result = (double*)result->data;
            
                // Element-wise minimum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MIN(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_a = (float*)tensora->data;
                volatile float* data_b = (float*)tensorb->data;
                volatile float* data_result = (float*)result->data;
        
                // Element-wise minimum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MIN(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_a = (int32_t*)tensora->data;
                volatile int32_t* data_b = (int32_t*)tensorb->data;
                volatile int32_t* data_result = (int32_t*)result->data;
        
                // Element-wise minimum calculation
                for (size_t i = 0; i < len; i++) {
                    data_result[i] = MIN(data_a[i], data_b[i]);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_a);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_result = (double*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MIN(nnl2_convert_to_float64(elem_a, dtype_a), nnl2_convert_to_float64(elem_b, dtype_b));
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_result = (float*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MIN(nnl2_convert_to_float32(elem_a, dtype_a), nnl2_convert_to_float32(elem_b, dtype_b));
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_result = (int32_t*)result->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_a = (char*)tensora->data + i * get_dtype_size(dtype_a);
                    void* elem_b = (char*)tensorb->data + i * get_dtype_size(dtype_b);
                    
                    data_result[i] = MIN(nnl2_convert_to_int32(elem_a, dtype_a), nnl2_convert_to_int32(elem_b, dtype_b));
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
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
 * @brief Backend implementations for min operation
 * @details
 * Array follows the common backend registration pattern for element-wise minimum operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation min_backends[] = {
	REGISTER_BACKEND(naive_min, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for min operation
 * @ingroup backend_system 
 */
minfn nnl2_min;

/** 
 * @brief Makes the min backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(min);

/** 
 * @brief Sets the backend for min operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_min_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(min_backends, nnl2_min, backend_name, current_backend(min));
}

/** 
 * @brief Gets the name of the active backend for min operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_min_backend() {
	return current_backend(min);
}

/** 
 * @brief Function declaration for getting all available min backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(min);

/**
 * @brief Function declaration for getting the number of available min backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(min);

/** @brief 
 * Calculates the absolute values of the tensor elements in place
 *
 ** @param tensor
 * A pointer to a tensor that will be modified
 *
 ** @see fabs
 ** @see fabsf
 ** @see abs
 **/
void naive_absinplace(Tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; // If tensor is empty then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = fabs(cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = fabsf(cast_data[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data = (int32_t*)data;	
			for(size_t i = 0; i < total_elems; i++) cast_data[i] = abs(cast_data[i]);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for absinplace operation
 * @details
 * Array follows the common backend registration pattern for in-place absolute value operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation absinplace_backends[] = {
	REGISTER_BACKEND(naive_absinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for absinplace operation
 * @ingroup backend_system 
 */
absinplacefn absinplace;

/** 
 * @brief Makes the absinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(absinplace);

/** 
 * @brief Sets the backend for absinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_absinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(absinplace_backends, absinplace, backend_name, current_backend(absinplace));
}

/** 
 * @brief Gets the name of the active backend for absinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_absinplace_backend() {
	return current_backend(absinplace);
}

/** 
 * @brief Function declaration for getting all available absinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(absinplace);

/**
 * @brief Function declaration for getting the number of available absinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(absinplace);

/** @brief
 * Naive implementation of absolute value operation
 *
 ** @param tensor 
 * Input tensor
 *
 ** @return 
 * New tensor with absolute values of input elements
 */
Tensor* naive_abs(Tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	size_t total_elems = product(tensor->shape, tensor->rank);	
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	if(total_elems == 0) return result;
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = fabs(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = fabsf(cast_data_t[i]);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = abs(cast_data_t[i]);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for abs operation
 * @details
 * Array follows the common backend registration pattern for absolute value operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation abs_backends[] = {
	REGISTER_BACKEND(naive_abs, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for abs operation
 * @ingroup backend_system 
 */
absfn nnl2_abs;

/** 
 * @brief Makes the abs backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(abs);

/** 
 * @brief Sets the backend for abs operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_abs_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(abs_backends, nnl2_abs, backend_name, current_backend(abs));
}

/** 
 * @brief Gets the name of the active backend for abs operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_abs_backend() {
	return current_backend(abs);
}

/** 
 * @brief Function declaration for getting all available abs backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(abs);

/**
 * @brief Function declaration for getting the number of available abs backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(abs);

/** @brief
 * Get the raw data pointer from a tensor
 *
 ** @param tensor
 * Pointer to the tensor
 *
 ** @return
 * Void pointer to the tensor's underlying data storage
 */
void* get_tensor_data(Tensor* tensor) {
	return tensor->data;
}

/** @brief 
 * Performs horizontal stacking of two tensors (naive implementation)
 *
 ** @param tensora
 * Pointer to the first input tensor
 *
 ** @param tensorb
 * Pointer to the second input tensor
 *
 ** @return
 * Pointer to a new tensor containing the horizontally stacked result
 *
 ** @note
 * Tensors must have the same rank and compatible shapes (all dimensions
 * except axis=1 must match)
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_hstack(Tensor* tensora, Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    int ranka = tensora->rank;
    int rankb = tensorb->rank;

	// Safety checks for tensor compatibility
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(ranka != rankb) {
			NNL2_ERROR("Tensors dimensions are different");
			return NULL;
		}

		// Check if all dimensions except axis=1 are equal
		for(int i = 0; i < ranka; i++) {
			if(i != 1 && tensora->shape[i] != tensorb->shape[i]) {
				NNL2_ERROR("Tensors shapes are incompatible for hstack");
				return NULL;
			}
		}
	#endif

    TensorType winner_type = MAX(typea, typeb);

	// Allocate memory for result shape
    int* shapec = malloc(ranka * sizeof(int));
    if (shapec == NULL) {
        NNL2_ERROR("Memory allocation failed");
        return NULL; 
    }
    
	// Calculate result shape
    for(int i = 0; i < ranka; i++) {
        if(i == 1) {
            shapec[i] = tensora->shape[i] + tensorb->shape[i];
        } else {
            shapec[i] = tensora->shape[i];
        }
    }
    
	// Create empty result tensor with calculated shape and winning type
    Tensor* result = nnl2_empty(shapec, ranka, winner_type);
    free(shapec);
    
    if(result == NULL) {
        NNL2_ERROR("Failed to create result tensor in hstack");
        return NULL;
    }
    
	// Handle empty tensors case
    size_t total_elements = product(result->shape, result->rank);
    if(total_elements == 0) {
        return result;
    }
    
    size_t shapea_0 = (size_t)tensora->shape[0];
    size_t shapeb_0 = (size_t)tensorb->shape[0];
    
	// Handle 1D tensors
    if(ranka == 1) {
        if(typea == typeb && typea == winner_type) {
			// Same types, use memcpy
            size_t item_size = get_dtype_size(winner_type);
            memcpy(result->data, tensora->data, shapea_0 * item_size);
            memcpy((char*)result->data + shapea_0 * item_size, tensorb->data, shapeb_0 * item_size);
        } else {
		    // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;

					// Convert and copy first tensor
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float64(elem, typea);
                    }
                
					// Convert and copy second tensor
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_float64(elem, typeb);
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Convert and copy first tensor
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float32(elem, typea);
                    }
                    
					// Convert and copy second tensor
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_float32(elem, typeb);
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
					// Convert and copy first tensor
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_int32(elem, typea);
                    }
                    
					// Convert and copy second tensor
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_int32(elem, typeb);
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    free(result);
                    return NULL;
                }
            }
        }
    } else {
		// Handle multi-dimensional tensors
        size_t outer_dim = (size_t)tensora->shape[0];
        size_t inner_elements_a = product(tensora->shape + 1, ranka - 1);
        size_t inner_elements_b = product(tensorb->shape + 1, rankb - 1);
        
        if(typea == typeb && typea == winner_type) {
			// Same types, use memcpy 
            size_t item_size = get_dtype_size(winner_type);
            size_t row_size_a = inner_elements_a * item_size;
            size_t row_size_b = inner_elements_b * item_size;
            
            char* src_a = tensora->data;
            char* src_b = tensorb->data;
            char* dst = result->data;
            
			// Process each outer dimension (e.g., each matrix in a 3D tensor)
            for(size_t i = 0; i < outer_dim; i++) {
				// Copy slice from first tensor
                memcpy(dst, src_a, row_size_a);
                dst += row_size_a;
                src_a += row_size_a;
                
				// Copy slice from second tensor
                memcpy(dst, src_b, row_size_b);
                dst += row_size_b;
                src_b += row_size_b;
            }
        } else {
			// Type conversion needed for multi-dimensional case
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
					// Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
						// Convert and copy slice from first tensor
                        for(size_t j = 0; j < inner_elements_a; j++) {
                            size_t src_idx = i * inner_elements_a + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typea);
                        }
						
                        // Convert and copy slice from second tensor
                        for(size_t j = 0; j < inner_elements_b; j++) {
                            size_t src_idx = i * inner_elements_b + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + inner_elements_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typeb);
                        }
                    }
					
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
						// Convert and copy slice from first tensor
                        for(size_t j = 0; j < inner_elements_a; j++) {
                            size_t src_idx = i * inner_elements_a + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typea);
                        }
                        
						// Convert and copy slice from second tensor
                        for(size_t j = 0; j < inner_elements_b; j++) {
                            size_t src_idx = i * inner_elements_b + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + inner_elements_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typeb);
                        }
                    }
					
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
					// Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
						// Convert and copy slice from first tensor
                        for(size_t j = 0; j < inner_elements_a; j++) {
                            size_t src_idx = i * inner_elements_a + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typea);
                        }
                        
						// Convert and copy slice from second tensor
                        for(size_t j = 0; j < inner_elements_b; j++) {
                            size_t src_idx = i * inner_elements_b + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + inner_elements_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typeb);
                        }
                    }
					
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    free(result);
                    return NULL;
                }
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
 * @brief Backend implementations for hstack operation
 * @details
 * Array follows the common backend registration pattern for horizontal stacking operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for horizontal tensor concatenation
 * 
 * @see nnl2_naive
 * @see naive_hstack
 */
Implementation hstack_backends[] = {
	REGISTER_BACKEND(naive_hstack, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for hstack operation
 * @ingroup backend_system 
 */
hstackfn hstack;

/** 
 * @brief Makes the hstack backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(hstack);

/** 
 * @brief Sets the backend for hstack operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for horizontal stacking
 * @see ESET_BACKEND_BY_NAME
 */
void set_hstack_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(hstack_backends, hstack, backend_name, current_backend(hstack));
}

/** 
 * @brief Gets the name of the active backend for hstack operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_hstack_backend() {
	return current_backend(hstack);
}

/** 
 * @brief Function declaration for getting all available hstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(hstack);

/**
 * @brief Function declaration for getting the number of available hstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 * @details
 * Returns the total number of registered horizontal stacking backend implementations.
 * Useful for iterating through available backends.
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(hstack);

/** @brief
 * Performs vertical stacking of two tensors (naive implementation)
 *
 ** @param tensora 
 * Pointer to the first input tensor
 *
 ** @param tensorb
 * Pointer to the second input tensor
 *
 ** @return 
 * Pointer to a new tensor containing the vertically stacked result
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_vstack(const Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    int ranka = tensora->rank;
    int rankb = tensorb->rank;
    
    TensorType winner_type = MAX(typea, typeb);
    
	// Calculate total number of elements in each tensor
    size_t sizea = product(tensora->shape, tensora->rank);
    size_t sizeb = product(tensorb->shape, tensorb->rank);
    
    void* dataa = tensora->data;
    void* datab = tensorb->data;
    
    int* shapea = tensora->shape;
    int* shapeb = tensorb->shape;
    
    size_t shapea_0 = (size_t)shapea[0];
    size_t shapea_1 = (size_t)(ranka > 1 ? shapea[1] : 0);
	
    size_t shapeb_0 = (size_t)shapeb[0];
    size_t shapeb_1 = (size_t)(rankb > 1 ? shapeb[1] : 0);
    
    Tensor* result = NULL;
    
	// Handle 1D-1D case
    if(ranka == 1 && rankb == 1) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			// Vectors must have same length for vertical stacking
            if(shapea_0 != shapeb_0) {
                NNL2_ERROR("Vectors must have same length for vstack");
                return NULL;
            }
        #endif
        
		// Allocate memory for result shape [2, vector_length]
        int* shapec = malloc(2 * sizeof(int));
        
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
        shapec[0] = 2;
        shapec[1] = (int)shapea_0;
        result = nnl2_empty(shapec, 2, winner_type);
        free(shapec); 
        
        if(typea == typeb && typea == winner_type) {
            // Fast path: same types
            size_t item_size = get_dtype_size(winner_type);
            size_t total_size_a = sizea * item_size;
            size_t total_size_b = sizeb * item_size;
            
            memcpy(result->data, dataa, total_size_a);
            memcpy((char*)result->data + total_size_a, datab, total_size_b);
        } else {
            // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Convert and copy first vector
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float64(elem, typea);
                    }
                    
                    // Convert and copy second vector
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_float64(elem, typeb);
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Convert and copy first vector
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float32(elem, typea);
                    }
                    
					// Convert and copy second vector
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_float32(elem, typeb);
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
					// Convert and copy first vector
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_int32(elem, typea);
                    }
                    
					// Convert and copy second vector
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_int32(elem, typeb);
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    return NULL;
                }
            }
        }
    } 
    
	// Handle 2D-1D case
    else if(ranka == 2 && rankb == 1) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(shapea_1 != shapeb_0) {
                NNL2_ERROR("Matrix columns must match vector length for vstack");
                return NULL;
            }
        #endif
        
		// Allocate memory for result shape [rows+1, columns]
        int* shapec = malloc(2 * sizeof(int));
        
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
        shapec[0] = (int)shapea_0 + 1;
        shapec[1] = (int)shapea_1;
        
        result = nnl2_empty(shapec, 2, winner_type);
        free(shapec);
        
        if(typea == typeb && typea == winner_type) {
            // Same types
            size_t item_size = get_dtype_size(winner_type);
            size_t row_size = shapea_1 * item_size;
            
			// Copy matrix data
            memcpy(result->data, dataa, shapea_0 * row_size);
			
			// Append vector as new row
            memcpy((char*)result->data + shapea_0 * row_size, datab, row_size);
        } else {
            // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Copy and convert matrix
                    for(size_t i = 0; i < shapea_0; i++) {
                        for(size_t j = 0; j < shapea_1; j++) {
                            size_t src_idx = i * shapea_1 + j;
                            size_t dst_idx = i * shapea_1 + j;
                            
                            void* elem = (char*)dataa + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typea);
                        }
                    }
                    
                    // Copy and convert vector
                    for(size_t j = 0; j < shapeb_0; j++) {
                        size_t dst_idx = shapea_0 * shapea_1 + j;
                        void* elem = (char*)datab + j * get_dtype_size(typeb);
                        dst[dst_idx] = nnl2_convert_to_float64(elem, typeb);
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Process matrix rows with type conversion
                    for(size_t i = 0; i < shapea_0; i++) {
                        for(size_t j = 0; j < shapea_1; j++) {
                            size_t src_idx = i * shapea_1 + j;
                            size_t dst_idx = i * shapea_1 + j;
                            
                            void* elem = (char*)dataa + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typea);
                        }
                    }
                    
					// Process vector with type conversion
                    for(size_t j = 0; j < shapeb_0; j++) {
                        size_t dst_idx = shapea_0 * shapea_1 + j;
                        void* elem = (char*)datab + j * get_dtype_size(typeb);
                        dst[dst_idx] = nnl2_convert_to_float32(elem, typeb);
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
					// Convert matrix elements to int32
                    for(size_t i = 0; i < shapea_0; i++) {
                        for(size_t j = 0; j < shapea_1; j++) {
                            size_t src_idx = i * shapea_1 + j;
                            size_t dst_idx = i * shapea_1 + j;
                            
                            void* elem = (char*)dataa + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typea);
                        }
                    }
                    
					// Convert vector elements to int32
                    for(size_t j = 0; j < shapeb_0; j++) {
                        size_t dst_idx = shapea_0 * shapea_1 + j;
                        void* elem = (char*)datab + j * get_dtype_size(typeb);
                        dst[dst_idx] = nnl2_convert_to_int32(elem, typeb);
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    return NULL;
                }
            }
        }
    } 
	
	// Handle 1D-2D case
    else if(ranka == 1 && rankb == 2) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			// Vector length must match matrix columns
            if(shapea_0 != shapeb_1) {
                NNL2_ERROR("Vector length must match matrix columns for vstack");
                return NULL;
            }
        #endif
        
		// Allocate memory for result shape [rows+1, columns]
        int* shapec = malloc(2 * sizeof(int));
        
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
        shapec[0] = (int)shapeb_0 + 1;
        shapec[1] = (int)shapeb_1;
        
        result = nnl2_empty(shapec, 2, winner_type);
        free(shapec);
        
        if(typea == typeb && typea == winner_type) {
            // Same types
            size_t item_size = get_dtype_size(winner_type);
            size_t row_size = shapeb_1 * item_size;
            
			// Copy vector as first row
            memcpy(result->data, dataa, row_size);
			
			// Copy matrix data after the vector
            memcpy((char*)result->data + row_size, datab, shapeb_0 * row_size);
        } else {
            // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Copy and convert vector as first row
                    for(size_t j = 0; j < shapea_0; j++) {
                        void* elem = (char*)dataa + j * get_dtype_size(typea);
                        dst[j] = nnl2_convert_to_float64(elem, typea);
                    }
                    
                    // Copy and convert matrix rows
                    for(size_t i = 0; i < shapeb_0; i++) {
                        for(size_t j = 0; j < shapeb_1; j++) {
                            size_t src_idx = i * shapeb_1 + j;
                            size_t dst_idx = (i + 1) * shapeb_1 + j;
                            
                            void* elem = (char*)datab + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typeb);
                        }
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Convert vector to float32 for first row
                    for(size_t j = 0; j < shapea_0; j++) {
                        void* elem = (char*)dataa + j * get_dtype_size(typea);
                        dst[j] = nnl2_convert_to_float32(elem, typea);
                    }
                    
					// Convert matrix to float32 for remaining rows
                    for(size_t i = 0; i < shapeb_0; i++) {
                        for(size_t j = 0; j < shapeb_1; j++) {
                            size_t src_idx = i * shapeb_1 + j;
                            size_t dst_idx = (i + 1) * shapeb_1 + j;
                            
                            void* elem = (char*)datab + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typeb);
                        }
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
                    // Convert vector to int32 for first row
					for(size_t j = 0; j < shapea_0; j++) {
                        void* elem = (char*)dataa + j * get_dtype_size(typea);
                        dst[j] = nnl2_convert_to_int32(elem, typea);
                    }
                    
					// Convert matrix to int32 for remaining rows
                    for(size_t i = 0; i < shapeb_0; i++) {
                        for(size_t j = 0; j < shapeb_1; j++) {
                            size_t src_idx = i * shapeb_1 + j;
                            size_t dst_idx = (i + 1) * shapeb_1 + j;
                            
                            void* elem = (char*)datab + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typeb);
                        }
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    return NULL;
                }
            }
        }
    } 

	// Handle general ND-ND case
    else {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN	
            if(ranka != rankb) {
                NNL2_ERROR("Tensors must have same rank for general vstack");
                return NULL;
            }
            
			// All dimensions except axis=0 must match
            for(int i = 1; i < ranka; i++) {
                if(shapea[i] != shapeb[i]) {
                    NNL2_ERROR("Tensors must have same dimensions except axis=0 for vstack");
                    return NULL;
                }
            }
        #endif
        
		// Allocate memory for result shape
        int* shapec = malloc(ranka * sizeof(int));
        
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
		// Concatenate along axis=0, keep other dimensions
        shapec[0] = (int)(shapea_0 + shapeb_0);
        
        for(int i = 1; i < ranka; i++) {
            shapec[i] = shapea[i];
        }

        result = nnl2_empty(shapec, ranka, winner_type);
        free(shapec); 
        
        if(typea == typeb && typea == winner_type) {
			// Same types
            size_t item_size = get_dtype_size(winner_type);
            size_t total_size_a = sizea * item_size;
            size_t total_size_b = sizeb * item_size;
			
			// Copy first tensor
            memcpy(result->data, dataa, total_size_a); 
			
			// Append second tensor
            memcpy((char*)result->data + total_size_a, datab, total_size_b);
        } else {
			// Type conversion needed 
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Convert and copy first tensor
                    for(size_t i = 0; i < sizea; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float64(elem, typea);
                    }
                    
                    // Convert and copy second tensor
                    for(size_t i = 0; i < sizeb; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[sizea + i] = nnl2_convert_to_float64(elem, typeb);
                    }
					
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Convert and copy first tensor
                    for(size_t i = 0; i < sizea; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float32(elem, typea);
                    }
                    
					// Convert and copy second tensor
                    for(size_t i = 0; i < sizeb; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[sizea + i] = nnl2_convert_to_float32(elem, typeb);
                    }
					
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
			    	// Convert and copy first tensor
                    for(size_t i = 0; i < sizea; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_int32(elem, typea);
                    }
                    
					// Convert and copy second tensor
                    for(size_t i = 0; i < sizeb; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[sizea + i] = nnl2_convert_to_int32(elem, typeb);
                    }
					
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    return NULL;
                }
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
 * @brief Backend implementations for vstack operation
 * @details
 * Array follows the common backend registration pattern for vertical stacking operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for vertical tensor concatenation
 * 
 * @see nnl2_naive
 * @see naive_vstack
 */
Implementation vstack_backends[] = {
	REGISTER_BACKEND(naive_vstack, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for vstack operation
 * @ingroup backend_system 
 */
vstackfn vstack;

/** 
 * @brief Makes the vstack backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(vstack);

/** 
 * @brief Sets the backend for vstack operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for vertical stacking
 * @see ESET_BACKEND_BY_NAME
 */
void set_vstack_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(vstack_backends, vstack, backend_name, current_backend(vstack));
}

/** 
 * @brief Gets the name of the active backend for vstack operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_vstack_backend() {
	return current_backend(vstack);
}

/** 
 * @brief Function declaration for getting all available vstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(vstack);

/**
 * @brief Function declaration for getting the number of available vstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(vstack);

/** @brief
 * Applies ReLU (ReLU(x) = max(x, 0)) activation function in-place to a tensor (naive implementation)
 *
 ** @param tensor
 * Pointer to the input tensor that will be modified in-place
 *
 ** @return
 * None (void function)
 *
 ** @see nnl2_relu_float64_inplace
 ** @see nnl2_relu_float32_inplace
 ** @see nnl2_relu_int32_inplace
 ** @see product
 **/
void naive_reluinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	int total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If tensor is empty return
	
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
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place ReLU operation
 * @details
 * Array follows the common backend registration pattern for in-place ReLU operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place ReLU activation
 * 
 * @see nnl2_naive
 * @see naive_reluinplace
 */
Implementation reluinplace_backends[] = {
	REGISTER_BACKEND(naive_reluinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for in-place ReLU operation
 * @ingroup backend_system 
 */
reluinplacefn reluinplace;

/** 
 * @brief Makes the reluinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(reluinplace);

/** 
 * @brief Sets the backend for in-place ReLU operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for in-place ReLU
 * @see ESET_BACKEND_BY_NAME
 */
void set_reluinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(reluinplace_backends, reluinplace, backend_name, current_backend(reluinplace));
}

/** 
 * @brief Gets the name of the active backend for in-place ReLU operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_reluinplace_backend() {
	return current_backend(reluinplace);
}

/** 
 * @brief Function declaration for getting all available in-place ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(reluinplace);

/**
 * @brief Function declaration for getting the number of available in-place ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(reluinplace);

/** @brief
 * Applies ReLU (ReLU(x) = max(x, 0)) activation function to a tensor, returning a new tensor (naive implementation)
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @return
 * Pointer to a new tensor containing the ReLU-activated values
 * Returns NULL in case of failure
 *
 ** @see nnl2_relu_float64
 ** @see nnl2_relu_float32
 ** @see nnl2_relu_int32
 ** @see nnl2_empty
 ** @see product
 **/
Tensor* naive_relu(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if(tensor == NULL) {
			NNL2_ERROR("Passed tensor is NULL");
		}
	#endif

	int total_elems = product(tensor->shape, tensor->rank);	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	
	if(total_elems == 0) return result; // If tensor is empty return tensor with 0 elements
	
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
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for ReLU operation
 * @details
 * Array follows the common backend registration pattern for ReLU operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for ReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_relu
 */
Implementation relu_backends[] = {
	REGISTER_BACKEND(naive_relu, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for ReLU operation
 * @ingroup backend_system 
 */
relufn relu;

/** 
 * @brief Makes the relu backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(relu);

/** 
 * @brief Sets the backend for ReLU operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for ReLU
 * @see ESET_BACKEND_BY_NAME
 */
void set_relu_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(relu_backends, relu, backend_name, current_backend(relu));
}

/** 
 * @brief Gets the name of the active backend for ReLU operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_relu_backend() {
	return current_backend(relu);
}

/** 
 * @brief Function declaration for getting all available ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(relu);

/**
 * @brief Function declaration for getting the number of available ReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(relu);

/** @brief
 * Applies Leaky ReLU (max(alpha * x, x)) function to an in-place tensor
 *
 * Example 1: leakyrelu(1) -> 1
 * Example 2: leakyrelu(-1, alpha=0.01) -> -0.01
 *	
 ** @param tensor
 * A pointer to a tensor for modification]
 *
 ** @param alpha
 * Slope coefficient for negative values (usually a small positive number)
 */
void naive_leakyreluinplace(Tensor* tensor, float alpha) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	int total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If tensor is empty return
	
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
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for LeakyReLU in-place operation
 * @details
 * Array follows the common backend registration pattern for LeakyReLU in-place operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for LeakyReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_leakyreluinplace
 */
Implementation leakyreluinplace_backends[] = {
	REGISTER_BACKEND(naive_leakyreluinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for LeakyReLU in-place operation
 * @ingroup backend_system 
 */
leakyreluinplacefn leakyreluinplace;

/** 
 * @brief Makes the leakyreluinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(leakyreluinplace);

/** 
 * @brief Sets the backend for LeakyReLU in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for LeakyReLU in-place
 * @see ESET_BACKEND_BY_NAME
 */
void set_leakyreluinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(leakyreluinplace_backends, leakyreluinplace, backend_name, current_backend(leakyreluinplace));
}

/** 
 * @brief Gets the name of the active backend for LeakyReLU in-place operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_leakyreluinplace_backend() {
	return current_backend(leakyreluinplace);
}

/** 
 * @brief Function declaration for getting all available LeakyReLU in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(leakyreluinplace);

/**
 * @brief Function declaration for getting the number of available LeakyReLU in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(leakyreluinplace);

/** @brief
 * A function that does what you already know with a tensor, but documentation is required
 *
 ** @details
 * Applies Leaky ReLU (max(x * alpha, x)) function element-wise
 *
 * Example 1: leakyrelu(1) -> 1
 * Example 2: leakyrelu(-1, alpha=0.01) -> -0.01
 *
 ** @param tensor
 * Input tensor to apply Leaky ReLU to
 *
 ** @param alpha
 * Negative slope coefficient for values less than zero
 *
 ** @param save_type
 * Tries to preserve the initial tensor type if possible. It is recommended to set true
 *
 ** @return
 * New tensor with Leaky ReLU applied
 */
Tensor* naive_leakyrelu(Tensor* tensor, float alpha, bool save_type) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate total number of elements
	int total_elems = product(tensor->shape, tensor->rank);	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype); 
	
	if(total_elems == 0) return result; // If tensor is empty then return new empty tensor
	
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
			
			// Check if conversion to float64 is needed for int32
			for(int i = 0; i < total_elems; i++) {
				if(cast_data_t[i] < 0) {
					float result_val = cast_data_t[i] * alpha;
					
					// Check if result has fractional part
					if(fmodf(result_val, 1.0f) != 0.0f) {
						float64_conversion = true;
						break;
					}
				}
			}
			
			// Handle int32 with potential type conversion
			if(float64_conversion || !save_type) {
				nnl2_free_tensor(result); 
				
				result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64); // this is, by the way, ineffective, but I'm too lazy to redo it
				
				// I don't care if you need performance, I'll only add it if I need it myself
				
				// Besides, in that case, I'll probably abandon this implementation 
				// and do the parallelization right away
				
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
				for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_int32(cast_data_t[i], alpha); // oh my god, there are extra int checks in every call, it's so unnecessary
			}
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for LeakyReLU operation
 * @details
 * Array follows the common backend registration pattern for LeakyReLU operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for LeakyReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_leakyrelu
 */
Implementation leakyrelu_backends[] = {
	REGISTER_BACKEND(naive_leakyrelu, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for LeakyReLU operation
 * @ingroup backend_system 
 */
leakyrelufn leakyrelu;

/** 
 * @brief Makes the leakyrelu backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(leakyrelu);

/** 
 * @brief Sets the backend for LeakyReLU operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for LeakyReLU
 * @see ESET_BACKEND_BY_NAME
 */
void set_leakyrelu_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(leakyrelu_backends, leakyrelu, backend_name, current_backend(leakyrelu));
}

/** 
 * @brief Gets the name of the active backend for LeakyReLU operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_leakyrelu_backend() {
	return current_backend(leakyrelu);
}

/** 
 * @brief Function declaration for getting all available LeakyReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(leakyrelu);

/**
 * @brief Function declaration for getting the number of available LeakyReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(leakyrelu);

/** @brief
 * Applies a sigmoid function to a tensor in place
 *
 ** @details
 * Sigmoid function (I write from memory): sigmoid(x) = 1/exp(-x)
 *
 * Example 1: sigmoid(1) = 1/exp(-1) = 0.731
 * Example 2: sigmoid(0.5) = 1/exp(-0.5) = 0.622
 *
 * oh, look, I think I've miscalculated the formula. Okay. Here's the correct formula:
 * sigmoid(x) = 1 / (1 + exp(-x))
 *
 ** @param tensor
 * A pointer to a tensor for conversion
 *
 * This @param is very useful, and without its documentation, 
 * you probably wouldn't know what it does or why it's needed
 *
 ** @see nnl2_product
 **/
void naive_sigmoidinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate total number of elements
	int total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If 0 elems then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_sigmoid_float64_inplace(&cast_data[i]); // Maximum efficiency
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(int i = 0; i < total_elems; i++) nnl2_sigmoid_float32_inplace(&cast_data[i]); // Maximum efficiency
			break;
		}
		
		case INT32: {
			NNL2_FATAL("Sigmoid in-place cannot be applied to the provided tensor");
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for sigmoidinplace operation
 * @details
 * Array follows the common backend registration pattern for sigmoidinplace operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place sigmoid activation function
 * 
 * @see nnl2_naive
 * @see naive_sigmoidinplace
 */
Implementation sigmoidinplace_backends[] = {
	REGISTER_BACKEND(naive_sigmoidinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for sigmoidinplace operation
 * @ingroup backend_system 
 */
sigmoidinplacefn sigmoidinplace;

/** 
 * @brief Makes the sigmoidinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(sigmoidinplace);

/** 
 * @brief Sets the backend for sigmoidinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for sigmoidinplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_sigmoidinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoidinplace_backends, sigmoidinplace, backend_name, current_backend(sigmoidinplace));
}

/** 
 * @brief Gets the name of the active backend for sigmoidinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sigmoidinplace_backend() {
	return current_backend(sigmoidinplace);
}

/** 
 * @brief Function declaration for getting all available sigmoidinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sigmoidinplace);

/**
 * @brief Function declaration for getting the number of available sigmoidinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoidinplace);

/** @brief
 * Calculates the sigmoid function for each tensor element
 *
 ** @details
 * Sigmoid function: sigmoid(x) = sigmoid(x) = 1 / (1 + exp(-x))
 *
 * Example 1: sigmoid(1) = 1 / (1 + exp(-1)) = 0.731
 * Example 2: sigmoid(0.5) = 1 / (1 + exp(-0.5)) = 0.622
 *
 ** @note
 * int32 Is automatically converted to float64
 *
 ** @return
 * Pointer to a new tensor with the result of applying a sigmoid function
 *
 ** @see nnl2_sigmoid_float64
 ** @see nnl2_sigmoid_float32
 ** @see nnl2_empty
 **/
Tensor* naive_sigmoid(Tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);	
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
	
	// Ultra mega super optimization
	if(total_elems == 0) return result; // Increases speed by 100-150% (If tensor is empty then return empty result)
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;
			double* cast_data_r = (double*)data_r;
			
			// For int32 return float64 tensor
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float64((double)cast_data_t[i]); 
			break;
		}
		
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float64(cast_data_t[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(size_t i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_sigmoid_float32(cast_data_t[i]);
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for sigmoid operation
 * @details
 * Array follows the common backend registration pattern for sigmoid operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for sigmoid activation function
 * 
 * @see nnl2_naive
 * @see naive_sigmoid
 */
Implementation sigmoid_backends[] = {
	REGISTER_BACKEND(naive_sigmoid, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for sigmoid operation
 * @ingroup backend_system 
 */
sigmoidfn sigmoid;

/** 
 * @brief Makes the sigmoid backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(sigmoid);

/** 
 * @brief Sets the backend for sigmoid operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for sigmoid
 * @see ESET_BACKEND_BY_NAME
 */
void set_sigmoid_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoid_backends, sigmoid, backend_name, current_backend(sigmoid));
}

/** 
 * @brief Gets the name of the active backend for sigmoid operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sigmoid_backend() {
	return current_backend(sigmoid);
}

/** 
 * @brief Function declaration for getting all available sigmoid backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sigmoid);

/**
 * @brief Function declaration for getting the number of available sigmoid backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoid);

/** @brief
 * Calculates the hyperbolic tangent for all tensor elements in place
 *
 ** @details
 * Look up the hyperbolic tangent formula on the internet
 * I'm too lazy to write it manually
 *
 ** @param tensor
 * A pointer to a tensor for processing
 *
 ** @note
 * Does not work with integer data types
 * 
 ** @see nnl2_product
 ** @see tanh
 ** @see tanhf
 **/
void naive_tanhinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	int total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If tensor is empty then return
	
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
			NNL2_FATAL("Tanh (in-place) cannot be applied to the provided tensor");
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for tanh in-place operation
 * @details
 * Array follows the common backend registration pattern for hyperbolic tangent 
 * in-place operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tanh activation function
 * 
 * @see nnl2_naive
 * @see naive_tanhinplace
 */
Implementation tanhinplace_backends[] = {
	REGISTER_BACKEND(naive_tanhinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for tanh in-place operation
 * @ingroup backend_system 
 */
tanhinplacefn tanhinplace;

/** 
 * @brief Makes the tanh in-place backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(tanhinplace);

/** 
 * @brief Sets the backend for tanh in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for tanh in-place
 * @see ESET_BACKEND_BY_NAME
 */
void set_tanhinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(tanhinplace_backends, tanhinplace, backend_name, current_backend(tanhinplace));
}

/** 
 * @brief Gets the name of the active backend for tanh in-place operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_tanhinplace_backend() {
	return current_backend(tanhinplace);
}

/** 
 * @brief Function declaration for getting all available tanh in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(tanhinplace);

/**
 * @brief Function declaration for getting the number of available tanh in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(tanhinplace);

/** @brief
 * Calculates the hyperbolic tangent (tanh) for each tensor element
 *
 ** @details
 * You can see all the details in nnl2_naive_tanhinplace including the full formula
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @note
 * For integer types, always returns float64
 *
 ** @return
 * Pointer to the resulting tensor
 *
 ** @see nnl2_naive_tanhinplace
 ** @see tanh
 ** @see tanhf
 **/
Tensor* naive_tanh(Tensor* tensor) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

	int total_elems = product(tensor->shape, tensor->rank);
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
	if(total_elems == 0) return result; // If tensor is empty, return empty result
	
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
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for tanh operation
 * @details
 * Array follows the common backend registration pattern for hyperbolic tangent 
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tanh activation function
 * 
 * @see nnl2_naive
 * @see naive_tanh
 */
Implementation tanh_backends[] = {
	REGISTER_BACKEND(naive_tanh, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for tanh operation
 * @ingroup backend_system 
 */
tanhfn nnl2_tanh;

/** 
 * @brief Makes the tanh backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(tanh);

/** 
 * @brief Sets the backend for tanh operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for tanh
 * @see ESET_BACKEND_BY_NAME
 */
void set_tanh_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(tanh_backends, nnl2_tanh, backend_name, current_backend(tanh));
}

/** 
 * @brief Gets the name of the active backend for tanh operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_tanh_backend() {
	return current_backend(tanh);
}

/** 
 * @brief Function declaration for getting all available tanh backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(tanh);

/**
 * @brief Function declaration for getting the number of available tanh backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(tanh);

/** @brief
 * Performs concatenation of two tensors along specified axis (naive implementation)
 * Supports different data types with automatic type promotion
 *
 ** @details 
 * This function performs concatenation of two tensors along the given axis
 * The function supports automatic type promotion - if input tensors have
 * different data types, the result will use the "winner" type (higher precision)
 *
 ** @param tensora 
 * Pointer to the first input tensor
 *
 ** @param tensorb
 * Pointer to the second input tensor
 *
 ** @param axis
 * Axis along which to concatenate (0-based)
 *
 ** @return 
 * Pointer to a new tensor containing the concatenated result
 *
 ** @example
 * Tensor* a = ones(shape_a, rank, FLOAT32);
 * Tensor* b = ones(shape_b, rank, FLOAT32);
 * Tensor* result = naive_concat(a, b, 1); // Concatenate along axis 1
 * if (result != NULL) {
 *     // Use concatenated result
 *     nnl2_free_tensor(result);
 * }
 *
 * @retval NULL If ranks of input tensors does not match
 * @retval NULL If axis is out of valid range
 * @retval NULL If non-concatenation dimensions have incompatible shapes
 * @retval NULL If memory allocation fails
 * @retval NULL If unsupported data type encountered
 *
 ** @warning
 * DRY final boss 
 *
 ** @see get_dtype_size()
 ** @see nnl2_empty
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 ** @see nnl2_free_tensor()
 **/
Tensor* naive_concat(Tensor* tensora, Tensor* tensorb, int axis) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    int ranka = tensora->rank;
    int rankb = tensorb->rank;
    
    TensorType winner_type = MAX(typea, typeb);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (ranka != rankb) {
            NNL2_ERROR("Ranks are different (%d != %d) in concat", ranka, rankb);
            return NULL;
        }

        if (axis < 0 || axis >= ranka) {
            NNL2_ERROR("Invalid axis %d (must be 0-%d) in concat", axis, ranka - 1);
            return NULL;
        }

        // Check compatible shapes using strides for efficiency
        for (int i = 0; i < ranka; i++) {
            if (i != axis && tensora->shape[i] != tensorb->shape[i]) {
                NNL2_ERROR("Incompatible shapes along axis %d (%d != %d) in concat", i, tensora->shape[i], tensorb->shape[i]);
                return NULL;
            }
        }
    #endif
    
    // Calculate result shape
    int* shapec = malloc(ranka * sizeof(int));
    if (shapec == NULL) {
        NNL2_ERROR("Memory allocation failed for shape in concat");
        return NULL;
    }
    
    for (int i = 0; i < ranka; i++) {
        shapec[i] = (i == axis) ? (tensora->shape[i] + tensorb->shape[i]) : tensora->shape[i];
    }    
    
    Tensor* result = nnl2_empty(shapec, ranka, winner_type);
    free(shapec);
    
    if (result == NULL) {
        return NULL;
    }
    
    size_t item_size = get_dtype_size(winner_type);
    
    // Use precomputed strides for efficient memory access
    size_t outer_elements = 1;
    for (int i = 0; i < axis; i++) {
        outer_elements *= tensora->shape[i];
    }
    
    size_t inner_elements = 1;
    for (int i = axis + 1; i < ranka; i++) {
        inner_elements *= tensora->shape[i];
    }
    
    size_t a_axis_elements = tensora->shape[axis];
    size_t b_axis_elements = tensorb->shape[axis];
    
    char* a_data = (char*)tensora->data;
    char* b_data = (char*)tensorb->data;
    char* c_data = (char*)result->data;
    
    if (typea == typeb && typea == winner_type) {
        // Handling case with same types, use memcpy with strides
        size_t a_slice_size = a_axis_elements * inner_elements * item_size;
        size_t b_slice_size = b_axis_elements * inner_elements * item_size;

        for (size_t outer = 0; outer < outer_elements; outer++) {
            size_t a_offset = outer * tensora->strides[axis] * a_axis_elements;
            size_t b_offset = outer * tensorb->strides[axis] * b_axis_elements;
            size_t c_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements);
            
            memcpy(c_data + c_offset, a_data + a_offset, a_slice_size);
            memcpy(c_data + c_offset + a_slice_size, b_data + b_offset, b_slice_size);
        }
    } else {
		// Now we handle the case when the variable names 
		// indicate that the code was written with AI support
		
        // Type conversion needed - process using strides
        switch(winner_type) { 
            case FLOAT64: {
				// Cast result data pointer 
                double* dst = (double*)result->data;
                
                for (size_t outer = 0; outer < outer_elements; outer++) {
					// Copy elements from first tensor (tensora)
                    for (size_t a_pos = 0; a_pos < a_axis_elements; a_pos++) {
						// Process all elements after the concatenation axis
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            // Calculate source offset in tensor A
							
							// outer * tensora->strides[axis] * a_axis_elements --- offset for outer dimensions
							// a_pos * tensora->strides[axis] --- offset along concatenation axis
							// inner * (axis < ranka - 1 ? tensora->strides[axis + 1] : 1) --- offset for inner dimensions
					
                            size_t a_src_offset = outer * tensora->strides[axis] * a_axis_elements + a_pos * tensora->strides[axis] + inner * (axis < ranka - 1 ? tensora->strides[axis + 1] : 1);
							
							// Calculate destination offset in result tensor
							// Similar calculation but with total axis size (a_axis_elements + b_axis_elements)
							
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + a_pos * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
							// Get pointer to source element in tensor A
                            void* elem = a_data + a_src_offset * get_dtype_size(typea);
							
							// Convert element to double and store in result
                            dst[dst_offset] = nnl2_convert_to_float64(elem, typea);
                        }
                    }
                    
					// Copy elements from second tensor (tensorb)
                    for (size_t b_pos = 0; b_pos < b_axis_elements; b_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            // Calculate source offset in tensor B
                            size_t b_src_offset = outer * tensorb->strides[axis] * b_axis_elements + b_pos * tensorb->strides[axis] + inner * (axis < rankb - 1 ? tensorb->strides[axis + 1] : 1);
							
							// Calculate destination offset in result tensor
							// Position along concatenation axis is offset by a_axis_elements
							
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + (a_axis_elements + b_pos) * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
							// Get pointer to source element in tensor B
                            void* elem = b_data + b_src_offset * get_dtype_size(typeb);
							
							// Convert element to double and store in result
                            dst[dst_offset] = nnl2_convert_to_float64(elem, typeb);
                        }
                    }
                }
				
                break;
            }
            
            case FLOAT32: {
				// Cast result data pointer
                float* dst = (float*)result->data;
                
			    // Same nested loop structure as FLOAT64 case
                for (size_t outer = 0; outer < outer_elements; outer++) {
                    for (size_t a_pos = 0; a_pos < a_axis_elements; a_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            size_t a_src_offset = outer * tensora->strides[axis] * a_axis_elements + a_pos * tensora->strides[axis] + inner * (axis < ranka - 1 ? tensora->strides[axis + 1] : 1);
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + a_pos * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
                            void* elem = a_data + a_src_offset * get_dtype_size(typea);
							
							// Convert element to float32 instead of float64
                            dst[dst_offset] = nnl2_convert_to_float32(elem, typea);
                        }
                    }
                    
                    for (size_t b_pos = 0; b_pos < b_axis_elements; b_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            size_t b_src_offset = outer * tensorb->strides[axis] * b_axis_elements + b_pos * tensorb->strides[axis] + inner * (axis < rankb - 1 ? tensorb->strides[axis + 1] : 1);
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + (a_axis_elements + b_pos) * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
                            void* elem = b_data + b_src_offset * get_dtype_size(typeb);
							
							// Convert element to float32
                            dst[dst_offset] = nnl2_convert_to_float32(elem, typeb);
                        }
                    }
                }
				
                break;
            }
            
            case INT32: {
				// Cast result data pointer
                int32_t* dst = (int32_t*)result->data;
                
			    // Same nested loop structure for integer concatenation
                for (size_t outer = 0; outer < outer_elements; outer++) {
                    for (size_t a_pos = 0; a_pos < a_axis_elements; a_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            size_t a_src_offset = outer * tensora->strides[axis] * a_axis_elements + a_pos * tensora->strides[axis] + inner * (axis < ranka - 1 ? tensora->strides[axis + 1] : 1);                            
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + a_pos * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
                            void* elem = a_data + a_src_offset * get_dtype_size(typea);
							
							// Convert element to int32
                            dst[dst_offset] = nnl2_convert_to_int32(elem, typea);
                        }
                    }
                    
                    for (size_t b_pos = 0; b_pos < b_axis_elements; b_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            size_t b_src_offset = outer * tensorb->strides[axis] * b_axis_elements + b_pos * tensorb->strides[axis] + inner * (axis < rankb - 1 ? tensorb->strides[axis + 1] : 1);
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + (a_axis_elements + b_pos) * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
                            void* elem = b_data + b_src_offset * get_dtype_size(typeb);
							
							// Convert element to int32
                            dst[dst_offset] = nnl2_convert_to_int32(elem, typeb);
                        }
                    }
                }
				
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_type);
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
 * @brief Backend implementations for concat operation
 * @details
 * Array follows the common backend registration pattern for concatenation 
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor concatenation
 * 
 * @see nnl2_naive
 * @see naive_concat
 */
Implementation concat_backends[] = {
	REGISTER_BACKEND(naive_concat, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for concat operation
 * @ingroup backend_system 
 */
concatfn nnl2_concat;

/** 
 * @brief Makes the concat backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(concat);

/** 
 * @brief Sets the backend for concat operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for concat
 * @see ESET_BACKEND_BY_NAME
 */
void set_concat_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(concat_backends, nnl2_concat, backend_name, current_backend(concat));
}

/** 
 * @brief Gets the name of the active backend for concat operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_concat_backend() {
	return current_backend(concat);
}

/** 
 * @brief Function declaration for getting all available concat backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(concat);

/**
 * @brief Function declaration for getting the number of available concat backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(concat);

/** @brief
 * Creates a tensor from random numbers in the specified range
 *
 ** @details
 * Nick Land - organicist technospecialization, pedagogical 
 * authoritarianism, and territorial sectorization end in 
 * numerical illiteracy and mass insignificance
 *
 ** @param shape
 * Array of integers defining the dimensions of the tensor
 *
 ** @param rank
 * Number of dimensions (length of shape array)
 *
 ** @param dtype
 * Data type of the tensor elements 
 *
 ** @param from
 * Pointer to the minimum value
 *
 ** @param to
 * Pointer to the maximum value
 *
 ** @return
 * Pointer to the newly created Tensor
 *
 ** @example
 * // Create a 3x3 tensor of random floats between 0.0 and 1.0
 * float min = 0.0f, max = 1.0f;
 * Tensor* quantum_blockchain_nft = nnl2_naive_randn((int[]){3, 3}, 2, FLOAT64, &min, &max);
 *
 ** @see nnl2_empty
 ** @see product
 **/
Tensor* naive_randn(int* shape, int rank, TensorType dtype, void* from, void* to) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	Tensor* result = nnl2_empty(shape, rank, dtype);
	
	size_t total_elems = product(shape, rank);
	if(total_elems == 0) return result; // If zero elems then result empty result
	
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
			NNL2_TYPE_ERROR(dtype);
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
 * @brief Backend implementations for randn operation
 * @details
 * Array follows the common backend registration pattern for random number 
 * generation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for random number generation
 * 
 * @see nnl2_naive
 * @see naive_randn
 */
Implementation randn_backends[] = {
	REGISTER_BACKEND(naive_randn, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for randn operation
 * @ingroup backend_system 
 */
randnfn randn;

/** 
 * @brief Makes the randn backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(randn);

/** 
 * @brief Sets the backend for randn operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for randn
 * @see ESET_BACKEND_BY_NAME
 */
void set_randn_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(randn_backends, randn, backend_name, current_backend(randn));
}

/** 
 * @brief Gets the name of the active backend for randn operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_randn_backend() {
	return current_backend(randn);
}

/** 
 * @brief Function declaration for getting all available randn backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(randn);

/**
 * @brief Function declaration for getting the number of available randn backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(randn);

/** @brief
 * Initializing a tensor using the Xavier distribution
 *
 * Standard deviation is calculated as: gain * sqrt(distribution / (in + out))
 *
 ** @param shape
 * Pointer to an array of tensor dimensions
 *
 ** @param rank
 * The rank of the tensor
 *
 ** @param dtype
 * Tensor data type
 *
 ** @param in
 * Number of input neurons
 *
 ** @param out
 * Number of output neurons
 *
 ** @param gain
 * Gain factor
 *
 ** @param distribution
 * Distribution parameter (usually 2.0 or 6.0)
 *
 ** @note
 * Integer data types are not supported
 *
 ** @see RAND_MAX
 ** @see nnl2_empty
 **/
Tensor* naive_xavier(int* shape, int rank, TensorType dtype, int in, int out, float gain, float distribution) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	if(dtype == INT32) {
		NNL2_FATAL("INT32 Can't be used for xavier distribution");
		return NULL;
	}
	
	Tensor* result = nnl2_empty(shape, rank, dtype);
	
	size_t total_elems = product(shape, rank);
	if(total_elems == 0) return result; // If tensor is empty return empty result
	
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
			NNL2_TYPE_ERROR(dtype);
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
 * @brief Backend implementations for xavier operation
 * @details
 * Array follows the common backend registration pattern for xavier initialization
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for xavier initialization
 * 
 * @see nnl2_naive
 * @see naive_xavier
 */
Implementation xavier_backends[] = {
	REGISTER_BACKEND(naive_xavier, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for xavier operation
 * @ingroup backend_system 
 */
xavierfn xavier;

/** 
 * @brief Makes the xavier backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(xavier);

/** 
 * @brief Sets the backend for xavier operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for xavier
 * @see ESET_BACKEND_BY_NAME
 */
void set_xavier_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(xavier_backends, xavier, backend_name, current_backend(xavier));
}

/** 
 * @brief Gets the name of the active backend for xavier operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_xavier_backend() {
	return current_backend(xavier);
}

/** 
 * @brief Function declaration for getting all available xavier backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(xavier);

/**
 * @brief Function declaration for getting the number of available xavier backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(xavier);

/** @brief
 * Transposes a matrix in place (naive implementation)
 *
 ** @warning
 * Not thread-safe
 *
 ** @param tensor
 * Pointer to a tensor for transposition
 *
 ** @see product
 ** @see https://en.wikipedia.org/wiki/Dont_repeat_yourself 
 **/
void naive_transposeinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		int32_t tensor_rank = tensor->rank;
		if(tensor_rank < 2 || tensor_rank > 2) {
			NNL2_FATAL("Tensor have an incorrect rank: %d (expected 2)", tensor_rank);
			return;
		}
	#endif
	
	// Calculating the total number of elements in a tensor
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	int* shape = tensor->shape;
	
	int rows = shape[0];
	int cols = shape[1];
	
	switch(tensor->dtype) {
		case FLOAT64: {
			size_t total_bytes = total_elems * sizeof(double);
			
			// Allocating memory for temporary storage of the transposed matrix
			double* trans_data = (double*)malloc(total_bytes);
			double* cast_data = (double*)tensor->data;
			
			if (trans_data == NULL) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
			
			// Matrix transposition: element [i][j] -> [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			// Copying the result back to the original tensor
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		case FLOAT32: {
			size_t total_bytes = total_elems * sizeof(float);
			
			// Allocating memory for temporary storage of the transposed matrix
			float* trans_data = (float*)malloc(total_bytes);
			float* cast_data = (float*)tensor->data;
			
			if (trans_data == NULL) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
			
			// Matrix transposition: element [i][j] -> [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) { 
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			// Copying the result back to the original tensor
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		case INT32: {
			size_t total_bytes = total_elems * sizeof(int32_t);
			
			// Allocating memory for temporary storage of the transposed matrix
			int32_t* trans_data = (int32_t*)malloc(total_bytes);
			int32_t* cast_data = (int32_t*)tensor->data;
			
			if (trans_data == NULL) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
			
			// Matrix transposition: element [i][j] -> [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) { 
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			// Copying the result back to the original tensor
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for transposeinplace operation
 * @details
 * Array follows the common backend registration pattern for in-place transpose
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place transpose
 * 
 * @see nnl2_naive
 * @see naive_transposeinplace
 */
Implementation transposeinplace_backends[] = {
	REGISTER_BACKEND(naive_transposeinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for transposeinplace operation
 * @ingroup backend_system 
 */
transposeinplacefn transposeinplace;

/** 
 * @brief Makes the transposeinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(transposeinplace);

/** 
 * @brief Sets the backend for transposeinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for transposeinplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_transposeinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transposeinplace_backends, transposeinplace, backend_name, current_backend(transposeinplace));
}

/** 
 * @brief Gets the name of the active backend for transposeinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_transposeinplace_backend() {
	return current_backend(transposeinplace);
}

/** 
 * @brief Function declaration for getting all available transposeinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(transposeinplace);

/**
 * @brief Function declaration for getting the number of available transposeinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transposeinplace);

/** @brief 
 * Performs 2D tensor transposition using a naive method
 *
 ** @param tensor 
 * Pointer to the original tensor
 *
 ** @return
 * Tensor* Pointer to a transposed tensor
 *
 ** @see https://en.wikipedia.org/wiki/Dont_repeat_yourself 
 **/
Tensor* naive_transpose(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Make sure the tensor is 2D
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		int32_t tensor_rank = tensor->rank;
		if(tensor_rank < 2 || tensor_rank > 2) {
			NNL2_FATAL("Tensor have an incorrect rank: %d (expected 2)", tensor_rank);
			return NULL;
		}
	#endif
	
	// Create a tensor result with the same parameters, but swap the shape
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	if (result == NULL) {
        NNL2_FATAL("Failed to initialize the tensor in transposition");
		return NULL;
    }

	int rows = tensor->shape[0];
	int cols = tensor->shape[1];
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* src_data = (double*)tensor->data;
			double* dest_data = (double*)result->data;

			// Transposition: element [i][j] becomes [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix

					dest_data[trans_index] = src_data[orig_index];
				}
			}

			break;
		}
		
		case FLOAT32: {
			float* src_data = (float*)tensor->data;
			float* dest_data = (float*)result->data;

			// Transposition: element [i][j] becomes [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix

					dest_data[trans_index] = src_data[orig_index];
				}
			}

			break;
		}
		
		case INT32: {
			int32_t* src_data = (int32_t*)tensor->data;
			int32_t* dest_data = (int32_t*)result->data;

			// Transposition: element [i][j] becomes [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix

					dest_data[trans_index] = src_data[orig_index];
				}
			}

			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for transpose operation
 * @details
 * Array follows the common backend registration pattern for transpose
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for transpose
 * 
 * @see nnl2_naive
 * @see naive_transpose
 */
Implementation transpose_backends[] = {
    REGISTER_BACKEND(naive_transpose, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for transpose operation
 * @ingroup backend_system 
 */
transposefn transpose;

/** 
 * @brief Makes the transpose backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(transpose);

/** 
 * @brief Sets the backend for transpose operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for transpose
 * @see ESET_BACKEND_BY_NAME
 */
void set_transpose_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transpose_backends, transpose, backend_name, current_backend(transpose));
}

/** 
 * @brief Gets the name of the active backend for transpose operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_transpose_backend() {
    return current_backend(transpose);
}

/** 
 * @brief Function declaration for getting all available transpose backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(transpose);

/**
 * @brief Function declaration for getting the number of available transpose backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transpose);

/** @brief
 * Computes the sum of all elements in a tensor without axis reduction
 * 
 ** @param tensor 
 * Pointer to the input tensor structure
 *
 ** @param result 
 * Pointer to the memory where the sum result will be stored
 */
void naive_sum_without_axis(Tensor* tensor, void* result) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; // If tensor if empty then return empty result
		
	switch(tensor->dtype) {
		case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double acc = 0.0;
            for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
            *((double*)result) = acc; // Store result in output pointer
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
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for sum without axis operation
 * @details
 * Array follows the common backend registration pattern for element-wise
 * summation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor summation
 * 
 * @see nnl2_naive
 * @see naive_sum_without_axis
 */
Implementation sum_without_axis_backends[] = {
    REGISTER_BACKEND(naive_sum_without_axis, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for sum without axis operation
 * @ingroup backend_system 
 */
sumwithoutaxisfn nnl2_sum_without_axis;

/** 
 * @brief Makes the sum without axis backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sum_without_axis);

/** 
 * @brief Sets the backend for sum without axis operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for summation
 * @see ESET_BACKEND_BY_NAME
 */
void set_sum_without_axis_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sum_without_axis_backends, nnl2_sum_without_axis, backend_name, CURRENT_BACKEND(sum_without_axis));
}

/** 
 * @brief Gets the name of the active backend for sum without axis operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sum_without_axis_backend() {
    return CURRENT_BACKEND(sum_without_axis);
}

/** 
 * @brief Function declaration for getting all available sum without axis backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sum_without_axis);

/**
 * @brief Function declaration for getting the number of available sum without axis backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sum_without_axis);

/** @brief
 * Calculates the source index in the tensor for the axis-wise summation operation
 *
 ** @param tensor
 * Original tensor
 *
 ** @param result
 * Result tensor
 *
 ** @param axis
 * The axis along which the summation is performed
 *
 ** @param result_index
 * Index in the resulting tensor
 * 
 ** @param axis_index
 * Position along the summation axis
 * 
 ** @return 
 * Calculated index in the original tensor
 */
NNL2_FORCE_INLINE static size_t nnl2_naive_calculate_original_index_for_sum_with_axis(Tensor* tensor, Tensor* result, int axis, size_t result_index, int axis_index);

/** @brief
 * Performs axis summation for a FLOAT64 tensor
 *
 ** @param tensor
 * Original tensor with double (float64) type
 *
 ** @param result
 * The preallocated resulting tensor
 *
 ** @param axis
 * Sum axis (0-based)
 *
 ** @param result_numel
 * The number of elements in the result
 *
 ** @param elements_along_axis
 * The number of elements along the summation axis
 *
 ** @return
 * Tensor* Pointer to the resulting tensor
 */
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float64(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

/** @brief
 * The documentation is identical to the 
 * nnl2_naive_sum_with_axis_float64 but with the float32 type
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

/** @brief
 * The documentation is identical to the 
 * nnl2_naive_sum_with_axis_float64 but with the int32 type
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_int32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

/** @brief
 * Computes the sum of tensor elements along the specified axis
 *
 ** @param tensor
 * Input tensor to perform summation on
 *
 ** @param axis
 * Axis along which to sum (0-based index)
 *
 ** @return
 * Tensor* New tensor with reduced rank containing the sums
 */
Tensor* naive_sum_with_axis(Tensor* tensor, int axis) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Validate axis parameter
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (axis < 0 || axis >= tensor->rank) {
			NNL2_ERROR("Invalid axis %d for tensor of rank %d", axis, tensor->rank);
			return NULL;
		}
	#endif
    
	// Allocate memory for result shape (rank reduced by 1)
    int* result_shape = malloc((tensor->rank - 1) * sizeof(int));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result_shape == NULL) {
			NNL2_ERROR("Failed to allocate memory");
			return NULL;
		}
	#endif
	
	// Construct result shape by excluding the summation axis
    int j = 0;
    for (int i = 0; i < tensor->rank; i++) {
        if (i != axis) {
            result_shape[j++] = tensor->shape[i];
        }
    }

    // Create empty result tensor with appropriate shape and dtype
    Tensor* result = nnl2_empty(result_shape, tensor->rank - 1, tensor->dtype);
    free(result_shape);
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result == NULL) {
			NNL2_ERROR("Failed to create result tensor");
			return NULL;
		}
	#endif
	
	// Select appropriate summation function based on data type
	Tensor* (*sum_func)(Tensor*, Tensor*, int, size_t, int) = NULL;
	
	switch(tensor->dtype) {
        case FLOAT32: {
            sum_func = nnl2_naive_sum_with_axis_float32;
            break;
		}
		
        case FLOAT64: {
            sum_func = nnl2_naive_sum_with_axis_float64;
            break;
		}
		
        case INT32: {
            sum_func = nnl2_naive_sum_with_axis_int32;
            break;
		}
		
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            nnl2_free_tensor(result);
            return NULL;
		}
    }
    
	// Calculate dimensions for the summation operation
    size_t result_numel = product(result->shape, result->rank);
    int elements_along_axis = tensor->shape[axis];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
	// Perform the actual summation
    return sum_func(tensor, result, axis, result_numel, elements_along_axis);
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_calculate_original_index_for_sum_with_axis
 **/
NNL2_FORCE_INLINE static size_t nnl2_naive_calculate_original_index_for_sum_with_axis(Tensor* tensor, Tensor* result, int axis, size_t result_index, int axis_index) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	size_t original_index = 0;
    int temp = result_index;
    
	// Reconstruct original coordinates from flattened result index
    for (int dim = result->rank - 1; dim >= 0; dim--) {
        int coord = temp % result->shape[dim];
        temp /= result->shape[dim];

		// Adjust dimension index to account for removed axis
        int original_dim = dim;
        if (original_dim >= axis) {
            original_dim++; 
        }
        
		// Add contribution from this dimension using original strides
        original_index += coord * tensor->strides[original_dim];
    }
    
	// Add contribution from the summation axis
    original_index += axis_index * tensor->strides[axis];
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
    return original_index;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float64(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    double* data = (double*)tensor->data;
    double* result_data = (double*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        double sum = 0.0;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_float32
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    float* data = (float*)tensor->data;
    float* result_data = (float*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        float sum = 0.0f;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_int32
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_int32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	int32_t* data = (int32_t*)tensor->data;
    int32_t* result_data = (int32_t*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        int32_t sum = 0;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for sum with axis operation
 * @details
 * Array follows the common backend registration pattern for axis-wise
 * summation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor summation along specified axes
 * 
 * @see nnl2_naive
 * @see naive_sum_with_axis
 */
Implementation sum_with_axis_backends[] = {
    REGISTER_BACKEND(naive_sum_with_axis, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for sum with axis operation
 * @ingroup backend_system 
 * 
 * This function pointer is used to call the currently active backend
 * implementation for summing tensor elements along specified dimensions.
 */
sumwithaxisfn nnl2_sum_with_axis;

/** 
 * @brief Makes the sum with axis backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sum_with_axis);

/** 
 * @brief Sets the backend for sum with axis operation
 * @ingroup backend_system
 * @param[in] backend_name Name of the backend to activate for axis-wise summation
 * @see ESET_BACKEND_BY_NAME
 * 
 * This function allows dynamic switching between different backend implementations
 * for the sum with axis operation at runtime.
 */
void set_sum_with_axis_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sum_with_axis_backends, nnl2_sum_with_axis, backend_name, CURRENT_BACKEND(sum_with_axis));
}

/** @brief
 * Computes the L2 norm (Euclidean norm) of a tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param axes
 * Array of axes for reduction
 *
 ** @param num_axes
 * Number of axes for reduction
 *
 ** @param result
 * Pointer to memory for storing the result
 *
 ** @see sqrt
 ** @see sqrtf
 **/
void naive_l2norm(Tensor* tensor, int* axes, int num_axes, void* result) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
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
                *((double*)result) = sqrt(acc); // Store result in output pointer
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
				NNL2_TYPE_ERROR(tensor->dtype);
				return;
			}
		}
	} else {
		NNL2_ERROR("Norm axes in development");
		
		// I don't plan to add support for axes 
		// in the near future because I don't need it
		
		return;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for L2 normalization operation
 * @details
 * Array follows the common backend registration pattern for L2 normalization
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for L2 normalization
 * 
 * @see nnl2_naive
 * @see naive_l2norm
 */
Implementation l2norm_backends[] = {
	REGISTER_BACKEND(naive_l2norm, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for L2 normalization operation
 * @ingroup backend_system 
 */
l2normfn l2norm;

/** 
 * @brief Makes the L2 normalization backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
MAKE_CURRENT_BACKEND(l2norm);

/** 
 * @brief Sets the backend for L2 normalization operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for L2 normalization
 * @see ESET_BACKEND_BY_NAME
 */
void set_l2norm_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(l2norm_backends, l2norm, backend_name, CURRENT_BACKEND(l2norm));
}

/** 
 * @brief Gets the name of the active backend for L2 normalization operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_l2norm_backend() {
	return CURRENT_BACKEND(l2norm);
}

/** 
 * @brief Function declaration for getting all available L2 normalization backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(l2norm);

/**
 * @brief Function declaration for getting the number of available L2 normalization backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(l2norm);

/** @brief 
 * Adds a scalar value to each element of a tensor (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor to which the value will be added
 * 
 ** @param inc 
 * Pointer to the scalar value to add
 */
void naive_add_incf_inplace(Tensor* tensor, void* inc) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; 
	
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
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}	

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place scalar addition operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar addition
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar addition
 * 
 * @see nnl2_naive
 * @see naive_add_incf_inplace
 */
Implementation add_incf_inplace_backends[] = {
	REGISTER_BACKEND(naive_add_incf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for in-place scalar addition operation
 * @ingroup backend_system 
 */
addincfinplacefn add_incf_inplace;

/** 
 * @brief Sets the backend for in-place scalar addition operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar addition
 * @see ESET_BACKEND_BY_NAME
 */
void set_add_incf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_incf_inplace_backends, add_incf_inplace, backend_name);
}

/** @brief
 * Performs element-wise addition of a scalar increment to a tensor
 *
 ** @param tensor
 * Pointer to the input tensor to which the increment will be added
 *
 ** @param inc
 * Pointer to the scalar increment value
 *
 ** @return
 * Pointer to a new tensor containing the result of the addition operation 
 * (or NULL in case of fail)
 */
Tensor* naive_add_incf(Tensor* tensor, void* inc) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result == NULL) {
			NNL2_ERROR("Failed to allocate new tensor");
		}
	#endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return result;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data; 
			double* cast_data_result = (double*)result->data;    // Casting
			double increment = *((double*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment; // Assigment
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
			NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for scalar addition with increment operation
 * @details
 * Array follows the common backend registration pattern for scalar addition with
 * increment operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar addition with increment
 * 
 * @see nnl2_naive
 * @see naive_add_incf
 */
Implementation add_incf_backends[] = {
    REGISTER_BACKEND(naive_add_incf, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for scalar addition with increment operation
 * @ingroup backend_system
 */
addincffn add_incf;

/** 
 * @brief Sets the backend for scalar addition with increment operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar addition with increment
 * @see SET_BACKEND_BY_NAME
 * @see add_incf_backends
 */
void set_add_incf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_incf_backends, add_incf, backend_name);
}

/** @brief 
 * Subtracts a scalar value from each element of a tensor (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor from which the value will be subtracted
 * 
 ** @param dec 
 * Pointer to the scalar value to subtract
 */
void naive_sub_decf_inplace(Tensor* tensor, void* dec) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;  // Casting
            double decrement = *((double*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float decrement = *((float*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t decrement = *((int32_t*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] -= decrement;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place scalar subtraction operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar subtraction
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar subtraction
 * 
 * @see nnl2_naive
 * @see naive_sub_decf_inplace
 */
Implementation sub_decf_inplace_backends[] = {
    REGISTER_BACKEND(naive_sub_decf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place scalar subtraction operation
 * @ingroup backend_system 
 */
subdecfinplacefn sub_decf_inplace;

/** 
 * @brief Sets the backend for in-place scalar subtraction operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar subtraction
 * @see SET_BACKEND_BY_NAME
 */
void set_sub_decf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_decf_inplace_backends, sub_decf_inplace, backend_name);
}

/** @brief
 * Performs element-wise subtraction of a scalar decrement from a tensor
 *
 ** @param tensor
 * Pointer to the input tensor from which the decrement will be subtracted
 *
 ** @param dec
 * Pointer to the scalar decrement value
 *
 ** @return
 * Pointer to a new tensor containing the result of the subtraction operation 
 * (or NULL in case of failure)
 */
Tensor* naive_sub_decf(const Tensor* tensor, void* dec) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;	// Casting
            double decrement = *((double*)dec); 
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement; // Assigment
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float decrement = *((float*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement;
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t decrement = *((int32_t*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for scalar subtraction with decrement operation
 * @details
 * Array follows the common backend registration pattern for scalar subtraction with
 * decrement operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar subtraction with decrement
 * 
 * @see nnl2_naive
 * @see naive_sub_decf
 */
Implementation sub_decf_backends[] = {
    REGISTER_BACKEND(naive_sub_decf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for scalar subtraction with decrement operation
 * @ingroup backend_system
 */
subdecffn sub_decf;

/** 
 * @brief Sets the backend for scalar subtraction with decrement operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar subtraction with decrement
 * @see SET_BACKEND_BY_NAME
 * @see sub_decf_backends
 */
void set_sub_decf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_decf_backends, sub_decf, backend_name);
}

/** @brief 
 * Multiplies each element of a tensor by a scalar value (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be multiplied
 * 
 ** @param multiplier 
 * Pointer to the scalar value to multiply by
 */
void naive_mul_mulf_inplace(Tensor* tensor, void* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double multiply = *((double*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float multiply = *((float*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t multiply = *((int32_t*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] *= multiply;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place scalar multiplication operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar multiplication
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar multiplication
 * 
 * @see nnl2_naive
 * @see naive_mul_mulf_inplace
 */
Implementation mul_mulf_inplace_backends[] = {
    REGISTER_BACKEND(naive_mul_mulf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place scalar multiplication operation
 * @ingroup backend_system 
 */
mulmulfinplacefn mul_mulf_inplace;

/** 
 * @brief Sets the backend for in-place scalar multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar multiplication
 * @see SET_BACKEND_BY_NAME
 */
void set_mul_mulf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_mulf_inplace_backends, mul_mulf_inplace, backend_name);
}

/** @brief
 * Performs element-wise multiplication of a tensor by a scalar multiplier
 *
 ** @param tensor
 * Pointer to the input tensor to be multiplied
 *
 ** @param multiplier
 * Pointer to the scalar multiplier value
 *
 ** @return
 * Pointer to a new tensor containing the result of the multiplication operation 
 * (or NULL in case of failure)
 */
Tensor* naive_mul_mulf(const Tensor* tensor, void* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;
            double multiply = *((double*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float multiply = *((float*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t multiply = *((int32_t*)multiplier);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] * multiply;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for scalar multiplication operation
 * @details
 * Array follows the common backend registration pattern for scalar multiplication
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar multiplication
 * 
 * @see nnl2_naive
 * @see naive_mul_mulf
 */
Implementation mul_mulf_backends[] = {
    REGISTER_BACKEND(naive_mul_mulf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for scalar multiplication operation
 * @ingroup backend_system
 */
mulmulffn mul_mulf;

/** 
 * @brief Sets the backend for scalar multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar multiplication
 * @see SET_BACKEND_BY_NAME
 * @see mul_mulf_backends
 */
void set_mul_mulf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(mul_mulf_backends, mul_mulf, backend_name);
}

/** @brief 
 * Divides each element of a tensor by a scalar value (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be divided
 * 
 ** @param divisor 
 * Pointer to the scalar value to divide by
 */
void naive_div_divf_inplace(Tensor* tensor, void* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double div = *((double*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float div = *((float*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t div = *((int32_t*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place scalar division operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar division
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar division
 * 
 * @see nnl2_naive
 * @see naive_div_divf_inplace
 */
Implementation div_divf_inplace_backends[] = {
    REGISTER_BACKEND(naive_div_divf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place scalar division operation
 * @ingroup backend_system 
 */
divdivfinplacefn div_divf_inplace;

/** 
 * @brief Sets the backend for in-place scalar division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar division
 * @see SET_BACKEND_BY_NAME
 */
void set_div_divf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_divf_inplace_backends, div_divf_inplace, backend_name);
}

/** @brief
 * Performs element-wise division of a tensor by a scalar divisor
 *
 ** @param tensor
 * Pointer to the input tensor to be divided
 *
 ** @param divisor
 * Pointer to the scalar divisor value
 *
 ** @return
 * Pointer to a new tensor containing the result of the division operation 
 * (or NULL in case of failure)
 */
Tensor* naive_div_divf(const Tensor* tensor, void* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;
            double div = *((double*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / div;
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float div = *((float*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / div;
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t div = *((int32_t*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] / div;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for scalar division operation
 * @details
 * Array follows the common backend registration pattern for scalar division
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar division
 * 
 * @see nnl2_naive
 * @see naive_div_divf
 */
Implementation div_divf_backends[] = {
    REGISTER_BACKEND(naive_div_divf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for scalar division operation
 * @ingroup backend_system
 */
divdivffn div_divf;

/** 
 * @brief Sets the backend for scalar division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar division
 * @see SET_BACKEND_BY_NAME
 * @see div_divf_backends
 */
void set_div_divf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_divf_backends, div_divf, backend_name);
}

/** @brief 
 * Raises each element of a tensor to a scalar power (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be exponentiated
 * 
 ** @param exponent 
 * Pointer to the scalar exponent value
 */
void naive_pow_powf_inplace(Tensor* tensor, void* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double pow_val = *((double*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = pow(cast_data[i], pow_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float pow_val = *((float*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = powf(cast_data[i], pow_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t pow_val = *((int32_t*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = (int32_t)pow((double)cast_data[i], pow_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place exponentiation operation
 * @details
 * Array follows the common backend registration pattern for in-place exponentiation
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for exponentiation
 * 
 * @see nnl2_naive
 * @see naive_pow_powf_inplace
 */
Implementation pow_powf_inplace_backends[] = {
    REGISTER_BACKEND(naive_pow_powf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place exponentiation operation
 * @ingroup backend_system 
 */
powpowfinplacefn pow_powf_inplace;

/** 
 * @brief Sets the backend for in-place exponentiation operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for exponentiation
 * @see SET_BACKEND_BY_NAME
 */
void set_pow_powf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_powf_inplace_backends, pow_powf_inplace, backend_name);
}

/** @brief
 * Performs element-wise exponentiation of a tensor by a scalar exponent
 *
 ** @param tensor
 * Pointer to the input tensor to be exponentiated
 *
 ** @param exponent
 * Pointer to the scalar exponent value
 *
 ** @return
 * Pointer to a new tensor containing the result of the exponentiation operation 
 * (or NULL in case of failure)
 */
Tensor* naive_pow_powf(const Tensor* tensor, void* exponent) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;
            double pow_val = *((double*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = pow(cast_data_original[i], pow_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float pow_val = *((float*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = powf(cast_data_original[i], pow_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t pow_val = *((int32_t*)exponent);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = (int32_t)pow((double)cast_data_original[i], pow_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for exponentiation operation
 * @details
 * Array follows the common backend registration pattern for exponentiation
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for exponentiation
 * 
 * @see nnl2_naive
 * @see naive_pow_powf
 */
Implementation pow_powf_backends[] = {
    REGISTER_BACKEND(naive_pow_powf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for exponentiation operation
 * @ingroup backend_system
 */
powpowffn pow_powf;

/** 
 * @brief Sets the backend for exponentiation operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for exponentiation
 * @see SET_BACKEND_BY_NAME
 * @see pow_powf_backends
 */
void set_pow_powf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(pow_powf_backends, pow_powf, backend_name);
}

/** @brief 
 * Applies element-wise maximum between tensor elements and a scalar value (in-place).
 * Each element is replaced by the maximum of its current value and the specified scalar.
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be processed
 * 
 ** @param threshold 
 * Pointer to the scalar threshold value for maximum operation
 */
void naive_max_maxf_inplace(Tensor* tensor, void* threshold) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double max_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], max_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float max_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], max_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t max_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MAX(cast_data[i], max_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place element-wise maximum operation
 * @details
 * Array follows the common backend registration pattern for in-place maximum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise maximum
 * 
 * @see nnl2_naive
 * @see naive_max_maxf_inplace
 */
Implementation max_maxf_inplace_backends[] = {
    REGISTER_BACKEND(naive_max_maxf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place element-wise maximum operation
 * @ingroup backend_system 
 */
maxmaxfinplacefn max_maxf_inplace;

/** 
 * @brief Sets the backend for in-place element-wise maximum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for maximum operation
 * @see SET_BACKEND_BY_NAME
 */
void set_max_maxf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_maxf_inplace_backends, max_maxf_inplace, backend_name);
}

/** @brief
 * Performs element-wise maximum operation between tensor elements and a scalar value
 *
 ** @param tensor
 * Pointer to the input tensor to be processed
 *
 ** @param threshold
 * Pointer to the scalar threshold value for maximum operation
 *
 ** @return
 * Pointer to a new tensor containing the result of the maximum operation 
 * (or NULL in case of failure)
 */
Tensor* naive_max_maxf(const Tensor* tensor, void* threshold) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;
            double max_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], max_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float max_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], max_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t max_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MAX(cast_data_original[i], max_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for element-wise maximum operation
 * @details
 * Array follows the common backend registration pattern for element-wise maximum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise maximum
 * 
 * @see nnl2_naive
 * @see naive_max_maxf
 */
Implementation max_maxf_backends[] = {
    REGISTER_BACKEND(naive_max_maxf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for element-wise maximum operation
 * @ingroup backend_system
 */
maxmaxffn max_maxf;

/** 
 * @brief Sets the backend for element-wise maximum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for maximum operation
 * @see SET_BACKEND_BY_NAME
 * @see max_maxf_backends
 */
void set_max_maxf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(max_maxf_backends, max_maxf, backend_name);
}

/** @brief 
 * Applies element-wise minimum between tensor elements and a scalar value (in-place).
 * Each element is replaced by the minimum of its current value and the specified scalar.
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be processed
 * 
 ** @param threshold 
 * Pointer to the scalar threshold value for minimum operation
 */
void naive_min_minf_inplace(Tensor* tensor, void* threshold) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double min_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], min_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float min_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], min_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t min_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] = MIN(cast_data[i], min_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}    

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place element-wise minimum operation
 * @details
 * Array follows the common backend registration pattern for in-place minimum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise minimum
 * 
 * @see nnl2_naive
 * @see naive_min_minf_inplace
 */
Implementation min_minf_inplace_backends[] = {
    REGISTER_BACKEND(naive_min_minf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for in-place element-wise minimum operation
 * @ingroup backend_system 
 */
minminfinplacefn min_minf_inplace;

/** 
 * @brief Sets the backend for in-place element-wise minimum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for minimum operation
 * @see SET_BACKEND_BY_NAME
 */
void set_min_minf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_minf_inplace_backends, min_minf_inplace, backend_name);
}

/** @brief
 * Performs element-wise minimum operation between tensor elements and a scalar value
 *
 ** @param tensor
 * Pointer to the input tensor to be processed
 *
 ** @param threshold
 * Pointer to the scalar threshold value for minimum operation
 *
 ** @return
 * Pointer to a new tensor containing the result of the minimum operation 
 * (or NULL in case of failure)
 */
Tensor* naive_min_minf(const Tensor* tensor, void* threshold) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;
            double min_val = *((double*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float min_val = *((float*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t min_val = *((int32_t*)threshold);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = MIN(cast_data_original[i], min_val);
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
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
 * @brief Backend implementations for element-wise minimum operation
 * @details
 * Array follows the common backend registration pattern for element-wise minimum
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for element-wise minimum
 * 
 * @see nnl2_naive
 * @see naive_min_minf
 */
Implementation min_minf_backends[] = {
    REGISTER_BACKEND(naive_min_minf, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for element-wise minimum operation
 * @ingroup backend_system
 */
minminffn min_minf;

/** 
 * @brief Sets the backend for element-wise minimum operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for minimum operation
 * @see SET_BACKEND_BY_NAME
 * @see min_minf_backends
 */
void set_min_minf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(min_minf_backends, min_minf, backend_name);
}

// why is everyone so mediocre

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

#endif  /** NNL2_TENSOR_ACCESSORS_H **/ 
