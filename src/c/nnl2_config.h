#ifndef NNL2_CONFIG_H
#define NNL2_CONFIG_H

/** @file nnl2_config.h
 ** @brief Contains auxiliary macros for other tensor files
 ** @copyright MIT License
 ** @date 2025
 ** Filepath: nnl2/src/c/nnl2_config.h
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
	#define NNL2_THREAD_CREATE_ERROR(error_code, function_name) NNL2_FATAL("Thread creation failed in %s: error code %d", function_name, error_code)
	#define NNL2_THREAD_JOIN_ERROR(error_code, function_name) NNL2_ERROR("Thread join failed in %s: error code %d", function_name, error_code)
	#define NNL2_THREAD_ERROR(function_name) NNL2_ERROR("Thread operation failed in %s", function_name)
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
												    **  when I was making a smart tensor output, I was thinking about threashold as in torch, 
												    **  but I abandoned this solution in favor of my own solution. these may be experimental 
												    **  macros that I later removed in format function **/
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

#endif /** NNL2_CONFIG_H **/
