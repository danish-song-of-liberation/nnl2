#include "nnl2_core.h"

#ifndef NNL2_TENSOR_BACKEND_H
#define NNL2_TENSOR_BACKEND_H

#include <string.h>
#include <stdlib.h>

/// NNL2

/** @file nnl2_tensor_backend.h
 ** @brief Contains the tensor and implemenets structures
 ** @copyright MIT License
 ** @date 2025
 *
 * The file contains full declarations of various structures, 
 * including tensors and everything related to them
 *
 ** Filepath: nnl2/src/c/nnl2_tensor_backend.h
 ** File: nnl2_tensor_backend.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2		
 **/

/// @{ 

/** @brief 
 * Register a backend implementation with metadata
 *
 ** @param fn 
 * Function pointer to implementation
 *
 ** @param speed 
 * Speed priority value
 *
 ** @param name 
 * Human-readable backend name
 */
#define REGISTER_BACKEND(fn, speed, name) { fn, speed, name }

/// @} 



/// @{ 

/** @brief 
 * Initialize backend by selecting the fastest available implementation
 *
 ** @param fn_var 
 * Variable to store selected function pointer
 *
 ** @param backends_array 
 * Array of backend implementations
 */
#define INIT_BACKEND(fn_var, backends_array) fn_var = init_backend(backends_array, sizeof(backends_array)/sizeof(backends_array[0]))

/// @}



/// @{ 

/** @brief 
 * Initialize backend and store selected backend name
 *
 ** @param fn_var 
 * Variable to store selected function pointer
 *
 ** @param backends_array 
 * Array of backend implementations
 *
 ** @param cur_pntr 
 * Pointer to buffer for backend name storage
 */
#define EINIT_BACKEND(fn_var, backends_array, cur_pntr) fn_var = einit_backend(backends_array, sizeof(backends_array)/sizeof(backends_array[0]), cur_pntr)

/// @}



/// @{ 

/** @brief 
 * Set backend function by name
 *
 ** @param backend 
 * Array of backend implementations
 *
 ** @param fn 
 * Function pointer variable to set
 * 
 ** @param backend_name 
 * Name of backend to select
 */
#define SET_BACKEND_BY_NAME(backend, fn, backend_name) set_backend_by_name(backend, sizeof(backend)/sizeof(backend[0]), (void**)&fn, backend_name)

/// @}



/// @{ 

/** @brief 
 * Get array of available backend names
 *
 ** @param backend 
 * Array of backend implementations
 *
 ** @return 
 *Array of backend name strings
 */
#define GET_BACKENDS(backend) get_backends(backend, sizeof(backend)/sizeof(backend[0]))

/// @}



/// @{  

/** @brief 
 * Set backend function by name and store selected name
 *
 ** @param backend 
 * Array of backend implementations
 *
 ** @param fn 
 * Function pointer variable to set
 *
 ** @param backend_name 
 * Name of backend to select
 *
 ** @param cur_pntr 
 * Pointer to buffer for backend name storage
 */
#define ESET_BACKEND_BY_NAME(backend, fn, backend_name, cur_pntr) eset_backend_by_name(backend, sizeof(backend)/sizeof(backend[0]), (void**)&fn, backend_name, cur_pntr)

/// @}



/// @{  

/** @brief 
 * Define a function to get backend names for specific operation
 *
 ** @param name 
 * Operation name prefix
 */
#define DEFINE_GET_BACKENDS_FUNCTION(name) const char** get_##name##_backends() { return GET_BACKENDS(name##_backends); }

/// @}



/// @{  

/** @brief 
 * Define a function to get number of available backends for specific operation
 *
 ** @param name 
 * Operation name prefix
 */
#define DEFINE_GET_NUMS_BACKENDS_FUNCTION(name) size_t get_##name##_num_backends() { return sizeof(name##_backends) / sizeof(name##_backends[0]); }

/// @}

/** @brief Maximum length for backend name strings **/
#define MAX_BACKEND_NAME_LENGTH 32
 
/// @{  

/** @brief 
 * Enumerations of available tensor types (INT32/INT, INT64/LONG, FLOAT32/FLOAT, FLOAT64/DOUBLE)
 */
typedef enum {
	INT32,
	INT64,
	FLOAT32,
	FLOAT64
} TensorType;

/// @}



/// @{  

typedef enum {
	nnl2_type_ts,	///< nnl2_tensor
	nnl2_type_ad,   ///< nnl2_ad_tensor
	nnl2_type_unknown 
} nnl2_object_type;

/// @}



/// @{

/** @brief 
 * Tensor structure representing a multi-dimensional array
 * 
 ** @details
 * This structure holds metadata and data for a multi-dimensional tensor
 * It supports various data types and can represent both owned data and views
 */
typedef struct {
	nnl2_object_type ts_type;   ///< To separate TS tensors from AD tensors
	TensorType dtype;	  		///< Data type of tensor elements
	void* data;			  		///< Pointer to the raw tensor data
	int32_t* shape;		  		///< Array of dimension sizes 
	int32_t* strides;	  		///< Array of byte strides for each dimension
	int32_t rank;		  		///< Number of dimensions (ndim)
	int8_t magic_number;  		///< This is necessary to avoid memory corruption when releasing the tensor
	bool is_view;		        ///< Flag indicating if this is a view (not owning data)
} Tensor;

/// @}



/// @{

/** @brief Implementation structure representing a function implementation
 ** @details
 * This structure holds information about a specific implementation of a function,
 * including its performance characteristics and availability status
 */
typedef struct {
    void* fn;			  ///< Pointer to the function implementation
    int speed_priority;	  ///< Speed priority (lower = slower, higher = faster)
    const char* name;	  ///< Human-readable name of the implementation	
} Implementation;

/// @}



/// @{

/** @brief Memory layout order for tensors */
typedef enum {
	nnl2ColMajor=101,  ///< Column-major order
	nnl2RowMajor=102   ///< Row-major order
} nnl2_order;

/// @}



/// @{

/** @brief 
 * Transpose operation specification for matrix operations
 * 
 ** @details
 * Specifies whether a matrix should be transposed before performing an operation
 * This is commonly used in BLAS	 operations to indicate if the input matrix
 * should be treated as transposed without physically rearranging the data
 */
typedef enum {
	nnl2NoTrans=111,
	nnl2Trans=112,
} nnl2_transpose;

/// @}



/// @{

/** @brief 
 * Implementation versions for optimized functions
 * 
 ** @details
 * This enum defines different versions of function implementations with varying
 * levels of optimization, from naive to highly optimized hardware-specific versions.
 */
typedef enum {
	nnl2_naive,		     ///< Basic naive implementation
	nnl2_unroll_128,	 ///< Loop unrolling with 128-bit optimization
	nnl2_unroll_256,	 ///< Loop unrolling with 256-bit optimization  
	nnl2_unroll_512,	 ///< Loop unrolling with 512-bit optimization
	nnl2_avx128,		 ///< AVX 128-bit SIMD optimized implementation
	nnl2_avx256,		 ///< AVX 256-bit SIMD optimized implementation
	nnl2_avx512,		 ///< AVX 512-bit SIMD optimized implementation
	nnl2_own,			 ///< Custom optimized implementation
	nnl2_lapack,	     ///< LAPACK library implementation	
	nnl2_blas,			 ///< BLAS library implementation
	nnl2_own_2,			 ///< Hyper-optimized own nnl2 implementation
	nnl2_implver_count	 ///< Count of implementation versions 
} implver;

/// @}



///@{

/** @typedef nnl2_tensor
 ** @brief nnl2 Tensor
 **/
typedef Tensor nnl2_tensor;

/** @typedef nnl2_tensor_type
 ** @brief nnl2 Tensor Type
 **/
typedef TensorType nnl2_tensor_type;

/** @typedef nnl2_tensor_type
 ** @brief nnl2 Tensor Type
 **/
typedef Implementation nnl2_runtime_implementation;

///@}



/// @{ [typedef]

/** @brief
 * All typedef declarations for nnl2_tensor functions
 */

/** @brief Function pointer for in-place nnl2_tensor filling operation
 ** @param nnl2_tensor Target nnl2_tensor to be filled
 ** @param value Pointer to the value to fill with
 ** @param dtype Data type of the value
 ** @return true if successful, false otherwise
 **/
typedef bool (*fn_inplace_fill)(nnl2_tensor*, void*, nnl2_tensor_type);

/** @brief Function pointer for creating an uninitialized nnl2_tensor
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of nnl2_tensor elements
 ** @return Pointer to newly created nnl2_tensor
 **/
typedef nnl2_tensor* (*fn_empty)(const int*, int, nnl2_tensor_type);

/** @brief Function pointer for creating a nnl2_tensor filled with zeros
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of nnl2_tensor elements
 ** @return Pointer to newly created zero-filled nnl2_tensor
 **/
typedef nnl2_tensor* (*fn_zeros)(const int*, int, nnl2_tensor_type);

/** @brief Function pointer for creating a nnl2_tensor filled with ones
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of nnl2_tensor elements
 ** @return Pointer to newly created one-filled nnl2_tensor
 */
typedef nnl2_tensor* (*fn_ones)(const int*, int, nnl2_tensor_type);

/** @brief Single-precision GEMM in-place function pointer
 ** @defgroup GEMM_ops
 ** @param order Memory layout order (row-major or column-major)
 ** @param transa Transpose flag for matrix A
 ** @param transb Transpose flag for matrix B
 ** @param m Number of rows in matrix A and C
 ** @param n Number of columns in matrix B and C
 ** @param k Number of columns in matrix A and rows in matrix B
 ** @param alpha Scaling factor for A*B
 ** @param A Input matrix A
 ** @param lda Leading dimension of matrix A
 ** @param B Input matrix B
 ** @param ldb Leading dimension of matrix B
 ** @param beta Scaling factor for matrix C
 ** @param C Output matrix C (modified in-place)
 ** @param ldc Leading dimension of matrix C
 **/
typedef void (*sgemminplacefn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
							   const int, const int, const int, const float, const nnl2_tensor*, 
							   const int, const nnl2_tensor*, const int, const float, nnl2_tensor*, const int);

/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 **/
typedef void (*dgemminplacefn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
							   const int, const int, const int, const double, const nnl2_tensor*, 
							   const int, const nnl2_tensor*, const int, const double, nnl2_tensor*, const int);
							   
/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 **/
typedef void (*i32gemminplacefn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
							     const int, const int, const int, const int32_t, const nnl2_tensor*, 
							     const int, const nnl2_tensor*, const int, const int32_t, nnl2_tensor*, const int);							   

/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 ** @see i32gemminplacefn
 **/
typedef void (*i64gemminplacefn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
							     const int, const int, const int, const int64_t, const nnl2_tensor*, 
							     const int, const nnl2_tensor*, const int, const int64_t, nnl2_tensor*, const int);
								 
/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 **/
typedef nnl2_tensor* (*sgemmfn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
					       const int, const int, const int, const float, const nnl2_tensor*, 
					       const int, const nnl2_tensor*, const int, const float);

/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 **/
typedef nnl2_tensor* (*dgemmfn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
					       const int, const int, const int, const double, const nnl2_tensor*, 
					       const int, const nnl2_tensor*, const int, const double);

/** @brief In-place nnl2_tensor addition function pointer
 ** @param a First nnl2_tensor (modified in-place)
 ** @param b Second nnl2_tensor
 */
typedef void (*addinplacefn)(nnl2_tensor*, const nnl2_tensor*);		

/** @brief In-place nnl2_tensor subtraction function pointer
 ** @param a First nnl2_tensor (modified in-place)
 ** @param b Second nnl2_tensor
 */	
typedef void (*subinplacefn)(nnl2_tensor*, const nnl2_tensor*);	

/** @brief nnl2_tensor addition function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @return New nnl2_tensor containing element-wise sum a + b
 **/
typedef nnl2_tensor* (*addfn)(const nnl2_tensor*, const nnl2_tensor*);		

/** @brief nnl2_tensor subtraction function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @return New nnl2_tensor containing element-wise difference a - b
 **/
typedef nnl2_tensor* (*subfn)(const nnl2_tensor*, const nnl2_tensor*);

/** @brief In-place nnl2_tensor multiplication function pointer
 ** @param a First nnl2_tensor (modified in-place)
 ** @param b Second nnl2_tensor
 **/		
typedef void (*mulinplacefn)(nnl2_tensor*, const nnl2_tensor*);	

/** @brief In-place nnl2_tensor division function pointer
 ** @param a First nnl2_tensor (modified in-place)
 ** @param b Second nnl2_tensor
 **/	
typedef void (*divinplacefn)(nnl2_tensor*, const nnl2_tensor*);		

/** @brief nnl2_tensor multiplication function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @return New nnl2_tensor containing element-wise product a * b
 **/
typedef nnl2_tensor* (*mulfn)(const nnl2_tensor*, const nnl2_tensor*);	   

/** @brief nnl2_tensor division function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @return New nnl2_tensor containing element-wise quotient a / b
 **/
typedef nnl2_tensor* (*divfn)(const nnl2_tensor*, const nnl2_tensor*);

/** @brief In-place nnl2_tensor power function pointer
 ** @param a Base nnl2_tensor (modified in-place)
 ** @param b Exponent nnl2_tensor
 **/	
typedef void (*powinplacefn)(nnl2_tensor*, const nnl2_tensor*);

/** @brief In-place exponential function pointer
 ** @param a Input nnl2_tensor (modified in-place with e^a)
 **/
typedef void (*expinplacefn)(nnl2_tensor*);

/** @brief nnl2_tensor power function pointer (creates new nnl2_tensor)
 ** @param a Base nnl2_tensor
 ** @param b Exponent nnl2_tensor
 ** @return New nnl2_tensor containing element-wise power a^b
 **/
typedef nnl2_tensor* (*powfn)(const nnl2_tensor*, const nnl2_tensor*);

/** @brief Exponential function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param inplace If true, may modify input nnl2_tensor for optimization
 ** @return New nnl2_tensor containing element-wise exponential e^a
 */
typedef nnl2_tensor* (*expfn)(const nnl2_tensor*, bool);

/** @brief In-place natural logarithm function pointer
 ** @param a Input nnl2_tensor (modified in-place with ln(a))
 **/
typedef void (*loginplacefn)(nnl2_tensor* a);

/** @brief Natural logarithm function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param inplace If true, may modify input nnl2_tensor for optimization
 ** @return New nnl2_tensor containing element-wise natural logarithm ln(a)
 **/
typedef nnl2_tensor* (*logfn)(const nnl2_tensor* a, bool inplace);

/** @brief In-place matrix transpose function pointer
 ** @param a Input nnl2_tensor (transposed in-place)
 **/
typedef void (*transposeinplacefn)(nnl2_tensor* a, bool force);

/** @brief Matrix transpose function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param force If true return mathematically correct result
 ** @return New nnl2_tensor containing transposed matrix
 **/
typedef nnl2_tensor* (*transposefn)(const nnl2_tensor* a, bool force);

/** @brief In-place matrix transposition function pointer (view)
 ** @param a Input nnl2_tensor (transposed in-place)
 **/
typedef void (*transpositioninplacefn)(nnl2_tensor* a);

/** @brief Matrix transposition function pointer (creates new nnl2_tensor) (view)
 ** @param a Input nnl2_tensor
 ** @return New nnl2_tensor containing transposed matrix
 **/
typedef nnl2_tensor* (*transpositionfn)(const nnl2_tensor* a);

/** @brief In-place scaling function pointer
 ** @param a Input nnl2_tensor (scaled in-place)
 ** @param scale Scaling factor
 **/
typedef void (*scaleinplacefn)(nnl2_tensor* a, float scale);

/** @brief Scaling function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param scale Scaling factor
 ** @param inplace If true, may modify input nnl2_tensor for optimization
 ** @return New nnl2_tensor containing scaled values a * scale
 **/
typedef nnl2_tensor* (*scalefn)(const nnl2_tensor* a, float scale, bool inplace);

/** @brief In-place element-wise maximum function pointer
 ** @param a First nnl2_tensor (modified in-place with max(a, b))
 ** @param b Second nnl2_tensor
 **/
typedef void (*maxinplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place element-wise minimum function pointer
 ** @param a First nnl2_tensor (modified in-place with min(a, b))
 ** @param b Second nnl2_tensor
 **/
typedef void (*mininplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Element-wise maximum function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @return New nnl2_tensor containing element-wise maximum max(a, b)
 **/
typedef nnl2_tensor* (*maxfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Element-wise minimum function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @return New nnl2_tensor containing element-wise minimum min(a, b)
 **/
typedef nnl2_tensor* (*minfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place absolute value function pointer
 ** @param a Input nnl2_tensor (modified in-place with |a|)
 **/
typedef void (*absinplacefn)(nnl2_tensor* a);

/** @brief Absolute value function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @return New nnl2_tensor containing element-wise absolute values |a|
 **/
typedef nnl2_tensor* (*absfn)(const nnl2_tensor* a);

/** @brief Horizontal stacking function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @return New nnl2_tensor with horizontally stacked matrices [a | b]
 **/
typedef nnl2_tensor* (*hstackfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Vertical stacking function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @return New nnl2_tensor with vertically stacked matrices [a; b]
 **/
typedef nnl2_tensor* (*vstackfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place ReLU activation function pointer
 ** @param a Input nnl2_tensor (modified in-place with max(0, a))
 **/
typedef void (*reluinplacefn)(nnl2_tensor* a);

/** @brief ReLU activation function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @return New nnl2_tensor containing ReLU activation max(0, a)
 **/
typedef nnl2_tensor* (*relufn)(const nnl2_tensor* a);

/** @brief In-place Leaky ReLU activation function pointer
 ** @param a Input nnl2_tensor (modified in-place with max(alpha * a, a))
 ** @param alpha Negative slope coefficient
 **/
typedef void (*leakyreluinplacefn)(nnl2_tensor* a, float alpha);

/** @brief Leaky ReLU activation function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param alpha Negative slope coefficient
 ** @param inplace If true, may modify input nnl2_tensor for optimization
 ** @return New nnl2_tensor containing Leaky ReLU activation max(alpha * a, a)
 **/
typedef nnl2_tensor* (*leakyrelufn)(const nnl2_tensor* a, float alpha, bool inplace);

/** @brief In-place sigmoid activation function pointer
 ** @param a Input nnl2_tensor (modified in-place with 1/(1 + exp(-a)))
 ** @param approx If true, use approximation for faster computation
 **/
typedef void (*sigmoidinplacefn)(nnl2_tensor* a, bool approx);

/** @brief Sigmoid activation function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param approx If true, use approximation for faster computation
 ** @return New nnl2_tensor containing sigmoid activation 1/(1 + exp(-a))
 **/
typedef nnl2_tensor* (*sigmoidfn)(const nnl2_tensor* a, bool approx);

/** @brief In-place hyperbolic tangent function pointer
 ** @param a Input nnl2_tensor (modified in-place with tanh(a))
 ** @param approx If true, use approximation for faster computation
 **/
typedef void (*tanhinplacefn)(nnl2_tensor* a, bool approx);

/** @brief Hyperbolic tangent function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param approx If true, use approximation for faster computation
 ** @return New nnl2_tensor containing element-wise tanh(a)
 **/
typedef nnl2_tensor* (*tanhfn)(const nnl2_tensor* a, bool approx);

/** @brief Concatenation function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @param axis Axis along which to concatenate
 ** @return New nnl2_tensor containing concatenated tensors along specified axis
 **/
typedef nnl2_tensor* (*concatfn)(const nnl2_tensor* a, const nnl2_tensor* b, int axis);

/** @brief Random normal distribution function pointer (creates new nnl2_tensor)
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of nnl2_tensor elements
 ** @param mean_ptr Pointer to mean value 
 ** @param stddev_ptr Pointer to standard deviation value 
 ** @return New nnl2_tensor filled with random values from normal distribution
 **/
typedef nnl2_tensor* (*uniformfn)(const int* shape, int rank, nnl2_tensor_type dtype, void* mean_ptr, void* stddev_ptr);

/** @brief Xavier initialization function pointer (creates new nnl2_tensor)
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of nnl2_tensor elements
 ** @param fan_in Number of input units
 ** @param fan_out Number of output units
 ** @param gain Scaling factor
 ** @param scale Additional scaling factor
 ** @return New nnl2_tensor with Xavier/Glorot initialized values
 **/
typedef nnl2_tensor* (*xavierfn)(int* shape, int rank, nnl2_tensor_type dtype, int fan_in, int fan_out, float gain, float scale);

/** @brief Sum without specified axis function pointer
 ** @param a Input nnl2_tensor (sum stored in provided memory)
 ** @param result_ptr Pointer to memory where sum result will be stored
 **/
typedef void (*sumwithoutaxisfn)(nnl2_tensor* a, void* result_ptr);

/** @brief Sum along specified axis function pointer
 ** @param a Input nnl2_tensor (summed along axis in-place)
 ** @param axis Axis along which to compute sum
 **/
typedef nnl2_tensor* (*sumwithaxisfn)(nnl2_tensor* a, int axis);

/** @brief L2 norm computation function pointer
 ** @param a Input nnl2_tensor
 ** @param result Pointer to where the result will be stored
 **/
typedef void (*l2normfn)(const nnl2_tensor* a, void* result);

/** @brief nnl2_tensor copy function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor to copy
 ** @param dtype Copy type
 ** @return New nnl2_tensor copy of the input
 **/
typedef nnl2_tensor* (*copyfn)(const nnl2_tensor* a, nnl2_tensor_type dtype);

/** @brief In-place addition with constant function pointer
 ** @param a Input nnl2_tensor (modified in-place with a + value)
 ** @param value_ptr Pointer to constant value to add
 **/
typedef void (*addincfinplacefn)(nnl2_tensor* a, void* value_ptr);

/** @brief Addition with constant function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param value_ptr Pointer to constant value to add
 ** @return New nnl2_tensor containing a + value
 **/
typedef nnl2_tensor* (*addincffn)(const nnl2_tensor* a, void* value_ptr);

/** @brief In-place subtraction with constant function pointer
 ** @param a Input nnl2_tensor (modified in-place with a - value)
 ** @param value_ptr Pointer to constant value to subtract
 **/
typedef void (*subdecfinplacefn)(nnl2_tensor* a, void* value_ptr);

/** @brief Subtraction with constant function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param value_ptr Pointer to constant value to subtract
 ** @return New nnl2_tensor containing a - value
 **/
typedef nnl2_tensor* (*subdecffn)(const nnl2_tensor* a, void* value_ptr);

/** @brief In-place multiplication with constant function pointer
 ** @param a Input nnl2_tensor (modified in-place with a * value)
 ** @param value_ptr Pointer to constant value to multiply
 **/
typedef void (*mulmulfinplacefn)(nnl2_tensor* a, void* value_ptr);

/** @brief Multiplication with constant function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param value_ptr Pointer to constant value to multiply
 ** @return New nnl2_tensor containing a * value
 **/
typedef nnl2_tensor* (*mulmulffn)(const nnl2_tensor* a, void* value_ptr);

/** @brief In-place division with constant function pointer
 ** @param a Input nnl2_tensor (modified in-place with a / value)
 ** @param value_ptr Pointer to constant value to divide by
 **/
typedef void (*divdivfinplacefn)(nnl2_tensor* a, void* value_ptr);

/** @brief Division with constant function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param value_ptr Pointer to constant value to divide by
 ** @return New nnl2_tensor containing a / value
 **/
typedef nnl2_tensor* (*divdivffn)(const nnl2_tensor* a, void* value_ptr);

/** @brief In-place power with constant function pointer
 ** @param a Input nnl2_tensor (modified in-place with a^value)
 ** @param value_ptr Pointer to constant exponent value
 **/
typedef void (*powpowfinplacefn)(nnl2_tensor* a, void* value_ptr);

/** @brief Power with constant function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param value_ptr Pointer to constant exponent value
 ** @return New nnl2_tensor containing a^value
 **/
typedef nnl2_tensor* (*powpowffn)(const nnl2_tensor* a, void* value_ptr);

/** @brief In-place maximum with constant function pointer
 ** @param a Input nnl2_tensor (modified in-place with max(a, value))
 ** @param value_ptr Pointer to constant value
 **/
typedef void (*maxmaxfinplacefn)(nnl2_tensor* a, void* value_ptr);

/** @brief Maximum with constant function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param value_ptr Pointer to constant value
 ** @return New nnl2_tensor containing max(a, value)
 **/
typedef nnl2_tensor* (*maxmaxffn)(nnl2_tensor* a, void* value_ptr);

/** @brief In-place minimum with constant function pointer
 ** @param a Input nnl2_tensor (modified in-place with min(a, value))
 ** @param value_ptr Pointer to constant value
 **/
typedef void (*minminfinplacefn)(nnl2_tensor* a, void* value_ptr);

/** @brief Minimum with constant function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param value_ptr Pointer to constant value
 ** @return New nnl2_tensor containing min(a, value)
 **/
typedef nnl2_tensor* (*minminffn)(nnl2_tensor* a, void* value_ptr);

/** @brief In-place broadcasting addition function pointer
 ** @param a First nnl2_tensor (modified in-place with broadcasting)
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 **/
typedef void (*addbroadcastinginplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Broadcasting addition function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 ** @return New nnl2_tensor containing broadcasted sum a + b
 **/
typedef nnl2_tensor* (*addbroadcastingfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place broadcasting subtraction function pointer
 ** @param a First nnl2_tensor (modified in-place with broadcasting)
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 **/
typedef void (*subbroadcastinginplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Broadcasting subtraction function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 ** @return New nnl2_tensor containing broadcasted difference a - b
 **/
typedef nnl2_tensor* (*subbroadcastingfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place broadcasting multiplication function pointer
 ** @param a First nnl2_tensor (modified in-place with broadcasting)
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 **/
typedef void (*mulbroadcastinginplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Broadcasting multiplication function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 ** @return New nnl2_tensor containing broadcasted product a * b
 **/
typedef nnl2_tensor* (*mulbroadcastingfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place broadcasting division function pointer
 ** @param a First nnl2_tensor (modified in-place with broadcasting)
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 **/
typedef void (*divbroadcastinginplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Broadcasting division function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 ** @return New nnl2_tensor containing broadcasted quotient a / b
 **/
typedef nnl2_tensor* (*divbroadcastingfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place broadcasting power function pointer
 ** @param a Base nnl2_tensor (modified in-place with broadcasting)
 ** @param b Exponent nnl2_tensor (broadcasted to match a's shape)
 **/
typedef void (*powbroadcastinginplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Broadcasting power function pointer (creates new nnl2_tensor)
 ** @param a Base nnl2_tensor
 ** @param b Exponent nnl2_tensor (broadcasted to match a's shape)
 ** @return New nnl2_tensor containing broadcasted power a^b
 **/
typedef nnl2_tensor* (*powbroadcastingfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place broadcasting maximum function pointer
 ** @param a First nnl2_tensor (modified in-place with broadcasting)
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 **/
typedef void (*maxbroadcastinginplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief In-place broadcasting minimum function pointer
 ** @param a First nnl2_tensor (modified in-place with broadcasting)
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 **/
typedef void (*minbroadcastinginplacefn)(nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Broadcasting maximum function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 ** @return New nnl2_tensor containing broadcasted maximum max(a, b)
 **/
typedef nnl2_tensor* (*maxbroadcastingfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Broadcasting minimum function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 ** @return New nnl2_tensor containing broadcasted minimum min(a, b)
 **/
typedef nnl2_tensor* (*minbroadcastingfn)(const nnl2_tensor* a, const nnl2_tensor* b);

/** @brief Fill nnl2_tensor with data function pointer
 ** @param a Target nnl2_tensor to be filled
 ** @param data_ptr Pointer to source data
 ** @param data_size Size of data in bytes
 **/
typedef void (*filltensorwithdatafn)(nnl2_tensor* a, void* data_ptr, size_t data_size);

/** @brief In-place AXPY operation function pointer (a = a + alpha * b)
 ** @param a First nnl2_tensor (modified in-place)
 ** @param b Second nnl2_tensor
 ** @param alpha Scaling factor
 **/
typedef void (*axpyinplacefn)(nnl2_tensor* a, const nnl2_tensor* b, float alpha);

/** @brief AXPY operation function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor
 ** @param alpha Scaling factor
 ** @return New nnl2_tensor containing result of a + alpha * b
 **/
typedef nnl2_tensor* (*axpyfn)(const nnl2_tensor* a, const nnl2_tensor* b, float alpha);

/** @brief In-place AXP operation function pointer (a = a + alpha * value)
 ** @param a Input nnl2_tensor (modified in-place)
 ** @param value_ptr Pointer to constant value
 ** @param alpha Scaling factor
 **/
typedef void (*axpfinplacefn)(nnl2_tensor* a, void* value_ptr, float alpha);

/** @brief AXP operation function pointer (creates new nnl2_tensor)
 ** @param a Input nnl2_tensor
 ** @param value_ptr Pointer to constant value
 ** @param alpha Scaling factor
 ** @return New nnl2_tensor containing result of a + alpha * value
 **/
typedef nnl2_tensor* (*axpffn)(const nnl2_tensor* a, void* value_ptr, float alpha);

/** @brief In-place broadcasting AXPY operation function pointer
 ** @param a First nnl2_tensor (modified in-place with broadcasting)
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 ** @param alpha Scaling factor
 **/
typedef void (*axpybroadcastinginplacefn)(nnl2_tensor* a, const nnl2_tensor* b, float alpha);

/** @brief Broadcasting AXPY operation function pointer (creates new nnl2_tensor)
 ** @param a First nnl2_tensor
 ** @param b Second nnl2_tensor (broadcasted to match a's shape)
 ** @param alpha Scaling factor
 ** @return New nnl2_tensor containing broadcasted result of a + alpha * b
 **/
typedef nnl2_tensor* (*axpybroadcastingfn)(const nnl2_tensor* a, const nnl2_tensor* b, float alpha);

/** @brief nnl2_tensor view creation function pointer
 ** @param a Input nnl2_tensor
 ** @param shape New shape array
 ** @param flags View creation flags
 ** @return Pointer to view nnl2_tensor sharing data with original
 **/
typedef void* (*viewfn)(nnl2_tensor* a, const int32_t* shape, uint8_t flags);

/** @brief nnl2_tensor reference getter function pointer
 ** @param a Input nnl2_tensor
 ** @param indices Array of indices
 ** @param flags Access flags
 ** @return Pointer to element at specified indices
 **/
typedef void* (*trefgetterfn)(nnl2_tensor* a, const int32_t* indices, uint8_t flags);

/** @brief nnl2_tensor reference setter function pointer
 ** @param a Target nnl2_tensor
 ** @param indices Array of indices
 ** @param rank Number of indices
 ** @param value_ptr Pointer to value to set
 ** @param inplace If true, modify nnl2_tensor in-place
 **/
typedef void* (*trefsetterfn)(nnl2_tensor* a, int* indices, int rank, void* value_ptr, bool inplace);

/** @brief Convert double array function pointer
 ** @param dest_ptr Destination pointer for nnl2_tensor data
 ** @param src_arr Source double array
 ** @param size Number of elements to convert
 ** @param dtype Target data type for conversion
 **/
typedef void (*convf64arrfn)(void* dest_ptr, double* src_arr, size_t size, nnl2_tensor_type dtype);

/** @brief Reshape nnl2_tensor function pointer (creates new nnl2_tensor or view)
 ** @param a Input nnl2_tensor
 ** @param new_shape New shape array
 ** @param new_rank Number of dimensions in new shape
 ** @param copy If true, create copy instead of view
 ** @return New nnl2_tensor with reshaped dimensions
 **/
typedef nnl2_tensor* (*reshapefn)(nnl2_tensor* a, int32_t* new_shape, int32_t new_rank, bool copy);

/** @brief Reinterpret nnl2_tensor function pointer (creates new view)
 ** @param a Input nnl2_tensor
 ** @param new_shape New shape array
 ** @param new_rank Number of dimensions in new shape
 ** @param copy If true, create copy instead of view
 ** @return New nnl2_tensor with reinterpreted dimensions and strides
 **/
typedef nnl2_tensor* (*reinterpretfn)(nnl2_tensor* a, int32_t* new_shape, int32_t new_rank, bool copy);

/** @brief Returns a truncated copy of the nnl2_tensor
 ** @param nnl2_tensor Input nnl2_tensor
 ** @param slice_from List of indices to start truncating the nnl2_tensor
 ** @param slice_to List of indices for the end of the nnl2_tensor truncation
 ** @return Sliced nnl2_tensor (copy)
 **/
typedef nnl2_tensor* (*slicefn)(nnl2_tensor* nnl2_tensor, int32_t* slice_from, int32_t* slice_t);

/** @brief Returns a truncated view of the nnl2_tensor
 ** @param nnl2_tensor Input nnl2_tensor
 ** @param slice_from List of indices to start truncating the nnl2_tensor
 ** @param slice_to List of indices for the end of the nnl2_tensor truncation
 ** @return Sliced nnl2_tensor (view)
 **/
typedef nnl2_tensor* (*cutfn)(nnl2_tensor* nnl2_tensor, int32_t* cut_from, int32_t* cut_to);

/** @brief Applies element-wise negation to the input nnl2_tensor (in-place)
 ** @details Each element of the nnl2_tensor is replaced with its negated value: nnl2_tensor[i] = -nnl2_tensor[i]
 ** @param nnl2_tensor Input nnl2_tensor to be negated in-place
 ** @note This operation modifies the input nnl2_tensor directly
 ** @see negfn
 **/
typedef void (*neginplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Returns a new nnl2_tensor with element-wise negation of the input nnl2_tensor
 ** @details Each element of the nnl2_tensor is replaced with its negated value: nnl2_tensor[i] = -nnl2_tensor[i]
 ** @param nnl2_tensor Input nnl2_tensor
 ** @return New nnl2_tensor containing the negated values of the input nnl2_tensor
 ** @note The original nnl2_tensor remains unchanged.=
 ** @see neginplacefn
 **/
typedef nnl2_tensor* (*negfn)(nnl2_tensor* nnl2_tensor);

/** @brief Fills the given nnl2_tensor with random values from the specified range (in-place)
 ** @param nnl2_tensor nnl2_tensor to fill with random values
 ** @param from Pointer to the lower bound of the random range
 ** @param to Pointer to the upper bound of the random range
 ** @see uniformfn
 **/
typedef void (*uniforminplacefn)(nnl2_tensor* nnl2_tensor, void* from, void* to);

/** @brief Initializes the given nnl2_tensor in-place using the Xavier distribution
 ** @param nnl2_tensor nnl2_tensor to initialize
 ** @param in Number of input neurons (fan_in)
 ** @param out Number of output neurons (fan_out)
 ** @param gain Gain factor applied to the standard deviation
 ** @param distribution Distribution scaling constant (usually 2.0 or 6.0)
 ** @see xavierfn
 **/
typedef void (*xavierinplacefn)(nnl2_tensor* nnl2_tensor, int in, int out, float gain, float distribution);

/** @brief Creates a new nnl2_tensor with element-wise square root of the input nnl2_tensor
 ** @param nnl2_tensor Input nnl2_tensor to compute square root of
 ** @return New nnl2_tensor with square root values
 ** @see sqrtfn
 **/
typedef nnl2_tensor* (*sqrtfn)(const nnl2_tensor* nnl2_tensor);

/** @brief Applies element-wise square root to the input nnl2_tensor (in-place)
 ** @param nnl2_tensor nnl2_tensor to modify with square root values
 ** @see sqrtinplacefn
 **/
typedef void (*sqrtinplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Function pointer for regional AXPY inplace operation
 ** @warning Strictly internal function. DO not use it in your code
 ** @warning There is no backend change
 */
typedef void (*axpy_inplace_region_fn)(nnl2_tensor* summand, nnl2_tensor* sumend, float alpha, int* from, int* to);

/** @brief Computes Mean Squared Error between prediction and target tensors
 ** @param prediction nnl2_tensor containing predicted values
 ** @param target nnl2_tensor containing ground truth values
 ** @param record Pointer to record result
 ** @return nnl2_tensor containing MSE loss value(s)
 ** @see nnl2_tensor
 */
typedef nnl2_tensor_type (*msefn)(nnl2_tensor* prediction, nnl2_tensor* target, void* record);

/** @brief Generates nnl2_tensor with random numbers from specific distribution
 ** @param shape Array defining nnl2_tensor dimensions
 ** @param rank Number of dimensions
 ** @param dtype Data type of elements
 ** @return New nnl2_tensor with random values
 ** @see nnl2_tensor
 ** @see nnl2_tensor_type
 */
typedef nnl2_tensor* (*randfn)(int* shape, int rank, nnl2_tensor_type dtype);

/** @brief Fills existing nnl2_tensor with random numbers from standard uniform distribution [0, 1]
 ** @param nnl2_tensor nnl2_tensor to fill with random values
 ** @see nnl2_tensor
 ** @see nnl2_tensor_type
 */
typedef void (*randinplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Generates nnl2_tensor with random numbers from normal distribution N(mean, std²)
 ** @param shape Array defining nnl2_tensor dimensions
 ** @param rank Number of dimensions
 ** @param dtype Data type of elements
 ** @param mean Mean of the normal distribution
 ** @param std Standard deviation of the normal distribution
 ** @return New nnl2_tensor with random values from N(mean, std²)
 ** @see nnl2_tensor
 ** @see nnl2_tensor_type
 */
typedef nnl2_tensor* (*randnfn)(int* shape, int rank, nnl2_tensor_type dtype, double mean, double std);

/** @brief Fills existing nnl2_tensor with random numbers from normal distribution N(mean, std^2)
 ** @param nnl2_tensor nnl2_tensor to fill with random values from N(mean, std^2)
 ** @param mean Mean of the normal distribution
 ** @param std Standard deviation of the normal distribution
 ** @see nnl2_tensor
 ** @see nnl2_tensor_type
 */
typedef void (*randninplacefn)(nnl2_tensor* nnl2_tensor, double mean, double std);

/** @brief Creates a new nnl2_tensor initialized using Kaiming (He) distribution
 ** @param shape nnl2_tensor shape array
 ** @param rank nnl2_tensor rank
 ** @param dtype nnl2_tensor data type
 ** @param fan_in Number of input neurons
 ** @param fan_out Number of output neurons
 ** @param gain Gain factor
 ** @param distribution Distribution parameter
 ** @param mode Initialization mode (fan_in/fan_out/fan_avg)
 ** @return New nnl2_tensor with Kaiming initialization
 ** @see nnl2_tensor
 ** @see nnl2_tensor_type
 */
typedef nnl2_tensor* (*kaimingfn)(int* shape, int rank, nnl2_tensor_type dtype, int fan_in, int fan_out, float gain, float distribution, int mode);

/** @brief In-place Kaiming initialization of a nnl2_tensor
 ** @param nnl2_tensor Pointer to the nnl2_tensor to initialize
 ** @param fan_in Number of input neurons
 ** @param fan_out Number of output neurons
 ** @param gain Gain factor
 ** @param distribution Distribution parameter
 ** @param mode Initialization mode (fan_in/fan_out/fan_avg)
 ** @see nnl2_tensor
 */
typedef void (*kaiminginplacefn)(nnl2_tensor* nnl2_tensor, int fan_in, int fan_out, float gain, float distribution, int mode);

/** @brief In-place sine operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the nnl2_tensor to apply sine to
 ** @see nnl2_tensor
 */
typedef void (*sininplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief In-place cosine operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the nnl2_tensor to apply cosine to
 ** @see nnl2_tensor
 */
typedef void (*cosinplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Out-of-place sine operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the input nnl2_tensor
 ** @return New nnl2_tensor with sine values
 ** @see nnl2_tensor
 */
typedef nnl2_tensor* (*sinfn)(const nnl2_tensor* nnl2_tensor);

/** @brief Out-of-place cosine operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the input nnl2_tensor
 ** @return New nnl2_tensor with cosine values
 ** @see nnl2_tensor
 */
typedef nnl2_tensor* (*cosfn)(const nnl2_tensor* nnl2_tensor);

/** @brief In-place arcsine operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the nnl2_tensor to apply arcsine to
 ** @note Input values must be in the range [-1, 1] for real results
 ** @see nnl2_tensor
 */
typedef void (*asininplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief In-place arccosine operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the nnl2_tensor to apply arccosine to
 ** @note Input values must be in the range [-1, 1] for real results
 ** @see nnl2_tensor
 */
typedef void (*acosinplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Out-of-place arcsine operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the input nnl2_tensor
 ** @return New nnl2_tensor with arcsine values
 ** @note Input values must be in the range [-1, 1] for real results
 ** @see nnl2_tensor
 */
typedef nnl2_tensor* (*asinfn)(const nnl2_tensor* nnl2_tensor);

/** @brief Out-of-place arccosine operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the input nnl2_tensor
 ** @return New nnl2_tensor with arccosine values
 ** @note Input values must be in the range [-1, 1] for real results
 ** @see nnl2_tensor
 */
typedef nnl2_tensor* (*acosfn)(const nnl2_tensor* nnl2_tensor);

/** @brief In-place tangent operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the nnl2_tensor to apply tangent to
 ** @see nnl2_tensor
 */
typedef void (*taninplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief In-place arctangent operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the nnl2_tensor to apply arctangent to
 ** @see nnl2_tensor
 */
typedef void (*ataninplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Out-of-place tangent operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the input nnl2_tensor
 ** @return New nnl2_tensor with tangent values
 ** @see nnl2_tensor
 */
typedef nnl2_tensor* (*tanfn)(const nnl2_tensor* nnl2_tensor);

/** @brief Out-of-place arctangent operation on a nnl2_tensor
 ** @param nnl2_tensor Pointer to the input nnl2_tensor
 ** @return New nnl2_tensor with arctangent values (in radians)
 ** @note Returns values in the range [-π/2, π/2]
 ** @see nnl2_tensor
 */
typedef nnl2_tensor* (*atanfn)(const nnl2_tensor* nnl2_tensor);

/** @brief Concatenates multiple tensors into a single vector
 ** @param tensors Array of pointers to input tensors
 ** @param count Number of tensors in the array
 ** @param dtype Data type for the resulting nnl2_tensor
 ** @return New nnl2_tensor containing concatenated data as a vector
 ** @note All input tensors are flattened and concatenated sequentially
 ** @see nnl2_tensor
 ** @see nnl2_tensor_type
 */
typedef nnl2_tensor* (*vectorconcatfn)(nnl2_tensor** tensors, size_t count, nnl2_tensor_type dtype);

/**
 * @brief Function pointer type for uniform crossover operation
 * @ingroup nn_ga_backend_system
 */
typedef nnl2_tensor* (*nn_ga_crossover_uniform_fn)(nnl2_tensor*, nnl2_tensor*, float);

/** @brief Function pointer type for a uniform mutation operation
 ** @param nnl2_tensor Pointer to the input nnl2_tensor (parent individual)
 ** @param mutate_rate Probability of mutating each element [0.0 – 1.0]
 ** @param delta Range of uniform mutation 
 ** @return nnl2_tensor* Pointer to a new nnl2_tensor (child) after mutation
 */
typedef nnl2_tensor* (*nn_ga_mutation_uniform_fn)(nnl2_tensor* nnl2_tensor, float mutate_rate, float delta);

/** @param prediction Pointer to prediction nnl2_tensor
 ** @param target Pointer to target nnl2_tensor
 ** @param record Pointer to memory where result will be stored
 */
typedef void (*maefn)(nnl2_tensor* prediction, nnl2_tensor* target, void* record);

/** @brief Function pointer type for atan2 operation
 ** @param y Pointer to y-coordinate nnl2_tensor (numerator)
 ** @param x Pointer to x-coordinate nnl2_tensor (denominator)
 ** @return Pointer to new nnl2_tensor with atan2(y/x) values
 */
typedef nnl2_tensor* (*atan2fn)(nnl2_tensor* y, nnl2_tensor* x);

/** @brief Function pointer type for atan2 in-place operation
 ** @param y nnl2_tensor to modify
 ** @param x Const nnl2_tensor as denominator
 */
typedef void (*atan2inplacefn)(nnl2_tensor* y, const nnl2_tensor* x);

/** @brief Function pointer type for atan2 operation with broadcasting
 ** @param y y-coordinate nnl2_tensor
 ** @param x x-coordinate nnl2_tensor
 ** @return Result nnl2_tensor
 */
typedef nnl2_tensor* (*atan2broadcastingfn)(nnl2_tensor* y, nnl2_tensor* x);

/** @brief Function pointer type for atan2 operation with broadcasting (in place)
 ** @param y y-coordinate nnl2_tensor (modified in place)
 ** @param x x-coordinate nnl2_tensor
 */
typedef void (*atan2broadcastinginplacefn)(nnl2_tensor* y, const nnl2_tensor* x);

/** @brief Function pointer type for atan2 operation with scalar x
 ** @param y y-coordinate nnl2_tensor
 ** @param x Pointer to scalar x-coordinate value
 ** @return Result nnl2_tensor
 */
typedef nnl2_tensor* (*atan2correspondencefn)(const nnl2_tensor* y, void* x);

/** @brief Function pointer type for in-place atan2 operation with scalar x
 ** @param y y-coordinate nnl2_tensor (modified in place)
 ** @param x Pointer to scalar x-coordinate value
 */
typedef void (*atan2correspondenceinplacefn)(nnl2_tensor* y, void* x);

/** @brief Function pointer type for base-10 logarithm operation
 ** @param nnl2_tensor Input nnl2_tensor
 ** @param save_type Flag to save data type for special case (all elements = 1)
 ** @return Result nnl2_tensor
 */
typedef nnl2_tensor* (*log10fn)(const nnl2_tensor* nnl2_tensor, bool save_type);

/** @brief Function pointer type for in-place base-10 logarithm operation
 ** @param nnl2_tensor nnl2_tensor to be modified in place
 */
typedef void (*log10inplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Function pointer type for base-2 logarithm operation
 ** @param nnl2_tensor Input nnl2_tensor
 ** @param save_type Flag to save data type for special case (all elements are powers of two)
 ** @return Result nnl2_tensor
 */
typedef nnl2_tensor* (*log2fn)(const nnl2_tensor* nnl2_tensor, bool save_type);

/** @brief Function pointer type for in-place base-2 logarithm operation
 ** @param nnl2_tensor nnl2_tensor to be modified in place
 */
typedef void (*log2inplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Function pointer type for log1p (log(1+x)) operation
 ** @param nnl2_tensor Input nnl2_tensor
 ** @param save_type Flag to save data type for special case (all elements = 0)
 ** @return Result nnl2_tensor
 */
typedef nnl2_tensor* (*log1pfn)(const nnl2_tensor* nnl2_tensor, bool save_type);

/** @brief Function pointer type for in-place log1p (log(1+x)) operation
 ** @param nnl2_tensor nnl2_tensor to be modified in place
 */
typedef void (*log1pinplacefn)(nnl2_tensor* nnl2_tensor);

/** @brief Function pointer type for row assignment operation
 ** @param dst Destination 3D nnl2_tensor [batch, seq, features]
 ** @param seq_index Sequence index to assign to (0..seq_length-1)
 ** @param src Source 2D nnl2_tensor [batch, features]
 */
typedef int (*assignrowfn)(nnl2_tensor* dst, int seq_index, nnl2_tensor* src);

/** @brief Function pointer type for double-precision Singular Value Decomposition (DGESVD) operation
 ** @param order Memory layout ordering (RowMajor or ColumnMajor)
 ** @param jobu Specifies options for computing left singular vectors U:
 ** 'A': all m columns of U are returned
 ** 'S': first min(m,n) columns of U are returned  
 ** 'O': first min(m,n) columns of U overwrite input A
 ** 'N': no columns of U are computed
 ** @param jobvt Specifies options for computing right singular vectors V^T:
 ** 'A': all n rows of V^T are returned
 ** 'S': first min(m,n) rows of V^T are returned
 ** 'O': first min(m,n) rows of V^T overwrite input A
 ** 'N': no rows of V^T are computed
 ** @param m Number of rows of the input matrix A
 ** @param n Number of columns of the input matrix A
 ** @param a Input tensor containing matrix A [m × n] (modified if jobu='O' or jobvt='O')
 ** @param lda Leading dimension of matrix A (>= n if RowMajor, >= m if ColMajor)
 ** @param s Output tensor for singular values [min(m,n)] (sorted descending)
 ** @param u Output tensor for left singular vectors U (size depends on jobu):
 ** jobu='A': [m × m] matrix
 ** jobu='S': [m × min(m,n)] matrix
 ** jobu='N': not referenced (can be NULL)
 ** @param ldu Leading dimension of matrix U (>= m if RowMajor, >= m if ColMajor for 'A', >= min(m,n) if 'S')
 ** @param vt Output tensor for right singular vectors V^T (size depends on jobvt):
 ** jobvt='A': [n × n] matrix
 ** jobvt='S': [min(m,n) × n] matrix
 ** jobvt='N': not referenced (can be NULL)
 ** @param ldvt Leading dimension of matrix VT (>= n if RowMajor, >= n if ColMajor for 'A', >= min(m,n) if 'S')
 ** @param superb Workspace tensor [min(m,n)-1] for internal use
 ** @return Integer info code:
 **  0: successful exit
 ** <0: the -info-th argument had an illegal value
 ** >0: the algorithm failed to converge
 ** @note Performs the decomposition: A = U × diag(s) × V^T
 ** @note All tensors must be of FLOAT64 (double) data type
 ** @note Leading dimensions must satisfy: lda >= n (RowMajor) or lda >= m (ColMajor) and similarly for ldu and ldvt
 ** @see LAPACKE_dgesvd()
 ** @see nnl2_lapacke_f64svd()
 */
typedef int (*f64dgesvdfn)(const nnl2_order order, const char jobu, const char jobvt,
                           const int m, const int n, nnl2_tensor* a, const int lda,
                           nnl2_tensor* s, nnl2_tensor* u, const int ldu,
                           nnl2_tensor* vt, const int ldvt, nnl2_tensor* superb);
						
/** @brief Function pointer type for single-precision Singular Value Decomposition (SGESVD) operation
 ** @param order Memory layout ordering (RowMajor or ColumnMajor)
 ** @param jobu Specifies options for computing left singular vectors U:
 ** 'A': all m columns of U are returned
 ** 'S': first min(m,n) columns of U are returned  
 ** 'O': first min(m,n) columns of U overwrite input A
 ** 'N': no columns of U are computed
 ** @param jobvt Specifies options for computing right singular vectors V^T:
 ** 'A': all n rows of V^T are returned
 ** 'S': first min(m,n) rows of V^T are returned
 ** 'O': first min(m,n) rows of V^T overwrite input A
 ** 'N': no rows of V^T are computed
 ** @param m Number of rows of the input matrix A
 ** @param n Number of columns of the input matrix A
 ** @param a Input tensor containing matrix A [m × n] (modified if jobu='O' or jobvt='O')
 ** @param lda Leading dimension of matrix A (>= n if RowMajor, >= m if ColMajor)
 ** @param s Output tensor for singular values [min(m,n)] (sorted descending)
 ** @param u Output tensor for left singular vectors U (size depends on jobu):
 ** jobu='A': [m × m] matrix
 ** jobu='S': [m × min(m,n)] matrix
 ** jobu='N': not referenced (can be NULL)
 ** @param ldu Leading dimension of matrix U (>= m if RowMajor, >= m if ColMajor for 'A', >= min(m,n) if 'S')
 ** @param vt Output tensor for right singular vectors V^T (size depends on jobvt):
 ** jobvt='A': [n × n] matrix
 ** jobvt='S': [min(m,n) × n] matrix
 ** jobvt='N': not referenced (can be NULL)
 ** @param ldvt Leading dimension of matrix VT (>= n if RowMajor, >= n if ColMajor for 'A', >= min(m,n) if 'S')
 ** @param superb Workspace tensor [min(m,n)-1] for internal use
 ** @return Integer info code:
 **  0: successful exit
 ** <0: the -info-th argument had an illegal value
 ** >0: the algorithm failed to converge
 ** @note Performs the decomposition: A = U × diag(s) × V^T
 ** @note All tensors must be of FLOAT32 (float) data type
 ** @note Leading dimensions must satisfy: lda >= n (RowMajor) or lda >= m (ColMajor) and similarly for ldu and ldvt
 ** @see LAPACKE_sgesvd()
 ** @see nnl2_lapacke_f32sgesvd()
 */
typedef int (*f32sgesvdfn)(const nnl2_order order, const char jobu, const char jobvt,
                           const int m, const int n, nnl2_tensor* a, const int lda,
                           nnl2_tensor* s, nnl2_tensor* u, const int ldu,
                           nnl2_tensor* vt, const int ldvt, nnl2_tensor* superb);						
						
/** @brief Function pointer type for double-precision Singular Value Decomposition using Divide-and-Conquer (DGESDD) operation
 ** @param order Memory layout ordering (RowMajor or ColumnMajor)
 ** @param jobz Specifies options for computing singular vectors:
 ** 'A': all m columns of U and all n rows of V^T are returned
 ** 'S': first min(m,n) columns of U and rows of V^T are returned
 ** 'O': 
 **     If m >= n: first n columns of U overwrite input A, V^T is computed
 **     If m < n: first m rows of V^T overwrite input A, U is computed
 ** 'N': neither U nor V^T are computed
 ** @param m Number of rows of the input matrix A
 ** @param n Number of columns of the input matrix A
 ** @param a Input tensor containing matrix A [m × n] (modified if jobz='O')
 ** @param lda Leading dimension of matrix A (>= n if RowMajor, >= m if ColMajor)
 ** @param s Output tensor for singular values [min(m,n)] (sorted descending)
 ** @param u Output tensor for left singular vectors U (size depends on jobz):
 ** jobz='A': [m × m] matrix
 ** jobz='S': [m × min(m,n)] matrix
 ** jobz='O' and m >= n: [m × n] matrix (overwrites A)
 ** jobz='N': not referenced (can be NULL)
 ** @param ldu Leading dimension of matrix U:
 ** jobz='A' or 'S': ldu >= m
 ** jobz='O' and m >= n: ldu >= m
 ** jobz='N': not referenced
 ** @param vt Output tensor for right singular vectors V^T (size depends on jobz):
 ** jobz='A': [n × n] matrix
 ** jobz='S': [min(m,n) × n] matrix
 ** jobz='O' and m < n: [n × m] matrix (overwrites A)
 ** jobz='N': not referenced (can be NULL)
 ** @param ldvt Leading dimension of matrix VT:
 ** jobz='A': ldvt >= n
 ** jobz='S': ldvt >= min(m,n)
 ** jobz='O' and m < n: ldvt >= n
 ** jobz='N': not referenced
 ** @param iwork Workspace integer tensor [8*min(m,n)] for internal use (must be INT32 type)
 ** @return Integer info code:
 **  0: successful exit
 ** <0: the -info-th argument had an illegal value
 ** >0: the algorithm failed to converge
 ** @note Performs the decomposition: A = U × diag(s) × V^T using divide-and-conquer algorithm
 ** @note All float tensors must be of FLOAT64 (double) data type, iwork must be INT32
 ** @note Leading dimensions must satisfy: lda >= n (RowMajor) or lda >= m (ColMajor)
 ** @note dgesdd is typically faster than dgesvd for large matrices but uses more workspace
 ** @warning When jobz='O', input matrix A is overwritten with left/right singular vectors
 ** @see LAPACKE_dgesdd()
 ** @see nnl2_lapacke_f64dgesdd()
 ** @see f64dgesvdfn
 */
typedef int (*f64dgesddfn)(const nnl2_order order, const char jobz,
                           const int m, const int n, nnl2_tensor* a, const int lda,
                           nnl2_tensor* s, nnl2_tensor* u, const int ldu,
                           nnl2_tensor* vt, const int ldvt, nnl2_tensor* iwork);

/** @brief Function pointer type for single-precision Singular Value Decomposition using Divide-and-Conquer (SGESDD) operation
 ** @param order Memory layout ordering (RowMajor or ColumnMajor)
 ** @param jobz Specifies options for computing singular vectors:
 ** 'A': all m columns of U and all n rows of V^T are returned
 ** 'S': first min(m,n) columns of U and rows of V^T are returned
 ** 'O': 
 **     If m >= n: first n columns of U overwrite input A, V^T is computed
 **     If m < n: first m rows of V^T overwrite input A, U is computed
 ** 'N': neither U nor V^T are computed
 ** @param m Number of rows of the input matrix A
 ** @param n Number of columns of the input matrix A
 ** @param a Input tensor containing matrix A [m × n] (modified if jobz='O')
 ** @param lda Leading dimension of matrix A (>= n if RowMajor, >= m if ColMajor)
 ** @param s Output tensor for singular values [min(m,n)] (sorted descending)
 ** @param u Output tensor for left singular vectors U (size depends on jobz):
 ** jobz='A': [m × m] matrix
 ** jobz='S': [m × min(m,n)] matrix
 ** jobz='O' and m >= n: [m × n] matrix (overwrites A)
 ** jobz='N': not referenced (can be NULL)
 ** @param ldu Leading dimension of matrix U:
 ** jobz='A' or 'S': ldu >= m
 ** jobz='O' and m >= n: ldu >= m
 ** jobz='N': not referenced
 ** @param vt Output tensor for right singular vectors V^T (size depends on jobz):
 ** jobz='A': [n × n] matrix
 ** jobz='S': [min(m,n) × n] matrix
 ** jobz='O' and m < n: [n × m] matrix (overwrites A)
 ** jobz='N': not referenced (can be NULL)
 ** @param ldvt Leading dimension of matrix VT:
 ** jobz='A': ldvt >= n
 ** jobz='S': ldvt >= min(m,n)
 ** jobz='O' and m < n: ldvt >= n
 ** jobz='N': not referenced
 ** @param iwork Workspace integer tensor [8*min(m,n)] for internal use (must be INT32 type)
 ** @return Integer info code:
 **  0: successful exit
 ** <0: the -info-th argument had an illegal value
 ** >0: the algorithm failed to converge
 ** @note Performs the decomposition: A = U × diag(s) × V^T using divide-and-conquer algorithm
 ** @note All float tensors must be of FLOAT32 (float) data type, iwork must be INT32
 ** @note Leading dimensions must satisfy: lda >= n (RowMajor) or lda >= m (ColMajor)
 ** @note sgesdd is typically faster than sgesvd for large matrices but uses more workspace
 ** @warning When jobz='O', input matrix A is overwritten with left/right singular vectors
 ** @see LAPACKE_sgesdd()
 ** @see nnl2_lapacke_f32sgesdd()
 ** @see f32sgesvdfn
 */
typedef int (*f32sgesddfn)(const nnl2_order order, const char jobz,
                           const int m, const int n, nnl2_tensor* a, const int lda,
                           nnl2_tensor* s, nnl2_tensor* u, const int ldu,
                           nnl2_tensor* vt, const int ldvt, nnl2_tensor* iwork);						
						
/** @brief Function pointer type for integer arange operation
 ** @param from Starting value
 ** @param to Ending value (exclusive)
 ** @param step Step size
 ** @param dtype Data type of the tensor
 ** @return Result tensor with arange values
 */
typedef nnl2_tensor* (*nnl2_int_arangefn)(int64_t from, int64_t to, int64_t step, nnl2_tensor_type dtype);

/** @brief Function pointer type for float arange operation
 ** @param from Starting value
 ** @param to Ending value (exclusive)
 ** @param step Step size
 ** @param dtype Data type of the tensor
 ** @return Result tensor with arange values
 */
typedef nnl2_tensor* (*nnl2_float_arangefn)(float from, float to, float step, nnl2_tensor_type dtype);						
						
/** @brief Function pointer type for integer linspace operation
 ** @param start Starting value (inclusive)
 ** @param stop Ending value (inclusive when endpoint=true)
 ** @param num Number of samples to generate
 ** @param endpoint If true, stop is included as last sample
 ** @param dtype Data type of the tensor (only INT32 or INT64)
 ** @return Result tensor with linearly spaced values
 ** @note For integer types, values are rounded to nearest integer
 ** @warning For num=0 returns empty tensor, for num=1 returns [start]
 */
typedef nnl2_tensor* (*nnl2_int_linspacefn)(int64_t start, int64_t stop, int64_t num, bool endpoint, nnl2_tensor_type dtype);

/** @brief Function pointer type for float linspace operation
 ** @param start Starting value (inclusive)
 ** @param stop Ending value (inclusive when endpoint=true)
 ** @param num Number of samples to generate
 ** @param endpoint If true, stop is included as last sample
 ** @param dtype Data type of the tensor (only FLOAT32 or FLOAT64)
 ** @return Result tensor with linearly spaced values
 ** @note By default includes both endpoints (endpoint=true)
 ** @warning Floating-point precision may affect exact endpoint values
 */
typedef nnl2_tensor* (*nnl2_float_linspacefn)(float start, float stop, int64_t num, bool endpoint, nnl2_tensor_type dtype);		

/** @brief Function pointer type for double-precision LU factorization (dgetrf) 
 ** @param order Matrix layout: nnl2RowMajor or nnl2ColMajor
 ** @param m Number of rows of matrix A
 ** @param n Number of columns of matrix A
 ** @param a Input/output matrix A (shape m×n, dtype FLOAT64)
 ** @param lda Leading dimension of A
 ** @param ipiv Output pivot indices (size min(m,n), dtype INT32)
 ** @return int Status code: 0=success, >0=singular, <0=error
 */
typedef int (*f64dgetrffn)(const nnl2_order order, const int m, const int n, nnl2_tensor* a, const int lda, nnl2_tensor* ipiv);		
						   
/** @brief Function pointer type for single-precision LU factorization (sgetrf)
 ** @param order Matrix layout: nnl2RowMajor or nnl2ColMajor
 ** @param m Number of rows of matrix A
 ** @param n Number of columns of matrix A
 ** @param a Input/output matrix A (shape m×n, dtype FLOAT32)
 ** @param lda Leading dimension of A
 ** @param ipiv Output pivot indices (size min(m,n), dtype INT32)
 ** @return int Status code: 0=success, >0=singular, <0=error
 */
typedef int (*f32sgetrffn)(const nnl2_order order, const int m, const int n, nnl2_tensor* a, const int lda, nnl2_tensor* ipiv);				
						
/// @} [typedef]



/// @{

/** @brief 
 * Get human-readable name for nnl2_tensor_type enum
 *
 ** @param 
 * dtype nnl2_tensor data type enum value
 *
 ** @return 
 * String representation of the data type
 */
inline static char* get_tensortype_name(nnl2_tensor_type dtype) {
	switch(dtype) {
		case INT32:   return "INT32";
		case INT64:   return "INT64";
		case FLOAT32: return "FLOAT32";
		case FLOAT64: return "FLOAT64";
		default:	  return "UNKNOWN";
	}	
}

/// @}



/// @{ [backends]

/** @brief 
 * Initialize backend by selecting the fastest available nnl2_runtime_implementation
 *
 ** @param backends 
 * Array of available implementations
 *
 ** @param count 
 * Number of implementations in the array
 *
 ** @return 
 * Pointer to the fastest available function nnl2_runtime_implementation
 */
void* init_backend(nnl2_runtime_implementation* backends, size_t count) {
    if (count == 0) return NULL;
    
    nnl2_runtime_implementation* best = &backends[0];
    for (size_t i = 1; i < count; i++) {
        if (backends[i].speed_priority > best->speed_priority) {
            best = &backends[i];
        }
    }

    return best->fn;
}

/** @brief 
 * Initialize backend and store the selected backend name
 *
 ** @param backends 
 * Array of available implementations
 *
 ** @param count 
 * Number of implementations in the array
 *
 ** @param cur_pntr 
 * Pointer to buffer for storing backend name
 *
 ** @return 
 * Pointer to the fastest available function nnl2_runtime_implementation
 */
void* einit_backend(nnl2_runtime_implementation* backends, size_t count, char* cur_pntr) {
    if (count == 0) return NULL;
    
    nnl2_runtime_implementation* best = &backends[0];
    for (size_t i = 1; i < count; i++) {
        if (backends[i].speed_priority > best->speed_priority) {
            best = &backends[i];
        }
    }
	
	strncpy(cur_pntr, best->name, MAX_BACKEND_NAME_LENGTH - 1);
    cur_pntr[MAX_BACKEND_NAME_LENGTH - 1] = '\0';

    return best->fn;
}

/** @brief 
 * Set backend function by name
 *
 ** @param backends 
 * Array of available implementations
 *
 ** @param count 
 * Number of implementations in the array
 *
 ** @param target_fn
 * Pointer to function pointer to be set
 *
 ** @param backend_name 
 * Name of the backend to select
 */
void set_backend_by_name(nnl2_runtime_implementation* backends, size_t count, void** target_fn, const char* backend_name) {
    for (size_t i = 0; i < count; i++) {
        if (strcmp(backends[i].name, backend_name) == 0) {
            *target_fn = backends[i].fn;
            return;
        }
    }
}

/** @brief 
 * Set backend function by name and store the selected backend name
 *
 ** @param backends 
 * Array of available implementations
 *
 ** @param count 
 * Number of implementations in the array
 *
 ** @param target_fn 
 * Pointer to function pointer to be set
 *
 ** @param backend_name 
 * Name of the backend to select
 *
 ** @param cur_pntr 
 * Pointer to buffer for storing backend name
 */
void eset_backend_by_name(nnl2_runtime_implementation* backends, size_t count, void** target_fn, const char* backend_name, char* cur_pntr) {
    for (size_t i = 0; i < count; i++) {
        if (strcmp(backends[i].name, backend_name) == 0) {
            *target_fn = backends[i].fn;
			strncpy(cur_pntr, backends[i].name, MAX_BACKEND_NAME_LENGTH - 1);
			cur_pntr[MAX_BACKEND_NAME_LENGTH - 1] = '\0';
            return;
        }
    }
}

/** @brief 
 * Get the fastest backend that is slower than the current backend
 *
 ** @param backends 
 * Array of available implementations
 *
 ** @param count 
 * Number of implementations in the array
 *
 ** @param current_backend 
 * Name of the current backend
 *
 ** @return 
 * Pointer to the fastest nnl2_runtime_implementation that is slower than current_backend,
 * or NULL if no such backend exists
 */
nnl2_runtime_implementation* nnl2_get_suboptimal_backend(nnl2_runtime_implementation* backends, size_t count, const char* current_backend) {
    nnl2_runtime_implementation* current_impl = NULL;
    nnl2_runtime_implementation* suboptimal_impl = NULL;
    
    // Find the current backend and its speed priority
    for (size_t i = 0; i < count; i++) {
        if (strcmp(backends[i].name, current_backend) == 0) {
            current_impl = &backends[i];
            break;
        }
    }
    
    // If current backend not found, return NULL
    if (current_impl == NULL) {
        return NULL;
    }
    
    // Find the fastest backend that is slower than current backend
    for (size_t i = 0; i < count; i++) {
        // Skip the current backend itself
        if (strcmp(backends[i].name, current_backend) == 0) {
            continue;
        }
        
        // Consider only backends that are slower than current
        if (backends[i].speed_priority < current_impl->speed_priority) {
            // If no suboptimal candidate yet, or this one is faster than current candidate
            if (suboptimal_impl == NULL || backends[i].speed_priority > suboptimal_impl->speed_priority) {
                suboptimal_impl = &backends[i];
            }
        }
    }
    
    return suboptimal_impl;
}

/** @brief 
 * Get the name of the fastest backend that is slower than the current backend
 *
 ** @param backends 
 * Array of available implementations
 *
 ** @param count 
 * Number of implementations in the array
 *
 ** @param current_backend 
 * Name of the current backend
 *
 ** @return 
 * Name of the fastest nnl2_runtime_implementation that is slower than current_backend,
 * or NULL if no such backend exists
 */
const char* get_suboptimal_backend_name(nnl2_runtime_implementation* backends, size_t count, const char* current_backend) {
    nnl2_runtime_implementation* impl = nnl2_get_suboptimal_backend(backends, count, current_backend);
    return impl ? impl->name : NULL;
}

/** @brief 
 * Get array of available backend names
 *
 ** @param backends 
 * Array of available implementations
 *
 ** @param count 
 * Number of implementations in the array
 *
 ** @return 
 * Array of backend name strings
 */
const char** get_backends(nnl2_runtime_implementation* backends, size_t count) {
    static const char* backend_names[nnl2_implver_count]; 
	
    if (count > nnl2_implver_count) {
        return NULL;
    }

    for (size_t it = 0; it < count; it++) {
        backend_names[it] = backends[it].name;
    }

    return backend_names;
}

/// @} [backends]



/// @{ [other]

/** @brief 
 * Lisp wrapper for getting nnl2_tensor.magic_number
 */
int8_t nnl2_ts_get_magic_number(nnl2_tensor* nnl2_tensor) {
	return nnl2_tensor -> magic_number;
}

/** @brief 
 * Lisp wrapper for setting nnl2_tensor.magic_number
 */
void nnl2_ts_set_magic_number(nnl2_tensor* nnl2_tensor, int8_t new_magic) {
	nnl2_tensor -> magic_number = new_magic;
}

/// @} [other]

#endif  /** NNL2_TENSOR_BACKEND_H **/
