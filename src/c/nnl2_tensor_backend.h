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
 * Enumerations of available tensor types (INT32/INT, FLOAT32/FLOAT, FLOAT64/DOUBLE)
 */
typedef enum {
	INT32,
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

///@}



/// @{ [typedef]

/** @brief
 * All typedef declarations for tensor functions
 */

/** @brief Function pointer for in-place tensor filling operation
 ** @param tensor Target tensor to be filled
 ** @param value Pointer to the value to fill with
 ** @param dtype Data type of the value
 ** @return true if successful, false otherwise
 **/
typedef bool (*fn_inplace_fill)(Tensor*, void*, TensorType);

/** @brief Function pointer for creating an uninitialized tensor
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of tensor elements
 ** @return Pointer to newly created tensor
 **/
typedef Tensor* (*fn_empty)(const int*, int, TensorType);

/** @brief Function pointer for creating a tensor filled with zeros
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of tensor elements
 ** @return Pointer to newly created zero-filled tensor
 **/
typedef Tensor* (*fn_zeros)(const int*, int, TensorType);

/** @brief Function pointer for creating a tensor filled with ones
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of tensor elements
 ** @return Pointer to newly created one-filled tensor
 */
typedef Tensor* (*fn_ones)(const int*, int, TensorType);

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
							   const int, const int, const int, const float, const Tensor*, 
							   const int, const Tensor*, const int, const float, Tensor*, const int);

/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 **/
typedef void (*dgemminplacefn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
							   const int, const int, const int, const double, const Tensor*, 
							   const int, const Tensor*, const int, const double, Tensor*, const int);
							   
/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 **/
typedef void (*i32gemminplacefn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
							     const int, const int, const int, const int32_t, const Tensor*, 
							     const int, const Tensor*, const int, const int32_t, Tensor*, const int);							   

/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 **/
typedef Tensor* (*sgemmfn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
					       const int, const int, const int, const float, const Tensor*, 
					       const int, const Tensor*, const int, const float);

/** @ingroup GEMM_ops
 ** @see sgemminplacefn
 **/
typedef Tensor* (*dgemmfn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
					       const int, const int, const int, const double, const Tensor*, 
					       const int, const Tensor*, const int, const double);

/** @brief In-place tensor addition function pointer
 ** @param a First tensor (modified in-place)
 ** @param b Second tensor
 */
typedef void (*addinplacefn)(Tensor*, const Tensor*);		

/** @brief In-place tensor subtraction function pointer
 ** @param a First tensor (modified in-place)
 ** @param b Second tensor
 */	
typedef void (*subinplacefn)(Tensor*, const Tensor*);	

/** @brief Tensor addition function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @return New tensor containing element-wise sum a + b
 **/
typedef Tensor* (*addfn)(const Tensor*, const Tensor*);		

/** @brief Tensor subtraction function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @return New tensor containing element-wise difference a - b
 **/
typedef Tensor* (*subfn)(const Tensor*, const Tensor*);

/** @brief In-place tensor multiplication function pointer
 ** @param a First tensor (modified in-place)
 ** @param b Second tensor
 **/		
typedef void (*mulinplacefn)(Tensor*, const Tensor*);	

/** @brief In-place tensor division function pointer
 ** @param a First tensor (modified in-place)
 ** @param b Second tensor
 **/	
typedef void (*divinplacefn)(Tensor*, const Tensor*);		

/** @brief Tensor multiplication function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @return New tensor containing element-wise product a * b
 **/
typedef Tensor* (*mulfn)(const Tensor*, const Tensor*);	   

/** @brief Tensor division function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @return New tensor containing element-wise quotient a / b
 **/
typedef Tensor* (*divfn)(const Tensor*, const Tensor*);

/** @brief In-place tensor power function pointer
 ** @param a Base tensor (modified in-place)
 ** @param b Exponent tensor
 **/	
typedef void (*powinplacefn)(Tensor*, const Tensor*);

/** @brief In-place exponential function pointer
 ** @param a Input tensor (modified in-place with e^a)
 **/
typedef void (*expinplacefn)(Tensor*);

/** @brief Tensor power function pointer (creates new tensor)
 ** @param a Base tensor
 ** @param b Exponent tensor
 ** @return New tensor containing element-wise power a^b
 **/
typedef Tensor* (*powfn)(const Tensor*, const Tensor*);

/** @brief Exponential function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param inplace If true, may modify input tensor for optimization
 ** @return New tensor containing element-wise exponential e^a
 */
typedef Tensor* (*expfn)(const Tensor*, bool);

/** @brief In-place natural logarithm function pointer
 ** @param a Input tensor (modified in-place with ln(a))
 **/
typedef void (*loginplacefn)(Tensor* a);

/** @brief Natural logarithm function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param inplace If true, may modify input tensor for optimization
 ** @return New tensor containing element-wise natural logarithm ln(a)
 **/
typedef Tensor* (*logfn)(const Tensor* a, bool inplace);

/** @brief In-place matrix transpose function pointer
 ** @param a Input tensor (transposed in-place)
 **/
typedef void (*transposeinplacefn)(Tensor* a, bool force);

/** @brief Matrix transpose function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param force If true return mathematically correct result
 ** @return New tensor containing transposed matrix
 **/
typedef Tensor* (*transposefn)(const Tensor* a, bool force);

/** @brief In-place matrix transposition function pointer (view)
 ** @param a Input tensor (transposed in-place)
 **/
typedef void (*transpositioninplacefn)(Tensor* a);

/** @brief Matrix transposition function pointer (creates new tensor) (view)
 ** @param a Input tensor
 ** @return New tensor containing transposed matrix
 **/
typedef Tensor* (*transpositionfn)(const Tensor* a);

/** @brief In-place scaling function pointer
 ** @param a Input tensor (scaled in-place)
 ** @param scale Scaling factor
 **/
typedef void (*scaleinplacefn)(Tensor* a, float scale);

/** @brief Scaling function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param scale Scaling factor
 ** @param inplace If true, may modify input tensor for optimization
 ** @return New tensor containing scaled values a * scale
 **/
typedef Tensor* (*scalefn)(const Tensor* a, float scale, bool inplace);

/** @brief In-place element-wise maximum function pointer
 ** @param a First tensor (modified in-place with max(a, b))
 ** @param b Second tensor
 **/
typedef void (*maxinplacefn)(Tensor* a, const Tensor* b);

/** @brief In-place element-wise minimum function pointer
 ** @param a First tensor (modified in-place with min(a, b))
 ** @param b Second tensor
 **/
typedef void (*mininplacefn)(Tensor* a, const Tensor* b);

/** @brief Element-wise maximum function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @return New tensor containing element-wise maximum max(a, b)
 **/
typedef Tensor* (*maxfn)(const Tensor* a, const Tensor* b);

/** @brief Element-wise minimum function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @return New tensor containing element-wise minimum min(a, b)
 **/
typedef Tensor* (*minfn)(const Tensor* a, const Tensor* b);

/** @brief In-place absolute value function pointer
 ** @param a Input tensor (modified in-place with |a|)
 **/
typedef void (*absinplacefn)(Tensor* a);

/** @brief Absolute value function pointer (creates new tensor)
 ** @param a Input tensor
 ** @return New tensor containing element-wise absolute values |a|
 **/
typedef Tensor* (*absfn)(const Tensor* a);

/** @brief Horizontal stacking function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @return New tensor with horizontally stacked matrices [a | b]
 **/
typedef Tensor* (*hstackfn)(const Tensor* a, const Tensor* b);

/** @brief Vertical stacking function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @return New tensor with vertically stacked matrices [a; b]
 **/
typedef Tensor* (*vstackfn)(const Tensor* a, const Tensor* b);

/** @brief In-place ReLU activation function pointer
 ** @param a Input tensor (modified in-place with max(0, a))
 **/
typedef void (*reluinplacefn)(Tensor* a);

/** @brief ReLU activation function pointer (creates new tensor)
 ** @param a Input tensor
 ** @return New tensor containing ReLU activation max(0, a)
 **/
typedef Tensor* (*relufn)(const Tensor* a);

/** @brief In-place Leaky ReLU activation function pointer
 ** @param a Input tensor (modified in-place with max(alpha * a, a))
 ** @param alpha Negative slope coefficient
 **/
typedef void (*leakyreluinplacefn)(Tensor* a, float alpha);

/** @brief Leaky ReLU activation function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param alpha Negative slope coefficient
 ** @param inplace If true, may modify input tensor for optimization
 ** @return New tensor containing Leaky ReLU activation max(alpha * a, a)
 **/
typedef Tensor* (*leakyrelufn)(const Tensor* a, float alpha, bool inplace);

/** @brief In-place sigmoid activation function pointer
 ** @param a Input tensor (modified in-place with 1/(1 + exp(-a)))
 ** @param approx If true, use approximation for faster computation
 **/
typedef void (*sigmoidinplacefn)(Tensor* a, bool approx);

/** @brief Sigmoid activation function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param approx If true, use approximation for faster computation
 ** @return New tensor containing sigmoid activation 1/(1 + exp(-a))
 **/
typedef Tensor* (*sigmoidfn)(const Tensor* a, bool approx);

/** @brief In-place hyperbolic tangent function pointer
 ** @param a Input tensor (modified in-place with tanh(a))
 ** @param approx If true, use approximation for faster computation
 **/
typedef void (*tanhinplacefn)(Tensor* a, bool approx);

/** @brief Hyperbolic tangent function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param approx If true, use approximation for faster computation
 ** @return New tensor containing element-wise tanh(a)
 **/
typedef Tensor* (*tanhfn)(const Tensor* a, bool approx);

/** @brief Concatenation function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @param axis Axis along which to concatenate
 ** @return New tensor containing concatenated tensors along specified axis
 **/
typedef Tensor* (*concatfn)(const Tensor* a, const Tensor* b, int axis);

/** @brief Random normal distribution function pointer (creates new tensor)
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of tensor elements
 ** @param mean_ptr Pointer to mean value 
 ** @param stddev_ptr Pointer to standard deviation value 
 ** @return New tensor filled with random values from normal distribution
 **/
typedef Tensor* (*randnfn)(const int* shape, int rank, TensorType dtype, void* mean_ptr, void* stddev_ptr);

/** @brief Xavier initialization function pointer (creates new tensor)
 ** @param shape Array of dimension sizes
 ** @param rank Number of dimensions
 ** @param dtype Data type of tensor elements
 ** @param fan_in Number of input units
 ** @param fan_out Number of output units
 ** @param gain Scaling factor
 ** @param scale Additional scaling factor
 ** @return New tensor with Xavier/Glorot initialized values
 **/
typedef Tensor* (*xavierfn)(int* shape, int rank, TensorType dtype, int fan_in, int fan_out, float gain, float scale);

/** @brief Sum without specified axis function pointer
 ** @param a Input tensor (sum stored in provided memory)
 ** @param result_ptr Pointer to memory where sum result will be stored
 **/
typedef void (*sumwithoutaxisfn)(Tensor* a, void* result_ptr);

/** @brief Sum along specified axis function pointer
 ** @param a Input tensor (summed along axis in-place)
 ** @param axis Axis along which to compute sum
 **/
typedef Tensor* (*sumwithaxisfn)(Tensor* a, int axis);

/** @brief L2 norm computation function pointer
 ** @param a Input tensor
 ** @param axes Array of axes along which to compute norm
 ** @param num_axes Number of axes in the array
 **/
typedef void (*l2normfn)(const Tensor* a, int* axes, int num_axes);

/** @brief Tensor copy function pointer (creates new tensor)
 ** @param a Input tensor to copy
 ** @param dtype Copy type
 ** @return New tensor copy of the input
 **/
typedef Tensor* (*copyfn)(const Tensor* a, TensorType dtype);

/** @brief In-place addition with constant function pointer
 ** @param a Input tensor (modified in-place with a + value)
 ** @param value_ptr Pointer to constant value to add
 **/
typedef void (*addincfinplacefn)(Tensor* a, void* value_ptr);

/** @brief Addition with constant function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param value_ptr Pointer to constant value to add
 ** @return New tensor containing a + value
 **/
typedef Tensor* (*addincffn)(const Tensor* a, void* value_ptr);

/** @brief In-place subtraction with constant function pointer
 ** @param a Input tensor (modified in-place with a - value)
 ** @param value_ptr Pointer to constant value to subtract
 **/
typedef void (*subdecfinplacefn)(Tensor* a, void* value_ptr);

/** @brief Subtraction with constant function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param value_ptr Pointer to constant value to subtract
 ** @return New tensor containing a - value
 **/
typedef Tensor* (*subdecffn)(const Tensor* a, void* value_ptr);

/** @brief In-place multiplication with constant function pointer
 ** @param a Input tensor (modified in-place with a * value)
 ** @param value_ptr Pointer to constant value to multiply
 **/
typedef void (*mulmulfinplacefn)(Tensor* a, void* value_ptr);

/** @brief Multiplication with constant function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param value_ptr Pointer to constant value to multiply
 ** @return New tensor containing a * value
 **/
typedef Tensor* (*mulmulffn)(const Tensor* a, void* value_ptr);

/** @brief In-place division with constant function pointer
 ** @param a Input tensor (modified in-place with a / value)
 ** @param value_ptr Pointer to constant value to divide by
 **/
typedef void (*divdivfinplacefn)(Tensor* a, void* value_ptr);

/** @brief Division with constant function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param value_ptr Pointer to constant value to divide by
 ** @return New tensor containing a / value
 **/
typedef Tensor* (*divdivffn)(const Tensor* a, void* value_ptr);

/** @brief In-place power with constant function pointer
 ** @param a Input tensor (modified in-place with a^value)
 ** @param value_ptr Pointer to constant exponent value
 **/
typedef void (*powpowfinplacefn)(Tensor* a, void* value_ptr);

/** @brief Power with constant function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param value_ptr Pointer to constant exponent value
 ** @return New tensor containing a^value
 **/
typedef Tensor* (*powpowffn)(const Tensor* a, void* value_ptr);

/** @brief In-place maximum with constant function pointer
 ** @param a Input tensor (modified in-place with max(a, value))
 ** @param value_ptr Pointer to constant value
 **/
typedef void (*maxmaxfinplacefn)(Tensor* a, void* value_ptr);

/** @brief Maximum with constant function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param value_ptr Pointer to constant value
 ** @return New tensor containing max(a, value)
 **/
typedef Tensor* (*maxmaxffn)(Tensor* a, void* value_ptr);

/** @brief In-place minimum with constant function pointer
 ** @param a Input tensor (modified in-place with min(a, value))
 ** @param value_ptr Pointer to constant value
 **/
typedef void (*minminfinplacefn)(Tensor* a, void* value_ptr);

/** @brief Minimum with constant function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param value_ptr Pointer to constant value
 ** @return New tensor containing min(a, value)
 **/
typedef Tensor* (*minminffn)(Tensor* a, void* value_ptr);

/** @brief In-place broadcasting addition function pointer
 ** @param a First tensor (modified in-place with broadcasting)
 ** @param b Second tensor (broadcasted to match a's shape)
 **/
typedef void (*addbroadcastinginplacefn)(Tensor* a, const Tensor* b);

/** @brief Broadcasting addition function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor (broadcasted to match a's shape)
 ** @return New tensor containing broadcasted sum a + b
 **/
typedef Tensor* (*addbroadcastingfn)(const Tensor* a, const Tensor* b);

/** @brief In-place broadcasting subtraction function pointer
 ** @param a First tensor (modified in-place with broadcasting)
 ** @param b Second tensor (broadcasted to match a's shape)
 **/
typedef void (*subbroadcastinginplacefn)(Tensor* a, const Tensor* b);

/** @brief Broadcasting subtraction function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor (broadcasted to match a's shape)
 ** @return New tensor containing broadcasted difference a - b
 **/
typedef Tensor* (*subbroadcastingfn)(const Tensor* a, const Tensor* b);

/** @brief In-place broadcasting multiplication function pointer
 ** @param a First tensor (modified in-place with broadcasting)
 ** @param b Second tensor (broadcasted to match a's shape)
 **/
typedef void (*mulbroadcastinginplacefn)(Tensor* a, const Tensor* b);

/** @brief Broadcasting multiplication function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor (broadcasted to match a's shape)
 ** @return New tensor containing broadcasted product a * b
 **/
typedef Tensor* (*mulbroadcastingfn)(const Tensor* a, const Tensor* b);

/** @brief In-place broadcasting division function pointer
 ** @param a First tensor (modified in-place with broadcasting)
 ** @param b Second tensor (broadcasted to match a's shape)
 **/
typedef void (*divbroadcastinginplacefn)(Tensor* a, const Tensor* b);

/** @brief Broadcasting division function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor (broadcasted to match a's shape)
 ** @return New tensor containing broadcasted quotient a / b
 **/
typedef Tensor* (*divbroadcastingfn)(const Tensor* a, const Tensor* b);

/** @brief In-place broadcasting power function pointer
 ** @param a Base tensor (modified in-place with broadcasting)
 ** @param b Exponent tensor (broadcasted to match a's shape)
 **/
typedef void (*powbroadcastinginplacefn)(Tensor* a, const Tensor* b);

/** @brief Broadcasting power function pointer (creates new tensor)
 ** @param a Base tensor
 ** @param b Exponent tensor (broadcasted to match a's shape)
 ** @return New tensor containing broadcasted power a^b
 **/
typedef Tensor* (*powbroadcastingfn)(const Tensor* a, const Tensor* b);

/** @brief In-place broadcasting maximum function pointer
 ** @param a First tensor (modified in-place with broadcasting)
 ** @param b Second tensor (broadcasted to match a's shape)
 **/
typedef void (*maxbroadcastinginplacefn)(Tensor* a, const Tensor* b);

/** @brief In-place broadcasting minimum function pointer
 ** @param a First tensor (modified in-place with broadcasting)
 ** @param b Second tensor (broadcasted to match a's shape)
 **/
typedef void (*minbroadcastinginplacefn)(Tensor* a, const Tensor* b);

/** @brief Broadcasting maximum function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor (broadcasted to match a's shape)
 ** @return New tensor containing broadcasted maximum max(a, b)
 **/
typedef Tensor* (*maxbroadcastingfn)(const Tensor* a, const Tensor* b);

/** @brief Broadcasting minimum function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor (broadcasted to match a's shape)
 ** @return New tensor containing broadcasted minimum min(a, b)
 **/
typedef Tensor* (*minbroadcastingfn)(const Tensor* a, const Tensor* b);

/** @brief Fill tensor with data function pointer
 ** @param a Target tensor to be filled
 ** @param data_ptr Pointer to source data
 ** @param data_size Size of data in bytes
 **/
typedef void (*filltensorwithdatafn)(Tensor* a, void* data_ptr, size_t data_size);

/** @brief In-place AXPY operation function pointer (a = a + alpha * b)
 ** @param a First tensor (modified in-place)
 ** @param b Second tensor
 ** @param alpha Scaling factor
 **/
typedef void (*axpyinplacefn)(Tensor* a, const Tensor* b, float alpha);

/** @brief AXPY operation function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor
 ** @param alpha Scaling factor
 ** @return New tensor containing result of a + alpha * b
 **/
typedef Tensor* (*axpyfn)(const Tensor* a, const Tensor* b, float alpha);

/** @brief In-place AXP operation function pointer (a = a + alpha * value)
 ** @param a Input tensor (modified in-place)
 ** @param value_ptr Pointer to constant value
 ** @param alpha Scaling factor
 **/
typedef void (*axpfinplacefn)(Tensor* a, void* value_ptr, float alpha);

/** @brief AXP operation function pointer (creates new tensor)
 ** @param a Input tensor
 ** @param value_ptr Pointer to constant value
 ** @param alpha Scaling factor
 ** @return New tensor containing result of a + alpha * value
 **/
typedef Tensor* (*axpffn)(const Tensor* a, void* value_ptr, float alpha);

/** @brief In-place broadcasting AXPY operation function pointer
 ** @param a First tensor (modified in-place with broadcasting)
 ** @param b Second tensor (broadcasted to match a's shape)
 ** @param alpha Scaling factor
 **/
typedef void (*axpybroadcastinginplacefn)(Tensor* a, const Tensor* b, float alpha);

/** @brief Broadcasting AXPY operation function pointer (creates new tensor)
 ** @param a First tensor
 ** @param b Second tensor (broadcasted to match a's shape)
 ** @param alpha Scaling factor
 ** @return New tensor containing broadcasted result of a + alpha * b
 **/
typedef Tensor* (*axpybroadcastingfn)(const Tensor* a, const Tensor* b, float alpha);

/** @brief Tensor view creation function pointer
 ** @param a Input tensor
 ** @param shape New shape array
 ** @param flags View creation flags
 ** @return Pointer to view tensor sharing data with original
 **/
typedef void* (*viewfn)(Tensor* a, const int32_t* shape, uint8_t flags);

/** @brief Tensor reference getter function pointer
 ** @param a Input tensor
 ** @param indices Array of indices
 ** @param flags Access flags
 ** @return Pointer to element at specified indices
 **/
typedef void* (*trefgetterfn)(Tensor* a, const int32_t* indices, uint8_t flags);

/** @brief Tensor reference setter function pointer
 ** @param a Target tensor
 ** @param indices Array of indices
 ** @param rank Number of indices
 ** @param value_ptr Pointer to value to set
 ** @param inplace If true, modify tensor in-place
 **/
typedef void* (*trefsetterfn)(Tensor* a, int* indices, int rank, void* value_ptr, bool inplace);

/** @brief Convert double array function pointer
 ** @param dest_ptr Destination pointer for tensor data
 ** @param src_arr Source double array
 ** @param size Number of elements to convert
 ** @param dtype Target data type for conversion
 **/
typedef void (*convf64arrfn)(void* dest_ptr, double* src_arr, size_t size, TensorType dtype);

/** @brief Reshape tensor function pointer (creates new tensor or view)
 ** @param a Input tensor
 ** @param new_shape New shape array
 ** @param new_rank Number of dimensions in new shape
 ** @param copy If true, create copy instead of view
 ** @return New tensor with reshaped dimensions
 **/
typedef Tensor* (*reshapefn)(Tensor* a, int32_t* new_shape, int32_t new_rank, bool copy);

/** @brief Reinterpret tensor function pointer (creates new view)
 ** @param a Input tensor
 ** @param new_shape New shape array
 ** @param new_rank Number of dimensions in new shape
 ** @param copy If true, create copy instead of view
 ** @return New tensor with reinterpreted dimensions and strides
 **/
typedef Tensor* (*reinterpretfn)(Tensor* a, int32_t* new_shape, int32_t new_rank, bool copy);

/** @brief Returns a truncated copy of the tensor
 ** @param tensor Input tensor
 ** @param slice_from List of indices to start truncating the tensor
 ** @param slice_to List of indices for the end of the tensor truncation
 ** @return Sliced tensor (copy)
 **/
typedef Tensor* (*slicefn)(Tensor* tensor, int32_t* slice_from, int32_t* slice_t);

/** @brief Returns a truncated view of the tensor
 ** @param tensor Input tensor
 ** @param slice_from List of indices to start truncating the tensor
 ** @param slice_to List of indices for the end of the tensor truncation
 ** @return Sliced tensor (view)
 **/
typedef Tensor* (*cutfn)(Tensor* tensor, int32_t* cut_from, int32_t* cut_to);

/** @brief Applies element-wise negation to the input tensor (in-place)
 ** @details Each element of the tensor is replaced with its negated value: tensor[i] = -tensor[i]
 ** @param tensor Input tensor to be negated in-place
 ** @note This operation modifies the input tensor directly
 ** @see negfn
 **/
typedef void (*neginplacefn)(nnl2_tensor* tensor);

/** @brief Returns a new tensor with element-wise negation of the input tensor
 ** @details Each element of the tensor is replaced with its negated value: tensor[i] = -tensor[i]
 ** @param tensor Input tensor
 ** @return New tensor containing the negated values of the input tensor
 ** @note The original tensor remains unchanged.=
 ** @see neginplacefn
 **/
typedef nnl2_tensor* (*negfn)(nnl2_tensor* tensor);

/** @brief Fills the given tensor with random values from the specified range (in-place)
 ** @param tensor Tensor to fill with random values
 ** @param from Pointer to the lower bound of the random range
 ** @param to Pointer to the upper bound of the random range
 ** @see randnfn
 **/
typedef void (*randninplacefn)(nnl2_tensor* tensor, void* from, void* to);

/// @} [typedef]



/// @{

/** @brief 
 * Get human-readable name for TensorType enum
 *
 ** @param 
 * dtype Tensor data type enum value
 *
 ** @return 
 * String representation of the data type
 */
inline static char* get_tensortype_name(TensorType dtype) {
	switch(dtype) {
		case INT32:   return "INT32";
		case FLOAT32: return "FLOAT32";
		case FLOAT64: return "FLOAT64";
		default:	  return "UNKNOWN";
	}	
}

/// @}



/// @{ [backends]

/** @brief 
 * Initialize backend by selecting the fastest available implementation
 *
 ** @param backends 
 * Array of available implementations
 *
 ** @param count 
 * Number of implementations in the array
 *
 ** @return 
 * Pointer to the fastest available function implementation
 */
void* init_backend(Implementation* backends, size_t count) {
    if (count == 0) return NULL;
    
    Implementation* best = &backends[0];
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
 * Pointer to the fastest available function implementation
 */
void* einit_backend(Implementation* backends, size_t count, char* cur_pntr) {
    if (count == 0) return NULL;
    
    Implementation* best = &backends[0];
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
void set_backend_by_name(Implementation* backends, size_t count, void** target_fn, const char* backend_name) {
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
void eset_backend_by_name(Implementation* backends, size_t count, void** target_fn, const char* backend_name, char* cur_pntr) {
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
 * Pointer to the fastest implementation that is slower than current_backend,
 * or NULL if no such backend exists
 */
Implementation* nnl2_get_suboptimal_backend(Implementation* backends, size_t count, const char* current_backend) {
    Implementation* current_impl = NULL;
    Implementation* suboptimal_impl = NULL;
    
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
 * Name of the fastest implementation that is slower than current_backend,
 * or NULL if no such backend exists
 */
const char* get_suboptimal_backend_name(Implementation* backends, size_t count, const char* current_backend) {
    Implementation* impl = nnl2_get_suboptimal_backend(backends, count, current_backend);
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
const char** get_backends(Implementation* backends, size_t count) {
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

#endif  /** NNL2_TENSOR_BACKEND_H **/
