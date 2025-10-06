#ifndef NNL2_PARALLEL_BACKEND_H
#define NNL2_PARALLEL_BACKEND_H

/** @file nnl2_parallel_backend.h
 ** @brief I tried adding a pool, but it was slower
 **/

///@{ [single_arr_ptask]

typedef struct {
	void* data;    ///< Pointer to an array with tensor data
	size_t start;  ///< Index for entering the data array
	size_t end;    ///< Index for the end of the data entry into the array
} single_arr_ptask;  

///@} [single_arr_ptask]



///@{ [leaky_relu_single_arr_ptask]

typedef struct {
    void* data;	   ///< Pointer to an array with tensor data
    size_t start;  ///< Index for entering the data array
    size_t end;    ///< Index for the end of the data entry into the array
    float alpha;   ///< Negative slope for leaky relu
} leaky_relu_single_arr_ptask;

///@} [leaky_relu_single_arr_ptask]



///@{ [fill_ptask]

typedef struct {
    void* data;        ///< Pointer to data array
    size_t start;      ///< Start index for this thread
    size_t end;        ///< End index for this thread  
    void* value;       ///< Pointer to fill value
    TensorType dtype;  ///< Data type of elements
    bool aligned;      ///< Whether memory is aligned
} fill_ptask;

///@} [fill_ptask]



///@{ [add_ptask]

typedef struct {
    const void* summand_data;  ///< Pointer to summand tensor data 
    const void* addend_data;   ///< Pointer to addend tensor data 
    void* result_data;         ///< Pointer to result tensor data
    size_t start;              ///< Start index for this thread 
    size_t end;                ///< End index for this thread 
    TensorType dtype_summand;  ///< Data type of summand tensor 
    TensorType dtype_addend;   ///< Data type of addend tensor 
    TensorType result_dtype;   ///< Data type of result tensor 
} add_ptask;

///@} [add_ptask]



///@{ [sub_ptask]

typedef struct {
    const void* minuend_data;     ///< Pointer to minuend tensor data 
    const void* subtrahend_data;  ///< Pointer to subtrahend tensor data 
    void* result_data;            ///< Pointer to result tensor data 
    size_t start;                 ///< Start index for this thread 
    size_t end;                   ///< End index for this thread 
    TensorType dtype_minuend;     ///< Data type of minuend tensor 
    TensorType dtype_subtrahend;  ///< Data type of subtrahend tensor 
    TensorType result_dtype;      ///< Data type of result tensor 
} sub_ptask;

///@} [sub_ptask]



///@{ [mul_ptask]

typedef struct {
    const void* multiplicand_data; ///< Pointer to multiplicand tensor data 
    const void* multiplier_data;   ///< Pointer to multiplier tensor data 
    void* result_data;             ///< Pointer to result tensor data 
    size_t start;                  ///< Start index for this thread 
    size_t end;                    ///< End index for this thread 
    TensorType dtype_multiplicand; ///< Data type of multiplicand tensor 
    TensorType dtype_multiplier;   ///< Data type of multiplier tensor 
    TensorType result_dtype;       ///< Data type of result tensor 
} mul_ptask;

///@} [mul_ptask]



///@{ [div_ptask]

typedef struct {
    const void* dividend_data;    ///< Pointer to dividend tensor data 
    const void* divisor_data;     ///< Pointer to divisor tensor data 
    void* result_data;            ///< Pointer to result tensor data
    size_t start;                 ///< Start index for this thread 
    size_t end;                   ///< End index for this thread 
    TensorType dtype_dividend;    ///< Data type of dividend tensor 
    TensorType dtype_divisor;     ///< Data type of divisor tensor 
    TensorType result_dtype;      ///< Data type of result tensor 
} div_ptask;

///@} [div_ptask]



///@{ [abs_ptask]

typedef struct {
    const void* input_data;      ///< Pointer to input tensor data 
    void* result_data;           ///< Pointer to result tensor data 
    size_t start;                ///< Start index for this thread 
    size_t end;                  ///< End index for this thread 
    TensorType dtype;            ///< Data type of tensor 
} abs_ptask;

///@} [abs_ptask]



///@{ [abs_inplace_ptask]

/** @brief
 * Task structure for parallel in-place absolute value operation
 */
typedef struct {
    void* data;                 ///< Pointer to tensor data 
    size_t start;               ///< Start index for this thread 
    size_t end;                 ///< End index for this thread 
    TensorType dtype;           ///< Data type of tensor 
} abs_inplace_ptask;

///@} [abs_inplace_ptask]



///@{ [addinplace_ptask]
	
typedef struct {
    void* summand_data;           ///< Pointer to summand tensor data (mutable) 
    const void* addend_data;      ///< Pointer to addend tensor data (read-only) 
    size_t start;                 ///< Start index for this thread's chunk 
    size_t end;                   ///< End index for this thread's chunk 
    TensorType dtype_summand;     ///< Data type of summand tensor 
    TensorType dtype_addend;      ///< Data type of addend tensor 
    bool aligned_summand;         ///< Flag indicating if summand data is 32-byte aligned 
    bool aligned_addend;          ///< Flag indicating if addend data is 32-byte aligned 
    size_t addend_step;           ///< Step size in bytes for addend data access 
} addinplace_ptask;

///@} [addinplace_ptask]



///@{ [subinplace_ptask]

typedef struct {
    void* minuend_data;           ///< Pointer to minuend tensor data (mutable) 
    const void* subtrahend_data;  ///< Pointer to subtrahend tensor data (read-only) 
    size_t start;                 ///< Start index for this thread's chunk 
    size_t end;                   ///< End index for this thread's chunk 
    TensorType dtype_minuend;     ///< Data type of minuend tensor 
    TensorType dtype_subtrahend;  ///< Data type of subtrahend tensor 
    bool aligned_minuend;         ///< Flag indicating if minuend data is 32-byte aligned 
    bool aligned_subtrahend;      ///< Flag indicating if subtrahend data is 32-byte aligned 
    size_t subtrahend_step;       ///< Step size in bytes for subtrahend data access 
} subinplace_ptask;

///@} [subinplace_ptask]



///@{ [mulinplace_ptask]

typedef struct {
    void* multiplicand_data;       ///< Pointer to multiplicand tensor data (mutable) 
    const void* multiplier_data;   ///< Pointer to multiplier tensor data (read-only) 
    size_t start;                  ///< Start index for this thread's chunk 
    size_t end;                    ///< End index for this thread's chunk 
    TensorType dtype_multiplicand; ///< Data type of multiplicand tensor 
    TensorType dtype_multiplier;   ///< Data type of multiplier tensor 
    bool aligned_multiplicand;     ///< Flag indicating if multiplicand data is 32-byte aligned 
    bool aligned_multiplier;       ///< Flag indicating if multiplier data is 32-byte aligned 
    size_t multiplier_step;        ///< Step size in bytes for multiplier data access 
} mulinplace_ptask;

///@} [mulinplace_ptask] 



///@{ [divinplace_ptask]

typedef struct {
    void* dividend_data;           ///< Pointer to dividend tensor data (mutable) 
    const void* divisor_data;      ///< Pointer to divisor tensor data (read-only) 
    size_t start;                  ///< Start index for this thread's chunk 
    size_t end;                    ///< End index for this thread's chunk 
    TensorType dtype_dividend;     ///< Data type of dividend tensor 
    TensorType dtype_divisor;      ///< Data type of divisor tensor 
    bool aligned_dividend;         ///< Flag indicating if dividend data is 32-byte aligned 
    bool aligned_divisor;          ///< Flag indicating if divisor data is 32-byte aligned 
    size_t divisor_step;           ///< Step size in bytes for divisor data access 
} divinplace_ptask;

///@} [divinplace_ptask]



///@{ [addincfinplace_ptask]

typedef struct {
    void* tensor_data;          ///< Pointer to tensor data (mutable) 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    TensorType dtype;           ///< Data type of tensor 
    bool aligned;               ///< Flag indicating if tensor data is 32-byte aligned 
    union {
        double float64_inc;     ///< Scalar increment for FLOAT64 
        float float32_inc;      ///< Scalar increment for FLOAT32 
        int32_t int32_inc;      ///< Scalar increment for INT32 
    } increment;                ///< Scalar increment value 
} addincfinplace_ptask;

///@} [addincfinplace_ptask]



///@{ [subdecfinplace_ptask]

typedef struct {
    void* tensor_data;          ///< Pointer to tensor data (mutable) 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    TensorType dtype;           ///< Data type of tensor 
    bool aligned;               ///< Flag indicating if tensor data is 32-byte aligned 
    union {
        double float64_dec;     ///< Scalar decrement for FLOAT64 
        float float32_dec;      ///< Scalar decrement for FLOAT32 
        int32_t int32_dec;      ///< Scalar decrement for INT32 
    } decrement;                ///< Scalar decrement value 
} subdecfinplace_ptask;
	
///@} [subdecfinplace_ptask]



///@{ [mulmulfinplace_ptask]

typedef struct {
    void* tensor_data;          ///< Pointer to tensor data (mutable) 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    TensorType dtype;           ///< Data type of tensor 
    bool aligned;               ///< Flag indicating if tensor data is 32-byte aligned 
    union {
        double float64_mult;    ///< Scalar multiplier for FLOAT64 
        float float32_mult;     ///< Scalar multiplier for FLOAT32 
        int32_t int32_mult;     ///< Scalar multiplier for INT32 
    } multiplier;               ///< Scalar multiplier value 
} mulmulfinplace_ptask;

///@} [mulmulfinplace_ptask]



///@{ [divdivfinplace_ptask]

typedef struct {
    void* tensor_data;          ///< Pointer to tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    TensorType dtype;           ///< Data type of tensor
    bool aligned;               ///< Flag indicating if tensor data is 32-byte aligned
    union {
        double float64_div;     ///< Scalar divisor for FLOAT64
        float float32_div;      ///< Scalar divisor for FLOAT32
        int32_t int32_div;      ///< Scalar divisor for INT32
    } divisor;                  ///< Scalar divisor value
} divdivfinplace_ptask;

///@} [divdivfinplace_ptask]



///@{ [addincf_non_inplace_ptask]

typedef struct {
    const void* tensor_data;    ///< Pointer to original tensor data (read-only)
    void* result_data;          ///< Pointer to result tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    TensorType dtype;           ///< Data type of tensor
    bool aligned_tensor;        ///< Flag indicating if tensor data is 32-byte aligned
    bool aligned_result;        ///< Flag indicating if result data is 32-byte aligned
    union {
        double float64_inc;     ///< Scalar increment for FLOAT64
        float float32_inc;      ///< Scalar increment for FLOAT32
        int32_t int32_inc;      ///< Scalar increment for INT32
    } increment;                ///< Scalar increment value
} addincf_non_inplace_ptask;

///@} [addincf_non_inplace_ptask]



///@{ [subdecf_non_inplace_ptask]

typedef struct {
    const void* tensor_data;    ///< Pointer to original tensor data (read-only)
    void* result_data;          ///< Pointer to result tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    TensorType dtype;           ///< Data type of tensor
    bool aligned_tensor;        ///< Flag indicating if tensor data is 32-byte aligned
    bool aligned_result;        ///< Flag indicating if result data is 32-byte aligned
    union {
        double float64_dec;     ///< Scalar decrement for FLOAT64
        float float32_dec;      ///< Scalar decrement for FLOAT32
        int32_t int32_dec;      ///< Scalar decrement for INT32
    } decrement;                ///< Scalar decrement value
} subdecf_non_inplace_ptask;

///@} [subdecf_non_inplace_ptask]



///@{ [subdecf_non_inplace_ptask]

typedef struct {
    const void* tensor_data;    ///< Pointer to original tensor data (read-only)
    void* result_data;          ///< Pointer to result tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    TensorType dtype;           ///< Data type of tensor
    bool aligned_tensor;        ///< Flag indicating if tensor data is 32-byte aligned
    bool aligned_result;        ///< Flag indicating if result data is 32-byte aligned
    union {
        double float64_mult;    ///< Scalar multiplier for FLOAT64
        float float32_mult;     ///< Scalar multiplier for FLOAT32
        int32_t int32_mult;     ///< Scalar multiplier for INT32
    } multiplier;               ///< Scalar multiplier value
} mulmulf_non_inplace_ptask;

///@} [mulmulf_non_inplace_ptask]



///@{ [divdivf_non_inplace_ptask]

typedef struct {
    const void* tensor_data;    ///< Pointer to original tensor data (read-only)
    void* result_data;          ///< Pointer to result tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    TensorType dtype;           ///< Data type of tensor
    bool aligned_tensor;        ///< Flag indicating if tensor data is 32-byte aligned
    bool aligned_result;        ///< Flag indicating if result data is 32-byte aligned
    union {
        double float64_div;     ///< Scalar divisor for FLOAT64
        float float32_div;      ///< Scalar divisor for FLOAT32
        int32_t int32_div;      ///< Scalar divisor for INT32
    } divisor;                  ///< Scalar divisor value
} divdivf_non_inplace_ptask;

///@} [divdivf_non_inplace_ptask]



///@{ [addbroadcasting_inplace_ptask]

typedef struct {
    TensorType dtype;           ///< Data type of both tensors (must match for optimized path)
    bool aligned_summand;       ///< Flag indicating if summand data is 32-byte aligned
    bool aligned_sumend;        ///< Flag indicating if sumend data is 32-byte aligned  
    void* summand_data;         ///< Pointer to summand tensor data (mutable, modified in-place)
    const void* sumend_data;    ///< Pointer to sumend tensor data (read-only, broadcasted)
    size_t start;               ///< Start block index for this thread's chunk
    size_t end;                 ///< End block index for this thread's chunk
    size_t numel_sumend;        ///< Number of elements in sumend tensor
    size_t broadcast_ratio;     ///< Broadcast ratio (numel_summand / numel_sumend)
} addbroadcasting_inplace_ptask;

///@} [addbroadcasting_inplace_ptask]



///@{ [subbroadcasting_inplace_ptask]

typedef struct {
    TensorType dtype;            ///< Data type of both tensors (must match for optimized path)
    bool aligned_minuend;        ///< Flag indicating if minuend data is 32-byte aligned
    bool aligned_subtrahend;     ///< Flag indicating if subtrahend data is 32-byte aligned  
    void* minuend_data;          ///< Pointer to minuend tensor data (mutable, modified in-place)
    const void* subtrahend_data; ///< Pointer to subtrahend tensor data (read-only, broadcasted)
    size_t start;                ///< Start block index for this thread's chunk
    size_t end;                  ///< End block index for this thread's chunk
    size_t numel_subtrahend;     ///< Number of elements in subtrahend tensor
    size_t broadcast_ratio;      ///< Broadcast ratio (numel_minuend / numel_subtrahend)
} subbroadcasting_inplace_ptask;

///@} [subbroadcasting_inplace_ptask]



///@{ [mulbroadcasting_inplace_ptask]

typedef struct {
    TensorType dtype;               ///< Data type of both tensors (must match for optimized path)
    bool aligned_multiplicand;      ///< Flag indicating if multiplicand data is 32-byte aligned
    bool aligned_multiplier;        ///< Flag indicating if multiplier data is 32-byte aligned  
    void* multiplicand_data;        ///< Pointer to multiplicand tensor data (mutable, modified in-place)
    const void* multiplier_data;    ///< Pointer to multiplier tensor data (read-only, broadcasted)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_multiplier;        ///< Number of elements in multiplier tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_multiplicand / numel_multiplier)
} mulbroadcasting_inplace_ptask;

///@} [mulbroadcasting_inplace_ptask]



///@{ [divbroadcasting_inplace_ptask]

typedef struct {
    TensorType dtype;               ///< Data type of both tensors (must match for optimized path)
    bool aligned_dividend;          ///< Flag indicating if dividend data is 32-byte aligned
    bool aligned_divisor;           ///< Flag indicating if divisor data is 32-byte aligned  
    void* dividend_data;            ///< Pointer to dividend tensor data (mutable, modified in-place)
    const void* divisor_data;       ///< Pointer to divisor tensor data (read-only, broadcasted)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_divisor;           ///< Number of elements in divisor tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_dividend / numel_divisor)
} divbroadcasting_inplace_ptask;

///@} [divbroadcasting_inplace_ptask]



///@{ [addbroadcasting_ptask]

typedef struct {
    TensorType dtype;               ///< Data type of result tensor
    bool aligned_summand;           ///< Flag indicating if summand data is 32-byte aligned
    bool aligned_sumend;            ///< Flag indicating if sumend data is 32-byte aligned  
    bool aligned_result;            ///< Flag indicating if result data is 32-byte aligned
    const void* summand_data;       ///< Pointer to summand tensor data (read-only)
    const void* sumend_data;        ///< Pointer to sumend tensor data (read-only, broadcasted)
    void* result_data;              ///< Pointer to result tensor data (mutable)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_sumend;            ///< Number of elements in sumend tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_summand / numel_sumend)
} addbroadcasting_ptask;

///@} [addbroadcasting_ptask]



///@{ [mulbroadcasting_ptask]

typedef struct {
    TensorType dtype;               ///< Data type of result tensor
    bool aligned_multiplicand;      ///< Flag indicating if multiplicand data is 32-byte aligned
    bool aligned_multiplier;        ///< Flag indicating if multiplier data is 32-byte aligned  
    bool aligned_result;            ///< Flag indicating if result data is 32-byte aligned
    const void* multiplicand_data;  ///< Pointer to multiplicand tensor data (read-only)
    const void* multiplier_data;    ///< Pointer to multiplier tensor data (read-only, broadcasted)
    void* result_data;              ///< Pointer to result tensor data (mutable)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_multiplier;        ///< Number of elements in multiplier tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_multiplicand / numel_multiplier)
} mulbroadcasting_ptask;

///@} [mulbroadcasting_ptask]



///@{ [divbroadcasting_ptask]

typedef struct {
    TensorType dtype;               ///< Data type of result tensor
    bool aligned_dividend;          ///< Flag indicating if dividend data is 32-byte aligned
    bool aligned_divisor;           ///< Flag indicating if divisor data is 32-byte aligned  
    bool aligned_result;            ///< Flag indicating if result data is 32-byte aligned
    const void* dividend_data;      ///< Pointer to dividend tensor data (read-only)
    const void* divisor_data;       ///< Pointer to divisor tensor data (read-only, broadcasted)
    void* result_data;              ///< Pointer to result tensor data (mutable)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_divisor;           ///< Number of elements in divisor tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_dividend / numel_divisor)
} divbroadcasting_ptask;

///@} [divbroadcasting_ptask]



///@{ [macro]

/** @def
 * NNL2_MIN_ELEMS_PER_THREAD
 *
 ** @brief 
 * Minimum number of elements per thread for efficient parallelization
 *
 ** @details 
 * Threads are only used when there are at least this many elements per thread
 */
#define NNL2_MIN_ELEMS_PER_THREAD 1000

/** @def 
 * NNL2_MIN_THREADS  
 *
 ** @brief 
 * Minimum number of threads to use
 */
#define NNL2_MIN_THREADS 1

///@} [macro]

#endif /** NNL2_PARALLEL_BACKEND_H **/
