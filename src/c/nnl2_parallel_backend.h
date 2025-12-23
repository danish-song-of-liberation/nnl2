#ifndef NNL2_PARALLEL_BACKEND_H
#define NNL2_PARALLEL_BACKEND_H

/** @file nnl2_parallel_backend.h
 ** @brief I tried adding a pool, but it was slower
 **/

///@{ [single_arr_ptask]

typedef struct {
	void* data;    ///< Pointer to an array with nnl2_tensor data
	size_t start;  ///< Index for entering the data array
	size_t end;    ///< Index for the end of the data entry into the array
} single_arr_ptask;  

///@} [single_arr_ptask]



///@{ [leaky_relu_single_arr_ptask]

typedef struct {
    void* data;	   ///< Pointer to an array with nnl2_tensor data
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
    nnl2_tensor_type dtype;  ///< Data type of elements
    bool aligned;      ///< Whether memory is aligned
} fill_ptask;

///@} [fill_ptask]



///@{ [add_ptask]

typedef struct {
    const void* summand_data;  ///< Pointer to summand nnl2_tensor data 
    const void* addend_data;   ///< Pointer to addend nnl2_tensor data 
    void* result_data;         ///< Pointer to result nnl2_tensor data
    size_t start;              ///< Start index for this thread 
    size_t end;                ///< End index for this thread 
    nnl2_tensor_type dtype_summand;  ///< Data type of summand nnl2_tensor 
    nnl2_tensor_type dtype_addend;   ///< Data type of addend nnl2_tensor 
    nnl2_tensor_type result_dtype;   ///< Data type of result nnl2_tensor 
} add_ptask;

///@} [add_ptask]



///@{ [sub_ptask]

typedef struct {
    const void* minuend_data;     ///< Pointer to minuend nnl2_tensor data 
    const void* subtrahend_data;  ///< Pointer to subtrahend nnl2_tensor data 
    void* result_data;            ///< Pointer to result nnl2_tensor data 
    size_t start;                 ///< Start index for this thread 
    size_t end;                   ///< End index for this thread 
    nnl2_tensor_type dtype_minuend;     ///< Data type of minuend nnl2_tensor 
    nnl2_tensor_type dtype_subtrahend;  ///< Data type of subtrahend nnl2_tensor 
    nnl2_tensor_type result_dtype;      ///< Data type of result nnl2_tensor 
} sub_ptask;

///@} [sub_ptask]



///@{ [mul_ptask]

typedef struct {
    const void* multiplicand_data; ///< Pointer to multiplicand nnl2_tensor data 
    const void* multiplier_data;   ///< Pointer to multiplier nnl2_tensor data 
    void* result_data;             ///< Pointer to result nnl2_tensor data 
    size_t start;                  ///< Start index for this thread 
    size_t end;                    ///< End index for this thread 
    nnl2_tensor_type dtype_multiplicand; ///< Data type of multiplicand nnl2_tensor 
    nnl2_tensor_type dtype_multiplier;   ///< Data type of multiplier nnl2_tensor 
    nnl2_tensor_type result_dtype;       ///< Data type of result nnl2_tensor 
} mul_ptask;

///@} [mul_ptask]



///@{ [div_ptask]

typedef struct {
    const void* dividend_data;    ///< Pointer to dividend nnl2_tensor data 
    const void* divisor_data;     ///< Pointer to divisor nnl2_tensor data 
    void* result_data;            ///< Pointer to result nnl2_tensor data
    size_t start;                 ///< Start index for this thread 
    size_t end;                   ///< End index for this thread 
    nnl2_tensor_type dtype_dividend;    ///< Data type of dividend nnl2_tensor 
    nnl2_tensor_type dtype_divisor;     ///< Data type of divisor nnl2_tensor 
    nnl2_tensor_type result_dtype;      ///< Data type of result nnl2_tensor 
} div_ptask;

///@} [div_ptask]



///@{ [abs_ptask]

typedef struct {
    const void* input_data;      ///< Pointer to input nnl2_tensor data 
    void* result_data;           ///< Pointer to result nnl2_tensor data 
    size_t start;                ///< Start index for this thread 
    size_t end;                  ///< End index for this thread 
    nnl2_tensor_type dtype;            ///< Data type of nnl2_tensor 
} abs_ptask;

///@} [abs_ptask]



///@{ [abs_inplace_ptask]

/** @brief
 * Task structure for parallel in-place absolute value operation
 */
typedef struct {
    void* data;                 ///< Pointer to nnl2_tensor data 
    size_t start;               ///< Start index for this thread 
    size_t end;                 ///< End index for this thread 
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor 
} abs_inplace_ptask;

///@} [abs_inplace_ptask]



///@{ [addinplace_ptask]
	
typedef struct {
    void* summand_data;           ///< Pointer to summand nnl2_tensor data (mutable) 
    const void* addend_data;      ///< Pointer to addend nnl2_tensor data (read-only) 
    size_t start;                 ///< Start index for this thread's chunk 
    size_t end;                   ///< End index for this thread's chunk 
    nnl2_tensor_type dtype_summand;     ///< Data type of summand nnl2_tensor 
    nnl2_tensor_type dtype_addend;      ///< Data type of addend nnl2_tensor 
    bool aligned_summand;         ///< Flag indicating if summand data is 32-byte aligned 
    bool aligned_addend;          ///< Flag indicating if addend data is 32-byte aligned 
    size_t addend_step;           ///< Step size in bytes for addend data access 
} addinplace_ptask;

///@} [addinplace_ptask]



///@{ [subinplace_ptask]

typedef struct {
    void* minuend_data;           ///< Pointer to minuend nnl2_tensor data (mutable) 
    const void* subtrahend_data;  ///< Pointer to subtrahend nnl2_tensor data (read-only) 
    size_t start;                 ///< Start index for this thread's chunk 
    size_t end;                   ///< End index for this thread's chunk 
    nnl2_tensor_type dtype_minuend;     ///< Data type of minuend nnl2_tensor 
    nnl2_tensor_type dtype_subtrahend;  ///< Data type of subtrahend nnl2_tensor 
    bool aligned_minuend;         ///< Flag indicating if minuend data is 32-byte aligned 
    bool aligned_subtrahend;      ///< Flag indicating if subtrahend data is 32-byte aligned 
    size_t subtrahend_step;       ///< Step size in bytes for subtrahend data access 
} subinplace_ptask;

///@} [subinplace_ptask]



///@{ [mulinplace_ptask]

typedef struct {
    void* multiplicand_data;       ///< Pointer to multiplicand nnl2_tensor data (mutable) 
    const void* multiplier_data;   ///< Pointer to multiplier nnl2_tensor data (read-only) 
    size_t start;                  ///< Start index for this thread's chunk 
    size_t end;                    ///< End index for this thread's chunk 
    nnl2_tensor_type dtype_multiplicand; ///< Data type of multiplicand nnl2_tensor 
    nnl2_tensor_type dtype_multiplier;   ///< Data type of multiplier nnl2_tensor 
    bool aligned_multiplicand;     ///< Flag indicating if multiplicand data is 32-byte aligned 
    bool aligned_multiplier;       ///< Flag indicating if multiplier data is 32-byte aligned 
    size_t multiplier_step;        ///< Step size in bytes for multiplier data access 
} mulinplace_ptask;

///@} [mulinplace_ptask] 



///@{ [divinplace_ptask]

typedef struct {
    void* dividend_data;           ///< Pointer to dividend nnl2_tensor data (mutable) 
    const void* divisor_data;      ///< Pointer to divisor nnl2_tensor data (read-only) 
    size_t start;                  ///< Start index for this thread's chunk 
    size_t end;                    ///< End index for this thread's chunk 
    nnl2_tensor_type dtype_dividend;     ///< Data type of dividend nnl2_tensor 
    nnl2_tensor_type dtype_divisor;      ///< Data type of divisor nnl2_tensor 
    bool aligned_dividend;         ///< Flag indicating if dividend data is 32-byte aligned 
    bool aligned_divisor;          ///< Flag indicating if divisor data is 32-byte aligned 
    size_t divisor_step;           ///< Step size in bytes for divisor data access 
} divinplace_ptask;

///@} [divinplace_ptask]



///@{ [addincfinplace_ptask]

typedef struct {
    void* tensor_data;          ///< Pointer to nnl2_tensor data (mutable) 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor 
    bool aligned;               ///< Flag indicating if nnl2_tensor data is 32-byte aligned 
    union {
        nnl2_float64 float64_inc;    ///< Scalar increment for FLOAT64 
        nnl2_float32 float32_inc;    ///< Scalar increment for FLOAT32 
        nnl2_int32   int32_inc;      ///< Scalar increment for INT32 
    } increment;                ///< Scalar increment value 
} addincfinplace_ptask;

///@} [addincfinplace_ptask]



///@{ [subdecfinplace_ptask]

typedef struct {
    void* tensor_data;          ///< Pointer to nnl2_tensor data (mutable) 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor 
    bool aligned;               ///< Flag indicating if nnl2_tensor data is 32-byte aligned 
    union {
        nnl2_float64 float64_dec;     ///< Scalar decrement for FLOAT64 
        nnl2_float32 float32_dec;     ///< Scalar decrement for FLOAT32 
        nnl2_int32   int32_dec;       ///< Scalar decrement for INT32 
    } decrement;                ///< Scalar decrement value 
} subdecfinplace_ptask;
	
///@} [subdecfinplace_ptask]



///@{ [mulmulfinplace_ptask]

typedef struct {
    void* tensor_data;          ///< Pointer to nnl2_tensor data (mutable) 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor 
    bool aligned;               ///< Flag indicating if nnl2_tensor data is 32-byte aligned 
    union {
        nnl2_float64 float64_mult;    ///< Scalar multiplier for FLOAT64 
        nnl2_float32 float32_mult;    ///< Scalar multiplier for FLOAT32 
        nnl2_int32   int32_mult;      ///< Scalar multiplier for INT32 
    } multiplier;               ///< Scalar multiplier value 
} mulmulfinplace_ptask;

///@} [mulmulfinplace_ptask]



///@{ [divdivfinplace_ptask]

typedef struct {
    void* tensor_data;          ///< Pointer to nnl2_tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned;               ///< Flag indicating if nnl2_tensor data is 32-byte aligned
    union {
        nnl2_float64 float64_div;     ///< Scalar divisor for FLOAT64
        nnl2_float32 float32_div;     ///< Scalar divisor for FLOAT32
        nnl2_int32   int32_div;       ///< Scalar divisor for INT32
    } divisor;                  ///< Scalar divisor value
} divdivfinplace_ptask;

///@} [divdivfinplace_ptask]



///@{ [addincf_non_inplace_ptask]

typedef struct {
    const void* tensor_data;    ///< Pointer to original nnl2_tensor data (read-only)
    void* result_data;          ///< Pointer to result nnl2_tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned_tensor;        ///< Flag indicating if nnl2_tensor data is 32-byte aligned
    bool aligned_result;        ///< Flag indicating if result data is 32-byte aligned
    union {
        nnl2_float64 float64_inc;     ///< Scalar increment for FLOAT64
        nnl2_float32 float32_inc;     ///< Scalar increment for FLOAT32
        nnl2_int32   int32_inc;       ///< Scalar increment for INT32
    } increment;                ///< Scalar increment value
} addincf_non_inplace_ptask;

///@} [addincf_non_inplace_ptask]



///@{ [subdecf_non_inplace_ptask]

typedef struct {
    const void* tensor_data;    ///< Pointer to original nnl2_tensor data (read-only)
    void* result_data;          ///< Pointer to result nnl2_tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned_tensor;        ///< Flag indicating if nnl2_tensor data is 32-byte aligned
    bool aligned_result;        ///< Flag indicating if result data is 32-byte aligned
    union {
        nnl2_float64 float64_dec;     ///< Scalar decrement for FLOAT64
        nnl2_float32 float32_dec;     ///< Scalar decrement for FLOAT32
        nnl2_int32 int32_dec;         ///< Scalar decrement for INT32
    } decrement;                ///< Scalar decrement value
} subdecf_non_inplace_ptask;

///@} [subdecf_non_inplace_ptask]



///@{ [subdecf_non_inplace_ptask]

typedef struct {
    const void* tensor_data;    ///< Pointer to original nnl2_tensor data (read-only)
    void* result_data;          ///< Pointer to result nnl2_tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned_tensor;        ///< Flag indicating if nnl2_tensor data is 32-byte aligned
    bool aligned_result;        ///< Flag indicating if result data is 32-byte aligned
    union {
        nnl2_float64 float64_mult;    ///< Scalar multiplier for FLOAT64
        nnl2_float32 float32_mult;    ///< Scalar multiplier for FLOAT32
        nnl2_int32 int32_mult;        ///< Scalar multiplier for INT32
    } multiplier;               ///< Scalar multiplier value
} mulmulf_non_inplace_ptask;

///@} [mulmulf_non_inplace_ptask]



///@{ [divdivf_non_inplace_ptask]

typedef struct {
    const void* tensor_data;    ///< Pointer to original nnl2_tensor data (read-only)
    void* result_data;          ///< Pointer to result nnl2_tensor data (mutable)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned_tensor;        ///< Flag indicating if nnl2_tensor data is 32-byte aligned
    bool aligned_result;        ///< Flag indicating if result data is 32-byte aligned
    union {
        nnl2_float64 float64_div;     ///< Scalar divisor for FLOAT64
        nnl2_float32 float32_div;     ///< Scalar divisor for FLOAT32
        nnl2_int32 int32_div;         ///< Scalar divisor for INT32
    } divisor;                  ///< Scalar divisor value
} divdivf_non_inplace_ptask;

///@} [divdivf_non_inplace_ptask]



///@{ [addbroadcasting_inplace_ptask]

typedef struct {
    nnl2_tensor_type dtype;           ///< Data type of both tensors (must match for optimized path)
    bool aligned_summand;       ///< Flag indicating if summand data is 32-byte aligned
    bool aligned_sumend;        ///< Flag indicating if sumend data is 32-byte aligned  
    void* summand_data;         ///< Pointer to summand nnl2_tensor data (mutable, modified in-place)
    const void* sumend_data;    ///< Pointer to sumend nnl2_tensor data (read-only, broadcasted)
    size_t start;               ///< Start block index for this thread's chunk
    size_t end;                 ///< End block index for this thread's chunk
    size_t numel_sumend;        ///< Number of elements in sumend nnl2_tensor
    size_t broadcast_ratio;     ///< Broadcast ratio (numel_summand / numel_sumend)
} addbroadcasting_inplace_ptask;

///@} [addbroadcasting_inplace_ptask]



///@{ [subbroadcasting_inplace_ptask]

typedef struct {
    nnl2_tensor_type dtype;            ///< Data type of both tensors (must match for optimized path)
    bool aligned_minuend;        ///< Flag indicating if minuend data is 32-byte aligned
    bool aligned_subtrahend;     ///< Flag indicating if subtrahend data is 32-byte aligned  
    void* minuend_data;          ///< Pointer to minuend nnl2_tensor data (mutable, modified in-place)
    const void* subtrahend_data; ///< Pointer to subtrahend nnl2_tensor data (read-only, broadcasted)
    size_t start;                ///< Start block index for this thread's chunk
    size_t end;                  ///< End block index for this thread's chunk
    size_t numel_subtrahend;     ///< Number of elements in subtrahend nnl2_tensor
    size_t broadcast_ratio;      ///< Broadcast ratio (numel_minuend / numel_subtrahend)
} subbroadcasting_inplace_ptask;

///@} [subbroadcasting_inplace_ptask]



///@{ [mulbroadcasting_inplace_ptask]

typedef struct {
    nnl2_tensor_type dtype;               ///< Data type of both tensors (must match for optimized path)
    bool aligned_multiplicand;      ///< Flag indicating if multiplicand data is 32-byte aligned
    bool aligned_multiplier;        ///< Flag indicating if multiplier data is 32-byte aligned  
    void* multiplicand_data;        ///< Pointer to multiplicand nnl2_tensor data (mutable, modified in-place)
    const void* multiplier_data;    ///< Pointer to multiplier nnl2_tensor data (read-only, broadcasted)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_multiplier;        ///< Number of elements in multiplier nnl2_tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_multiplicand / numel_multiplier)
} mulbroadcasting_inplace_ptask;

///@} [mulbroadcasting_inplace_ptask]



///@{ [divbroadcasting_inplace_ptask]

typedef struct {
    nnl2_tensor_type dtype;               ///< Data type of both tensors (must match for optimized path)
    bool aligned_dividend;          ///< Flag indicating if dividend data is 32-byte aligned
    bool aligned_divisor;           ///< Flag indicating if divisor data is 32-byte aligned  
    void* dividend_data;            ///< Pointer to dividend nnl2_tensor data (mutable, modified in-place)
    const void* divisor_data;       ///< Pointer to divisor nnl2_tensor data (read-only, broadcasted)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_divisor;           ///< Number of elements in divisor nnl2_tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_dividend / numel_divisor)
} divbroadcasting_inplace_ptask;

///@} [divbroadcasting_inplace_ptask]



///@{ [addbroadcasting_ptask]

typedef struct {
    nnl2_tensor_type dtype;               ///< Data type of result nnl2_tensor
    bool aligned_summand;           ///< Flag indicating if summand data is 32-byte aligned
    bool aligned_sumend;            ///< Flag indicating if sumend data is 32-byte aligned  
    bool aligned_result;            ///< Flag indicating if result data is 32-byte aligned
    const void* summand_data;       ///< Pointer to summand nnl2_tensor data (read-only)
    const void* sumend_data;        ///< Pointer to sumend nnl2_tensor data (read-only, broadcasted)
    void* result_data;              ///< Pointer to result nnl2_tensor data (mutable)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_sumend;            ///< Number of elements in sumend nnl2_tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_summand / numel_sumend)
} addbroadcasting_ptask;

///@} [addbroadcasting_ptask]



///@{ [mulbroadcasting_ptask]

typedef struct {
    nnl2_tensor_type dtype;               ///< Data type of result nnl2_tensor
    bool aligned_multiplicand;      ///< Flag indicating if multiplicand data is 32-byte aligned
    bool aligned_multiplier;        ///< Flag indicating if multiplier data is 32-byte aligned  
    bool aligned_result;            ///< Flag indicating if result data is 32-byte aligned
    const void* multiplicand_data;  ///< Pointer to multiplicand nnl2_tensor data (read-only)
    const void* multiplier_data;    ///< Pointer to multiplier nnl2_tensor data (read-only, broadcasted)
    void* result_data;              ///< Pointer to result nnl2_tensor data (mutable)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_multiplier;        ///< Number of elements in multiplier nnl2_tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_multiplicand / numel_multiplier)
} mulbroadcasting_ptask;

///@} [mulbroadcasting_ptask]



///@{ [divbroadcasting_ptask]

typedef struct {
    nnl2_tensor_type dtype;               ///< Data type of result nnl2_tensor
    bool aligned_dividend;          ///< Flag indicating if dividend data is 32-byte aligned
    bool aligned_divisor;           ///< Flag indicating if divisor data is 32-byte aligned  
    bool aligned_result;            ///< Flag indicating if result data is 32-byte aligned
    const void* dividend_data;      ///< Pointer to dividend nnl2_tensor data (read-only)
    const void* divisor_data;       ///< Pointer to divisor nnl2_tensor data (read-only, broadcasted)
    void* result_data;              ///< Pointer to result nnl2_tensor data (mutable)
    size_t start;                   ///< Start block index for this thread's chunk
    size_t end;                     ///< End block index for this thread's chunk
    size_t numel_divisor;           ///< Number of elements in divisor nnl2_tensor
    size_t broadcast_ratio;         ///< Broadcast ratio (numel_dividend / numel_divisor)
} divbroadcasting_ptask;

///@} [divbroadcasting_ptask]



///@{ [max_ptask]

typedef struct {
    const nnl2_tensor* tensora;      ///< Pointer to first input nnl2_tensor 
    const nnl2_tensor* tensorb;      ///< Pointer to second input nnl2_tensor 
    nnl2_tensor* result;             ///< Pointer to output nnl2_tensor 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    nnl2_tensor_type dtype_a;         ///< Data type of first nnl2_tensor 
    nnl2_tensor_type dtype_b;         ///< Data type of second nnl2_tensor 
    nnl2_tensor_type result_dtype;    ///< Data type of result nnl2_tensor 
    bool aligned;               ///< Whether memory is properly aligned 
} max_ptask;

///@} [max_ptask]



///@{ [min_ptask]

typedef struct {
    const nnl2_tensor* tensora;      ///< Pointer to first input nnl2_tensor 
    const nnl2_tensor* tensorb;      ///< Pointer to second input nnl2_tensor 
    nnl2_tensor* result;             ///< Pointer to output nnl2_tensor 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    nnl2_tensor_type dtype_a;         ///< Data type of first nnl2_tensor 
    nnl2_tensor_type dtype_b;         ///< Data type of second nnl2_tensor 
    nnl2_tensor_type result_dtype;    ///< Data type of result nnl2_tensor 
    bool aligned;               ///< Whether memory is properly aligned 
} min_ptask;

///@} [min_ptask]



///@{ [maxinplace_ptask]

typedef struct {
    nnl2_tensor* tensora;            ///< Pointer to first input nnl2_tensor (modified in-place)
    const nnl2_tensor* tensorb;      ///< Pointer to second input nnl2_tensor 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    nnl2_tensor_type dtype_a;         ///< Data type of first nnl2_tensor 
    nnl2_tensor_type dtype_b;         ///< Data type of second nnl2_tensor 
    bool aligned;               ///< Whether memory is properly aligned 
} maxinplace_ptask;

///@} [maxinplace_ptask]



///@{ [mininplace_ptask]

typedef struct {
    nnl2_tensor* tensora;            ///< Pointer to first input nnl2_tensor (modified in-place)
    const nnl2_tensor* tensorb;      ///< Pointer to second input nnl2_tensor 
    size_t start;               ///< Start index for this thread's chunk 
    size_t end;                 ///< End index for this thread's chunk 
    nnl2_tensor_type dtype_a;         ///< Data type of first nnl2_tensor 
    nnl2_tensor_type dtype_b;         ///< Data type of second nnl2_tensor 
    bool aligned;               ///< Whether memory is properly aligned 
} mininplace_ptask;

///@} [mininplace_ptask]



///@{ [max_broadcasting_inplace_ptask]

typedef struct {
    nnl2_tensor* x;                  ///< Pointer to first input nnl2_tensor (modified in-place)
    const nnl2_tensor* y;            ///< Pointer to second input nnl2_tensor 
    size_t start_block;         ///< Start block index for this thread's chunk
    size_t end_block;           ///< End block index for this thread's chunk
    size_t block_size;          ///< Size of each broadcast block (numel_y)
    nnl2_tensor_type x_dtype;         ///< Data type of first nnl2_tensor 
    nnl2_tensor_type y_dtype;         ///< Data type of second nnl2_tensor 
    bool aligned;               ///< Whether memory is properly aligned 
} max_broadcasting_inplace_ptask;

///@} [max_broadcasting_inplace_ptask]



///@{ [min_broadcasting_inplace_ptask]

typedef struct {
    nnl2_tensor* x;                  ///< Pointer to first input nnl2_tensor (modified in-place)
    const nnl2_tensor* y;            ///< Pointer to second input nnl2_tensor 
    size_t start_block;         ///< Start block index for this thread's chunk
    size_t end_block;           ///< End block index for this thread's chunk
    size_t block_size;          ///< Size of each broadcast block (numel_y)
    nnl2_tensor_type x_dtype;         ///< Data type of first nnl2_tensor 
    nnl2_tensor_type y_dtype;         ///< Data type of second nnl2_tensor 
    bool aligned;               ///< Whether memory is properly aligned 
} min_broadcasting_inplace_ptask;

///@} [min_broadcasting_inplace_ptask]



///@{ [max_broadcasting_ptask]

typedef struct {
    const nnl2_tensor* x;            ///< Pointer to first input nnl2_tensor
    const nnl2_tensor* y;            ///< Pointer to second input nnl2_tensor 
    nnl2_tensor* result;             ///< Pointer to output nnl2_tensor
    size_t start_block;         ///< Start block index for this thread's chunk
    size_t end_block;           ///< End block index for this thread's chunk
    size_t block_size;          ///< Size of each broadcast block (numel_y)
    nnl2_tensor_type x_dtype;         ///< Data type of first nnl2_tensor 
    nnl2_tensor_type y_dtype;         ///< Data type of second nnl2_tensor 
    nnl2_tensor_type result_dtype;    ///< Data type of result nnl2_tensor
    bool aligned;               ///< Whether memory is properly aligned 
} max_broadcasting_ptask;

///@} [max_broadcasting_ptask]



///@{ [min_broadcasting_ptask]

typedef struct {
    const nnl2_tensor* x;            ///< Pointer to first input nnl2_tensor
    const nnl2_tensor* y;            ///< Pointer to second input nnl2_tensor 
    nnl2_tensor* result;             ///< Pointer to output nnl2_tensor
    size_t start_block;         ///< Start block index for this thread's chunk
    size_t end_block;           ///< End block index for this thread's chunk
    size_t block_size;          ///< Size of each broadcast block (numel_y)
    nnl2_tensor_type x_dtype;         ///< Data type of first nnl2_tensor 
    nnl2_tensor_type y_dtype;         ///< Data type of second nnl2_tensor 
    nnl2_tensor_type result_dtype;    ///< Data type of result nnl2_tensor
    bool aligned;               ///< Whether memory is properly aligned 
} min_broadcasting_ptask;

///@} [min_broadcasting_ptask]



///@{ [min_minf_ptask]

typedef struct {
    const nnl2_tensor* tensor;       ///< Pointer to input nnl2_tensor
    nnl2_tensor* result;             ///< Pointer to output nnl2_tensor
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned;               ///< Whether memory is properly aligned
    union {
        nnl2_float64 float64_threshold;   ///< Threshold value for FLOAT64
        nnl2_float32 float32_threshold;   ///< Threshold value for FLOAT32
        nnl2_int32   int32_threshold;     ///< Threshold value for INT32
    } threshold;
} min_minf_ptask;

///@} [min_minf_ptask]



///@{ [max_maxf_ptask]

/** @brief
 * Task structure for parallel element-wise maximum operation
 * 
 ** @details
 * Contains all necessary parameters for worker threads to compute
 * element-wise maximum between nnl2_tensor elements and scalar value
 */
typedef struct {
    const nnl2_tensor* tensor;       ///< Pointer to input nnl2_tensor
    nnl2_tensor* result;             ///< Pointer to output nnl2_tensor
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned;               ///< Whether memory is properly aligned
    union {
        nnl2_float64 float64_threshold;   ///< Threshold value for FLOAT64
        nnl2_float32 float32_threshold;   ///< Threshold value for FLOAT32
        nnl2_int32   int32_threshold;     ///< Threshold value for INT32
    } threshold;
} max_maxf_ptask;

///@} [max_maxf_ptask]



///@{ [max_maxf_inplace_ptask]

typedef struct {
    nnl2_tensor* tensor;             ///< Pointer to input nnl2_tensor (modified in-place)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned;               ///< Whether memory is properly aligned
    union {
        nnl2_float64 float64_threshold;   ///< Threshold value for FLOAT64
        nnl2_float32 float32_threshold;   ///< Threshold value for FLOAT32
        nnl2_int32   int32_threshold;     ///< Threshold value for INT32
    } threshold;
} max_maxf_inplace_ptask;

///@} [max_maxf_inplace_ptask]



///@{ [min_minf_inplace_ptask]

typedef struct {
    nnl2_tensor* tensor;             ///< Pointer to input nnl2_tensor (modified in-place)
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype;           ///< Data type of nnl2_tensor
    bool aligned;               ///< Whether memory is properly aligned
    union {
        nnl2_float64 float64_threshold;   ///< Threshold value for FLOAT64
        nnl2_float32 float32_threshold;   ///< Threshold value for FLOAT32
        nnl2_int32   int32_threshold;     ///< Threshold value for INT32
    } threshold;
} min_minf_inplace_ptask;

///@} [min_minf_inplace_ptask]



///@{ [axpy_inplace_ptask]

typedef struct {
    nnl2_tensor* summand;            ///< Pointer to summand nnl2_tensor (modified in-place)
    const nnl2_tensor* sumend;       ///< Pointer to sumend nnl2_tensor 
    size_t start;               ///< Start index for this thread's chunk
    size_t end;                 ///< End index for this thread's chunk
    nnl2_tensor_type dtype_summand;   ///< Data type of summand nnl2_tensor 
    nnl2_tensor_type dtype_sumend;    ///< Data type of sumend nnl2_tensor 
    bool aligned;               ///< Whether memory is properly aligned 
    union {
        nnl2_float64 float64_alpha;   ///< Alpha value for FLOAT64
        nnl2_float32 float32_alpha;   ///< Alpha value for FLOAT32
        nnl2_int32   int32_alpha;     ///< Alpha value for INT32
    } alpha;
} axpy_inplace_ptask;

///@} [axpy_inplace_ptask]



///@{ [axpy_ptask]

typedef struct {
    const nnl2_tensor* summand;          ///< Pointer to summand nnl2_tensor
    const nnl2_tensor* sumend;           ///< Pointer to sumend nnl2_tensor 
    nnl2_tensor* result;                 ///< Pointer to output nnl2_tensor
    size_t start;                   ///< Start index for this thread's chunk
    size_t end;                     ///< End index for this thread's chunk
    nnl2_tensor_type dtype_summand;       ///< Data type of summand nnl2_tensor 
    nnl2_tensor_type dtype_sumend;        ///< Data type of sumend nnl2_tensor 
    nnl2_tensor_type result_dtype;        ///< Data type of result nnl2_tensor
    bool aligned;                   ///< Whether memory is properly aligned 
    union {
        nnl2_float64 float64_alpha;       ///< Alpha value for FLOAT64
        nnl2_float32 float32_alpha;       ///< Alpha value for FLOAT32
        nnl2_int32   int32_alpha;         ///< Alpha value for INT32
    } alpha;
} axpy_ptask;

///@} [axpy_ptask]



///@{ [axpf_ptask]

typedef struct {
    const nnl2_tensor* summand;          ///< Pointer to summand nnl2_tensor
    nnl2_tensor* result;                 ///< Pointer to output nnl2_tensor
    size_t start;                   ///< Start index for this thread's chunk
    size_t end;                     ///< End index for this thread's chunk
    nnl2_tensor_type dtype;               ///< Data type of nnl2_tensor
    bool aligned;                   ///< Whether memory is properly aligned 
    union {
        nnl2_float64 float64_sumend;      ///< Sumend value for FLOAT64
        nnl2_float32 float32_sumend;      ///< Sumend value for FLOAT32
        nnl2_int32   int32_sumend;        ///< Sumend value for INT32
    } sumend;
    union {
        nnl2_float64 float64_alpha;       ///< Alpha value for FLOAT64
        nnl2_float32 float32_alpha;       ///< Alpha value for FLOAT32
        nnl2_int32   int32_alpha;         ///< Alpha value for INT32
    } alpha;
} axpf_ptask;

///@} [axpf_ptask]



///@{ [axpy_broadcasting_inplace_ptask]

typedef struct {
    nnl2_tensor* summand;                ///< Pointer to summand nnl2_tensor (modified in-place)
    const nnl2_tensor* sumend;           ///< Pointer to sumend nnl2_tensor 
    size_t start_block;             ///< Start block index for this thread's chunk
    size_t end_block;               ///< End block index for this thread's chunk
    size_t block_size;              ///< Size of each broadcast block (numel_sumend)
    nnl2_tensor_type summand_dtype;       ///< Data type of summand nnl2_tensor 
    nnl2_tensor_type sumend_dtype;        ///< Data type of sumend nnl2_tensor 
    bool aligned;                   ///< Whether memory is properly aligned 
    union {
        nnl2_float64 float64_alpha;       ///< Alpha value for FLOAT64
        nnl2_float32 float32_alpha;       ///< Alpha value for FLOAT32
        nnl2_int32   int32_alpha;         ///< Alpha value for INT32
    } alpha;
} axpy_broadcasting_inplace_ptask;

///@} [axpy_broadcasting_inplace_ptask]



///@{ [axpy_broadcasting_ptask]

typedef struct {
    nnl2_tensor* summand;                ///< Pointer to summand nnl2_tensor 
    nnl2_tensor* sumend;                 ///< Pointer to sumend nnl2_tensor 
    nnl2_tensor* result;                 ///< Pointer to result nnl2_tensor 
    size_t start_block;             ///< Start block index for this thread's chunk
    size_t end_block;               ///< End block index for this thread's chunk
    size_t block_size;              ///< Size of each broadcast block (numel_sumend)
    nnl2_tensor_type summand_dtype;       ///< Data type of summand nnl2_tensor 
    nnl2_tensor_type sumend_dtype;        ///< Data type of sumend nnl2_tensor 
    nnl2_tensor_type result_dtype;        ///< Data type of result nnl2_tensor 
    bool aligned;                   ///< Whether memory is properly aligned 
    union {
        nnl2_float64 float64_alpha;       ///< Alpha value for FLOAT64
        nnl2_float32 float32_alpha;       ///< Alpha value for FLOAT32
        nnl2_int32 int32_alpha;           ///< Alpha value for INT32
    } alpha;
} axpy_broadcasting_ptask;

///@} [axpy_broadcasting_ptask]



///@{ [axpf_inplace_ptask]

typedef struct {
    nnl2_tensor* summand;                ///< Pointer to summand nnl2_tensor (modified in-place)
    void* sumend;                   ///< Pointer to scalar sumend value
    size_t start_index;             ///< Start index for this thread's chunk
    size_t end_index;               ///< End index for this thread's chunk
    nnl2_tensor_type summand_dtype;       ///< Data type of summand nnl2_tensor
    bool aligned;                   ///< Whether memory is properly aligned
    union {
        nnl2_float64 float64_alpha;       ///< Alpha value for FLOAT64
        nnl2_float32 float32_alpha;       ///< Alpha value for FLOAT32
        nnl2_int32 int32_alpha;           ///< Alpha value for INT32
    } alpha;
    union {
        nnl2_float64 float64_sumend;      ///< Sumend value for FLOAT64
        nnl2_float32 float32_sumend;      ///< Sumend value for FLOAT32
        nnl2_int32 int32_sumend;          ///< Sumend value for INT32
    } sumend_val;
} axpf_inplace_ptask;



///@} [axpf_inplace_ptask]



///@{ [copy_ptask]

typedef struct {
    void* src_data;           ///< Pointer to source data 
    void* dst_data;           ///< Pointer to destination data 
    size_t start;             ///< Start index for this thread 
    size_t end;               ///< End index for this thread 
    nnl2_tensor_type dtype;         ///< Data type of the nnl2_tensor 
    bool aligned;             ///< Whether memory is aligned 
    nnl2_tensor_type target_dtype;  ///< Target data type for conversion 
} copy_ptask;

///@} [copy_ptask]



///@{ [hstack_ptask]

typedef struct {
    void* src_a;            		  ///< Pointer to first source nnl2_tensor data 
    void* src_b;             		  ///< Pointer to second source nnl2_tensor data 
    void* dst;               		  ///< Pointer to destination data 
    size_t start_idx;        		  ///< Start index for this thread 
    size_t end_idx;          		  ///< End index for this thread 
    size_t elements_per_row_a;		  ///< Elements per row in nnl2_tensor A 
    size_t elements_per_row_b;        ///< Elements per row in nnl2_tensor B 
    size_t elements_per_row_result;   ///< Elements per row in result 
    nnl2_tensor_type type_a;      		  ///< Data type of first nnl2_tensor 
    nnl2_tensor_type type_b;       		  ///< Data type of second nnl2_tensor 
    nnl2_tensor_type result_type;  		  ///< Result data type 
    bool aligned;             		  ///< Whether memory is aligned 
    bool same_type;           		  ///< Whether both tensors have same type 
} hstack_ptask;

///@} [hstack_ptask]



///@{ [vstack_ptask]

typedef struct {
    void* src_a;              ///< Pointer to first source nnl2_tensor data 
    void* src_b;              ///< Pointer to second source nnl2_tensor data 
    void* dst;                ///< Pointer to destination data 
    size_t start_idx;         ///< Start index for this thread 
    size_t end_idx;           ///< End index for this thread 
    size_t size_a;            ///< Total elements in nnl2_tensor A 
    size_t size_b;            ///< Total elements in nnl2_tensor B 
    size_t row_size_a;        ///< Row size in bytes for nnl2_tensor A 
    size_t row_size_b;        ///< Row size in bytes for nnl2_tensor B 
    nnl2_tensor_type type_a;        ///< Data type of first nnl2_tensor 
    nnl2_tensor_type type_b;        ///< Data type of second nnl2_tensor 
    nnl2_tensor_type result_type;   ///< Result data type 
    bool aligned;             ///< Whether memory is aligned 
    bool same_type;           ///< Whether both tensors have same type 
    int case_type;            ///< VStack case type: 0=1D-1D, 1=2D-1D, 2=1D-2D, 3=ND-ND 
} vstack_ptask;

///@} [vstack_ptask]



///@{ [concat_ptask]

typedef struct {
    void* src_a;                 ///< Pointer to first source nnl2_tensor data 
    void* src_b;                 ///< Pointer to second source nnl2_tensor data 
    void* dst;                   ///< Pointer to destination data 
    size_t start_idx;            ///< Start index for this thread 
    size_t end_idx;              ///< End index for this thread 
    size_t total_elements;       ///< Total elements in result 
    size_t a_axis_size;          ///< Size of concatenation axis in nnl2_tensor A 
    size_t item_size;            ///< Size of each element in bytes 
    int rank;                    ///< Rank of tensors 
    int axis;                    ///< Concatenation axis 
    int* result_shape;           ///< Shape of result nnl2_tensor 
    nnl2_int32* result_strides;  ///< Strides of result nnl2_tensor (in elements) 
    nnl2_int32* a_strides;       ///< Strides of nnl2_tensor A (in elements) 
    nnl2_int32* b_strides;       ///< Strides of nnl2_tensor B (in elements) 
    nnl2_tensor_type type_a;           ///< Data type of first nnl2_tensor 
    nnl2_tensor_type type_b;           ///< Data type of second nnl2_tensor 
    nnl2_tensor_type result_type;      ///< Result data type 
    bool aligned;                ///< Whether memory is aligned 
    bool same_type;              ///< Whether both tensors have same type 
} concat_ptask;

///@} [concat_ptask]



///@{ [reshape_ptask]

typedef struct {
    void* src_data;           ///< Pointer to source data 
    void* dst_data;           ///< Pointer to destination data
    size_t start_idx;         ///< Start index for this thread 
    size_t end_idx;           ///< End index for this thread 
    size_t item_size;         ///< Size of each element in bytes 
    bool aligned;             ///< Whether memory is aligned 
} reshape_ptask;

///@} [reshape_ptask]



///@{ [sum_axis_ptask]

typedef struct {
    void* src_data;           ///< Pointer to source data 
    void* dst_data;           ///< Pointer to destination data 
    size_t start_idx;         ///< Start result index for this thread 
    size_t end_idx;           ///< End result index for this thread 
    size_t result_numel;      ///< Total elements in result 
    int axis;                 ///< Summation axis 
    int elements_along_axis;  ///< Elements along summation axis 
    nnl2_tensor_type dtype;         ///< Data type 
    nnl2_tensor* tensor;           ///< Source nnl2_tensor 
    nnl2_tensor* result;           ///< Result nnl2_tensor 
} sum_axis_ptask;

///@} [sum_axis_ptask]



///@{ [l2norm_ptask]

typedef struct {
    void* src_data;           ///< Pointer to source data 
    size_t start_idx;         ///< Start index for this thread 
    size_t end_idx;           ///< End index for this thread 
    nnl2_tensor_type dtype;         ///< Data type of the nnl2_tensor 
    bool aligned;             ///< Whether memory is aligned 
    union {
        nnl2_float64 float64_acc;
        nnl2_float32 float32_acc;
        nnl2_int32   int32_acc;
    } accumulator;            ///< Thread-local accumulator for squared values 
} l2norm_ptask;

///@} [l2norm_ptask]



///@{ [relu_ptask]

typedef struct {
    void* src_data;           ///< Pointer to source data 
    void* dst_data;           ///< Pointer to destination data 
    size_t start_idx;         ///< Start index for this thread 
    size_t end_idx;           ///< End index for this thread 
    nnl2_tensor_type dtype;         ///< Data type of the nnl2_tensor 
    bool aligned;             ///< Whether memory is aligned 
    bool inplace;             ///< Whether operation is in-place 
} relu_ptask;

///@} [relu_ptask]



///@{ [leakyrelu_ptask]

typedef struct {
    void* src_data;           ///< Pointer to source data 
    void* dst_data;           ///< Pointer to destination data 
    size_t start_idx;         ///< Start index for this thread 
    size_t end_idx;           ///< End index for this thread 
    nnl2_tensor_type dtype;         ///< Data type of the nnl2_tensor
    float alpha;              ///< Negative slope coefficient
    bool inplace;             ///< Whether operation is in-place
} leakyrelu_ptask;

///@} [leakyrelu_ptask]



///@{ [tanh_ptask]

typedef struct {
    nnl2_tensor_type dtype;         ///< Data type of the nnl2_tensor
    bool approx;              ///< Whether to use approximation
    size_t start_idx;         ///< Start index for this thread
    size_t end_idx;           ///< End index for this thread
    void* src_data;           ///< Pointer to source data
    void* dst_data;           ///< Pointer to destination data
} tanh_ptask;

///@} [tanh_ptask]



///@{ [tanhinplace_ptask]

typedef struct {
    nnl2_tensor_type dtype;   ///< Data type of the nnl2_tensor
    bool approx;              ///< Whether to use approximation
    size_t start_idx;         ///< Start index for this thread
    size_t end_idx;           ///< End index for this thread
    void* data;               ///< Pointer to nnl2_tensor data (modified in-place)
} tanhinplace_ptask;

///@} [tanhinplace_ptask]



///@{ [sigmoid_ptask]
	
typedef struct {
    nnl2_tensor_type dtype;
    bool aligned;
    bool approx;
    size_t start_idx;
    size_t end_idx;
    void* src_data;
    void* dst_data;
} sigmoid_ptask;

///@} [sigmoid_ptask]



///@{ [sum_ptask]

typedef struct {
    void* src_data;           ///< Pointer to source data 
    size_t start_idx;         ///< Start index for this thread 
    size_t end_idx;           ///< End index for this thread 
    nnl2_tensor_type dtype;   ///< Data type of the nnl2_tensor 
    bool aligned;             ///< Whether memory is aligned 
    union {
        double float64_acc;
        float float32_acc;
        int32_t int32_acc;
    } accumulator;            ///< Thread-local accumulator 
} sum_ptask;

///@} [sum_ptask]

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
