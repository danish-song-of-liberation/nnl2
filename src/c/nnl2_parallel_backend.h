#ifndef NNL2_PARALLEL_BACKEND_H
#define NNL2_PARALLEL_BACKEND_H

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
