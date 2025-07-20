#include "nnl2_core.h"
#include "nnl2_tensor_backend.h"

#include <string.h>

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

#define NUM_TENSOR_TYPES 3 // int32, float32, float64

#if defined(__AVX512F__)
	#define TENSOR_MEM_ALIGNMENT 64
#elif defined(__AVX2__)
	#define TENSOR_MEM_ALIGNMENT 32
#elif defined(__AVX__)
	#define TENSOR_MEM_ALIGNMENT 32
#else 
	#define TENSOR_MEM_ALIGNMENT 16
#endif

#ifdef OPENBLAS_AVAILABLE
#include <cblas.h>
#endif

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
 **/
inline size_t get_dtype_size(TensorType dtype) {
	static const size_t sizes[] = {sizeof(int), sizeof(float), sizeof(double)};
		
	return sizes[dtype]; 
}	

/** @brief
 * calculates the total number of elements in the tensor specified by the shape (for calculating memory)
 *
 ** @param lst
 * pointer to an array of integers representing the tensor's shape
 *
 ** @param len
 * length of the array `lst`, which is the number of dimensions in the tensor
 *
 ** @return 
 * total number of elements in the tensor
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * size_t num_elements = product(shape, 3); // num_elements will be 24
 ** @endcode
 */
inline size_t product(const int* lst, int len) {
	size_t acc = 1;
	
	while (len--) acc *= *lst++;
	
	return acc;
}

/** @brief
 * Creates a new tensor without initializing the data.
 *
 * This function allocates memory for the Tensor structure and its data,
 * using the provided shape and data type. The data memory is not (!) initialized
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
 *** checks the input parameters for correctness
 ** then
 *** allocates memory for tensor structure
 ** then
 *** allocates memory for the shape array and copies the data into it
 ** then
 *** calculates the total size of the data required for the tensor
 ** then 
 *** allocates aligned memory for the tensor data (tensor->data)
 ** finally
 *** returns a pointer to the created tensor
 *
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * Tensor* my_tensor = empty(shape, 3, FLOAT32);
 ** @endcode
 ** @warning
 * do not forget to free the memory allocated for the tensor using free_tensor after using it
 *
 **/
Tensor* cpu64_empty(const int* shape, int rank, TensorType dtype) {	
	/** @brief
     * organicist technospecialization, pedagogical 
	 * authoritarianism, and territorial sectorization 
	 * end in numerical illiteracy and mass innumeracy
	 * (Nick Land)
	 */
	 
	// checks the input parameters for correctness
	
	if (shape == NULL) {
        fprintf(stderr, "Error at zeros: Bad shape pointer\n");
        return NULL;
    }	
	
	if (rank <= 0) {
		fprintf(stderr, "Error at zeros: Bad rank (%d). Rank must be positive\n", rank);
		return NULL;
	}
	
	if (dtype < 0 || dtype >= NUM_TENSOR_TYPES) {
        fprintf(stderr, "Error (Hello from C!): Bad tensor type (%d)\n", dtype);
        return NULL;
    }
	
	for (int i = 0; i < rank; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "Error (Hello from C!): Bad shape dimension at %d (%d). Dimensions must be positive\n",  i, shape[i]);
            return NULL;
        }
    }
	
	// allocating memory for tensor structure

	Tensor* tensor = malloc(sizeof(Tensor));
	
	if (tensor == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate memory for Tensor structure\n");
        return NULL;
    }
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	
	// allocates memory for the shape array and copies the data into it
	
	tensor->shape = malloc(rank * sizeof(int));
	
	if (tensor->shape == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate memory for shape\n");
        free(tensor);
        return NULL;
    }
	
	memcpy(tensor->shape, shape, rank * sizeof(int));
	
	// calculates the total size of the data required for the tensor
	
	size_t total_elems = product(shape, rank);
	size_t type_size = get_dtype_size(dtype);
	size_t total_size = total_elems * type_size;
	
	// allocates aligned memory for tensor data (tensor->data)
	
	void* data;
	
	ALLOC_ALIGNED(data, (size_t)TENSOR_MEM_ALIGNMENT, total_size);
	
	if (data == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate aligned memory\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
	
	tensor->data = data;
	
	return tensor;
}

Implementation empty_backends[] = {
	{cpu64_empty, 80, true, "CPU64"}
};

fn_empty empty;

void init_empty() {
	for(size_t i = 0; i < sizeof(empty_backends) / sizeof(empty_backends[0]); i++) {
		if (empty_backends[i].available) empty = empty_backends[i].fn;
	}
}

/** @brief
 * Creates a new tensor and initializes all elements to zero.
 *
 * This function allocates memory for tensor structure and its data,
 * using the provided shape and data type.
 *
 * memory is initialized to zero.
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
 *** checks the input parameters for correctness
 ** then
 *** allocates memory for tensor structure
 ** then
 *** allocates memory for the shape array and copies the data into it
 ** then
 *** calculates the total size of the data required for the tensor
 ** then 
 *** allocates aligned memory for the tensor data (tensor->data)
 ** finally
 *** returns a pointer to the created tensor
 *
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * Tensor* my_tensor = zeros(shape, 3, FLOAT32);
 ** @endcode
 ** @warning
 * do not forget to free the memory allocated for the tensor using free_tensor after using it
 *
 **/
Tensor* cpu64_zeros(const int* shape, int rank, TensorType dtype) {
	// checks the input parameters for correctness
	
	if (shape == NULL) {
        fprintf(stderr, "Error at zeros: Bad shape pointer\n");
        return NULL;
    }	
	
	if (rank <= 0) {
		fprintf(stderr, "Error at zeros: Bad rank (%d). Rank must be positive\n", rank);
		return NULL;
	}
	
	if (dtype < 0 || dtype >= NUM_TENSOR_TYPES) {
        fprintf(stderr, "Error (Hello from C!): Bad tensor type (%d)\n", dtype);
        return NULL;
    }
	
	for (int i = 0; i < rank; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "Error (Hello from C!): Bad shape dimension at %d (%d). Dimensions must be positive\n",  i, shape[i]);
            return NULL;
        }
    }
	
	// allocating memory for tensor structure

	Tensor* tensor = malloc(sizeof(Tensor));
	
	if (tensor == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate memory for Tensor structure\n");
        return NULL;
    }
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	
	// allocates memory for the shape array and copies the data into it
	 
	tensor->shape = malloc(rank * sizeof(int));
	
	if (tensor->shape == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate memory for shape\n");
        free(tensor);
        return NULL;
    }
	
	memcpy(tensor->shape, shape, rank * sizeof(int));
	
	// calculates the total size of the data required for the tensor
	
	size_t total_elems = product(shape, rank);
	size_t type_size = get_dtype_size(dtype);
	size_t total_size = total_elems * type_size;
	
	// allocates aligned memory for tensor data (tensor->data)
	 
	void* data;
	
	ALLOC_ALIGNED(data, (size_t)TENSOR_MEM_ALIGNMENT, total_size);
	
	if (data == NULL) {
        fprintf(stderr, "Error (Hello from C!): failed to allocate aligned memory\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
	
	tensor->data = data;
	memset(tensor->data, 0, total_size);
	
	return tensor;
}

Implementation zeros_backends[] = {
	{cpu64_zeros, 80, true, "CPU64"}
};

fn_zeros zeros;

void init_zeros() {
	for(size_t i = 0; i < sizeof(zeros_backends) / sizeof(zeros_backends[0]); i++) {
		if (zeros_backends[i].available) zeros = zeros_backends[i].fn;
	}
}

/** @brief
 * Frees the memory allocated for the tensor.
 *
 ** @param tensor 
 * A pointer to a tensor whose memory needs to be freed. If tensor is null, the function does nothing.
 *
 ** @note
 * After calling this function tensor pointer becomes invalid. Do not attempt to access the tensor after it has been freed (although you'll try it anyway, you idiot).
 *
 */
void free_tensor(Tensor* tensor) {
	/** @brief
	 * in fact, my fascination with ML and framework creation
	 * is nothing more than an act of selfishness and escapism
	 */
	
	if (tensor == NULL) return;
	
	free(tensor->shape); 
	FREE_ALIGNED(tensor->data); 
	free(tensor);
	
	/** @brief
	 * life in general is a miserable, miserable thing, 
	 * it has always been miserable and unhappy, and it will always 
	 * be miserable and unhappy, and nonexistence is better than existence 
	 * (Philipp Mainländer)
	 */
}

/** @brief
 * Returns a pointer to a tensor element with the specified indices or a subtensor.
 *
 ** @param tensor 
 * Pointer to a tensor
 *
 ** @param indices 
 * Pointer to an array of indices
 *
 ** @param num_indices
 * Number of provided indices.
 *
 ** @details
 * The function firstly:
 *
 *
 *** Checks the input parameters for correctness
 ** then
 *** If num_indices is equal to the rank of the tensor, it calculates the offset of the element in memory and returns a pointer to that element, casting it to the correct data type
 ** then 
 *** If num_indices is less than the rank of the tensor, it creates a new subtenor, copying information about the data type, rank and shape from the original tensor, calculates the offset
 *** in memory, sets the pointer data to the beginning of the subtenor data and returns a pointer to the new subtenor
 ** finally
 *** If num_indices is greater than the rank of the tensor returns null
 *
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * Tensor* my_tensor = zeros(shape, 3, FLOAT32);
 *
 * int shape_2[] = {1, 2, 3};
 * float* element = (float*)at(my_tensor, shape_2, 3);
 *
 * printf("%f\n", *element);
 *
 ** second example:
 * int shape[] = {2, 3, 4};
 * Tensor* my_tensor = zeros(shape, 3, FLOAT32);
 *
 * int shape_2[] = {1, 2};
 * float* element = (float*)at(my_tensor, shape_2, 2);
 *
 * for(int i = 0; i < 2; i++) {
 *     printf("%f\n", *element[i]);
 * } 
 * 
 ** @return
 * Pointer to a tensor element or subtensor
 *
 **/
void* at(Tensor* tensor, const int32_t* indices, uint8_t num_indices) {
	// Checks the input parameters for correctness
	 
	if (tensor == NULL) {
        fprintf(stderr, "Error (Hello from C!): Bad tensor pointer\n");
        return NULL;
    }
	
	if (indices == NULL) {
        fprintf(stderr, "Error (Hello from C!): Bad indices pointer\n");
        return NULL;
    }
	
	const int32_t tensor_rank = tensor->rank;
	
	if (tensor_rank <= 0) {
		fprintf(stderr, "Error (Hello from C!): Bad rank (%d). Rank must be positive\n", tensor_rank);
		return NULL;
	}	
	
	for (int i = 0; i < tensor_rank; i++) {
        if (indices[i] < 0) {
            fprintf(stderr, "Error (Hello from C!): Bad shape dimension at %d (%d). Dimensions must be positive\n",  i, indices[i]);
            return NULL;
        }
    }
	
	for (int i = 0; i < num_indices; i++) {
        if (indices[i] < 0 || (i < tensor->rank && indices[i] >= tensor->shape[i])) {
            fprintf(stderr, "Error (Hello from C!): Index %d (%d) out of bounds for dimension %d (size %d)\n", i, indices[i], i, tensor->shape[i]);
            return NULL;
        }
    }
	
	// If num_indices is equal to the rank of the tensor, it calculates the offset of the element 
	// in memory and returns a pointer to that element, casting it to the correct data type
	
	if (num_indices == tensor_rank) {
		size_t offset = 0;
		size_t stride = 1;
		
		for(uint32_t i = tensor_rank; i-- > 0;) {
			offset += indices[i] * stride;
			stride *= tensor->shape[i];
		}
		
		switch(tensor->dtype) {
			case INT32:   return (int32_t*)tensor->data + offset;
			case FLOAT32: return (float*)tensor->data + offset;
			case FLOAT64: return (double*)tensor->data + offset;
			default: return NULL;
		}
	} else if (num_indices < tensor_rank) {
		// If num_indices is less than the rank of the tensor, it creates a new subtenor, copying information about the data type, rank and shape from the 
	    // original tensor, calculates the offset in memory, sets the pointer data to the beginning of the subtenor data and returns a pointer to the new subtenor
		
		Tensor* subtensor = (Tensor*)malloc(sizeof(Tensor));
		
		if (subtensor == NULL) {
            fprintf(stderr, "Error (Hello from C!): Failed to allocate subtensor\n");
            return NULL;
        }
		
		subtensor->dtype = tensor->dtype;
		subtensor->rank = tensor_rank - num_indices;
		subtensor->shape = (int*)malloc(subtensor->rank * sizeof(int));
		
		if (!subtensor->shape) {
			fprintf(stderr, "Error (Hello from C!): Failed to allocate subtensor shape\n");
			free(subtensor);  
			return NULL;
		}
		
		memcpy(subtensor->shape, tensor->shape + num_indices, subtensor->rank * sizeof(int32_t));
		
		size_t offset = 0;
		size_t stride = 1;
		
		for(uint32_t i = num_indices; i-- > 0;) {
			offset += indices[i] * stride;
			stride *= tensor->shape[i];
		}
		
		subtensor->data = (uint8_t*)tensor->data + offset * get_dtype_size(tensor->dtype);
		
		return subtensor;
	} else {
		// If num_indices is greater than the rank of the tensor returns null
		
		fprintf(stderr, "Error (Hello from C!): Incorrent indices\n");
		
		return NULL;
	}
}

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
			
			size_t it = 0;
			
			for(; it < total_elems - 7; it += 8) _mm256_storeu_si256((__m256i*)(data + it), avx_filler);		
			for(size_t j = it; j < total_elems; j++) data[j] = filler;
				
			break;
		}
		
		case FLOAT32: {
			float filler = *(float*)value;
			float* data = (float*)tensor->data;
			
			__m256 avx_filler = _mm256_set1_ps(filler);
			
			size_t it = 0;
			
			for(; it < total_elems - 7; it += 8) _mm256_storeu_ps(data + it, avx_filler);		
			for(size_t j = it; j < total_elems; j++) data[j] = filler;
			
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
	{naive_inplace_fill, 10, true, "NAIVE"},
	
	#ifdef __AVX__
	{avx_inplace_fill, 70, true, "AVX"}
	#endif
};

fn_inplace_fill inplace_fill;

void init_inplace_fill() {
	for(size_t i = 0; i < sizeof(inplace_fill_backends) / sizeof(inplace_fill_backends[0]); i++) {
		if (inplace_fill_backends[i].available) inplace_fill = inplace_fill_backends[i].fn;
	}	
}

Tensor* cpu64_ones(const int* shape, int rank, TensorType dtype) {
    Tensor* tensor_t = empty(shape, rank, dtype);

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
            free_tensor(tensor_t);  
            return NULL;
        }
    }
    
    return tensor_t;
}

Implementation ones_backends[] = {
	{cpu64_ones, 80, true, "CPU64"}
};

fn_ones ones;

void init_ones() {
	for(size_t i = 0; i < sizeof(ones_backends) / sizeof(ones_backends[0]); i++) {
		if (ones_backends[i].available) ones = ones_backends[i].fn;
	}
}

Tensor* full(const int* shape, int rank, TensorType dtype, void* filler) {
	Tensor* tensor_t = empty(shape, rank, dtype);
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

Implementation sgemminplace_backends[] = {
	{naive_sgemminplace, 10, true, "NAIVE"}
};

sgemminplacefn sgemminplace;

void init_sgemminplace() {
	for(size_t i = 0; i < sizeof(sgemminplace_backends) / sizeof(sgemminplace_backends[0]); i++) {
		if (sgemminplace_backends[i].available) sgemminplace = sgemminplace_backends[i].fn;
	}
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

Implementation dgemminplace_backends[] = {
	{naive_dgemminplace, 10, true, "NAIVE"}
};

dgemminplacefn dgemminplace;

void init_dgemminplace() {
	for(size_t i = 0; i < sizeof(dgemminplace_backends) / sizeof(dgemminplace_backends[0]); i++) {
		if (dgemminplace_backends[i].available) dgemminplace = dgemminplace_backends[i].fn;
	}
}

Tensor* naive_sgemm(const nnl2_order order, const nnl2_transpose transa, 
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

Implementation sgemm_backends[] = {
	{naive_sgemm, 10, true, "NAIVE"}
};

sgemmfn sgemm;

void init_sgemm() {
	for(size_t i = 0; i < sizeof(sgemm_backends) / sizeof(sgemm_backends[0]); i++) {
		if (sgemm_backends[i].available) sgemm = sgemm_backends[i].fn;
	}
}

Tensor* naive_dgemm(const nnl2_order order, const nnl2_transpose transa, 
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

Implementation dgemm_backends[] = {
	{naive_dgemm, 10, true, "NAIVE"}
};

dgemmfn dgemm;

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

void init_dgemm() {
	for(size_t i = 0; i < sizeof(dgemm_backends) / sizeof(dgemm_backends[0]); i++) {
		if (dgemm_backends[i].available) dgemm = dgemm_backends[i].fn;
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

#endif
