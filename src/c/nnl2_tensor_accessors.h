#include "nnl2_core.h"
#include "nnl2_tensor_backend.h"

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
	if (tensor == NULL) return;
	
	free(tensor->shape); 
	FREE_ALIGNED(tensor->data); 
	free(tensor);
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
 ** second example (WARNING ITS NOT WORKING I WILL FIX THIS SOON):
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
	
	// WARNING: i remaded function after docs
	
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
			
			size_t avx_iters = total_elems / 8; // there are ghosts in my code
			for (size_t i = 0; i < avx_iters; i++) { // WHY DOESN'T AVX_ITERS WORK IN LISP, BUT IT WORKS IN C? WHAT IS GOING ON?
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
			
			for(; it < total_elems - 3; it += 4) _mm256_storeu_pd(data + it, avx_filler); // I MEAN THIS
			// wth does fleet64 work with this but not other data types, and only in Lisp wrappers?
			// I would make such concise loops everywhere, but ghosts don't like it
			
			for(size_t j = it; j < total_elems; j++) data[j] = filler;
			
			break;
		}
	}
}
#endif

Implementation inplace_fill_backends[] = {
	{naive_inplace_fill, 10, true, "NAIVE"},
	
	#ifdef __AVX__
	// todo {avx_inplace_fill, 70, true, "AVX"},
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
	{naive_sgemminplace, 10, true, "NAIVE"},
	
	#ifdef OPENBLAS_AVAILABLE
	{blas_sgemminplace, 100, true, "OPENBLAS"}
	#endif
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
	{naive_dgemminplace, 10, true, "NAIVE"},
	
	#ifdef OPENBLAS_AVAILABLE
	{blas_dgemminplace, 100, true, "OPENBLAS"}
	#endif
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

Implementation addinplace_backends[] = {
	{naive_addinplace, 10, true, "NAIVE"},
	
	#ifdef __AVX__
	{avx_addinplace, 70, true, "AVX"}
	#endif
};

addinplacefn addinplace;

void init_addinplace() {
	for(size_t i = 0; i < sizeof(addinplace_backends) / sizeof(addinplace_backends[0]); i++) {
		if (addinplace_backends[i].available) addinplace = addinplace_backends[i].fn;
	}
}

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
	{naive_subinplace, 10, true, "NAIVE"},
	
	#ifdef __AVX__
	{avx_subinplace, 70, true, "AVX"},
	#endif
};

subinplacefn subinplace;

void init_subinplace() {
	for(size_t i = 0; i < sizeof(subinplace_backends) / sizeof(subinplace_backends[0]); i++) {
		if (subinplace_backends[i].available) subinplace = subinplace_backends[i].fn;
	}
}

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
	
	Tensor* amount = zeros(summand->shape, summand->rank, dtype_summand);
	
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
    
    Tensor* sum = zeros(summand->shape, summand->rank, dtype_summand);
    
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
	{naive_add, 65, true, "NAIVE"},
	
	#ifdef __AVX__
	{avx_add, 70, true, "AVX"},
	#endif
};

addfn add; 

void init_add() {
	for(size_t i = 0; i < sizeof(add_backends) / sizeof(add_backends[0]); i++) {
		if (add_backends[i].available) add = add_backends[i].fn;
	}
}

Tensor* naive_sub(const Tensor* minuend, const Tensor* subtrahend) {
	size_t len = product(minuend->shape, minuend->rank);
	
	TensorType dtype_minuend = minuend->dtype;
	TensorType dtype_subtrahend = subtrahend->dtype;
	
	if(dtype_minuend != dtype_subtrahend) {
		fprintf(stderr, "Error (Hello from C!): In sub (in-place) data-types are other\n");
		return NULL;
	}
	
	Tensor* difference = zeros(minuend->shape, minuend->rank, dtype_minuend);
	
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
    
    Tensor* difference = zeros(minuend->shape, minuend->rank, dtype_minuend);
    
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
	{naive_sub, 65, true, "NAIVE"},
	
	#ifdef __AVX__
	{avx_sub, 70, true, "AVX"},
	#endif
};

addfn sub; 

void init_sub() {
	for(size_t i = 0; i < sizeof(sub_backends) / sizeof(sub_backends[0]); i++) {
		if (sub_backends[i].available) sub = sub_backends[i].fn;
	}
}

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
	{naive_mulinplace, 10, true, "NAIVE"},
};

mulinplacefn mulinplace; 

void init_mulinplace() {
	for(size_t i = 0; i < sizeof(mulinplace_backends) / sizeof(mulinplace_backends[0]); i++) {
		if (mulinplace_backends[i].available) mulinplace = mulinplace_backends[i].fn;
	}
}

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
	{naive_divinplace, 10, true, "NAIVE"},
};

divinplacefn divinplace; 

void init_divinplace() {
	for(size_t i = 0; i < sizeof(divinplace_backends) / sizeof(divinplace_backends[0]); i++) {
		if (divinplace_backends[i].available) divinplace = divinplace_backends[i].fn;
	}
}

Tensor* naive_mul(const Tensor* multiplicand, const Tensor* multiplier) {
    size_t len = product(multiplicand->shape, multiplicand->rank);
    
    TensorType dtype_multiplicand = multiplicand->dtype;
    TensorType dtype_multiplier = multiplier->dtype;
    
    if(dtype_multiplicand != dtype_multiplier) {
        fprintf(stderr, "Error (Hello from C!): In mul (in-place) data-types are other\n");
        return NULL;
    }
    
    Tensor* product = zeros(multiplicand->shape, multiplicand->rank, dtype_multiplicand);
    
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
	{naive_mul, 10, true, "NAIVE"},
};

mulfn mul; 

void init_mul() {
	for(size_t i = 0; i < sizeof(mul_backends) / sizeof(mul_backends[0]); i++) {
		if (mul_backends[i].available) mul = mul_backends[i].fn;
	}
}

Tensor* naive_div(const Tensor* dividend, const Tensor* divisor) {
    size_t len = product(dividend->shape, dividend->rank);
    
    TensorType dtype_dividend = dividend->dtype;
    TensorType dtype_divisor = divisor->dtype;
    
    if (dtype_dividend != dtype_divisor) {
        fprintf(stderr, "Error (Hello from C!): In div (in-place) data-types are different\n");
        return NULL;
    }
    
    Tensor* quotient = zeros(dividend->shape, dividend->rank, dtype_dividend);
    
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
                    free_tensor(quotient);
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
            free_tensor(quotient);
            return NULL;
        }
    }
    
    return quotient;
}

Implementation div_backends[] = {
	{naive_div, 10, true, "NAIVE"},
};

divfn nnl2_div;

void init_div() {
	for(size_t i = 0; i < sizeof(div_backends) / sizeof(div_backends[0]); i++) {
		if (div_backends[i].available) nnl2_div = div_backends[i].fn;
	}
}

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
	{naive_powinplace, 10, true, "NAIVE"}
};	
	
powinplacefn powinplace;	

void init_powinplace() {
	for(size_t i = 0; i < sizeof(powinplace_backends) / sizeof(powinplace_backends[0]); i++) {
		if (powinplace_backends[i].available) powinplace = powinplace_backends[i].fn;
	}
}
	
Tensor* naive_pow(const Tensor* base, const Tensor* exponent) {
    size_t len = product(base->shape, base->rank);
    
    TensorType dtype_base = base->dtype;
    TensorType dtype_exponent = exponent->dtype;
    
    if(dtype_base != dtype_exponent) {
        fprintf(stderr, "Error (Hello from C!): In pow data-types are different\n");
        return NULL;
    }
    
    Tensor* result = zeros(base->shape, base->rank, dtype_base);
    
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
	{naive_pow, 10, true, "NAIVE"}
};	
	
powfn nnl2_pow;	

void init_pow() {
	for(size_t i = 0; i < sizeof(pow_backends) / sizeof(pow_backends[0]); i++) {
		if (pow_backends[i].available) nnl2_pow = pow_backends[i].fn;
	}
}

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
	{naive_expinplace, 10, true, "NAIVE"}
};	

expinplacefn expinplace;

void init_expinplace() {
	for(size_t i = 0; i < sizeof(expinplace_backends) / sizeof(expinplace_backends[0]); i++) {
		if (expinplace_backends[i].available) expinplace = expinplace_backends[i].fn;
	}
}

Tensor* naive_exp(const Tensor* tensor) {
	size_t len = product(tensor->shape, tensor->rank);
	
	Tensor* result = empty(tensor->shape, tensor->rank, tensor->dtype);
	
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
	{naive_exp, 10, true, "NAIVE"}
};	

expfn nnl2_exp;

void init_exp() {
	for(size_t i = 0; i < sizeof(exp_backends) / sizeof(exp_backends[0]); i++) {
		if (exp_backends[i].available) nnl2_exp = exp_backends[i].fn;
	}
}

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
	{naive_loginplace, 10, true, "NAIVE"},
};	

loginplacefn loginplace;

void init_loginplace() {
	for(size_t i = 0; i < sizeof(loginplace_backends) / sizeof(loginplace_backends[0]); i++) {
		if (loginplace_backends[i].available) loginplace = loginplace_backends[i].fn;
	}
}

Tensor* naive_log(const Tensor* tensor) {
	size_t len = product(tensor->shape, tensor->rank);
	
	Tensor* result = empty(tensor->shape, tensor->rank, tensor->dtype);
	
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
	{naive_log, 10, true, "NAIVE"},
};	

logfn nnl2_log;

void init_log() {
	for(size_t i = 0; i < sizeof(log_backends) / sizeof(log_backends[0]); i++) {
		if (log_backends[i].available) nnl2_log = log_backends[i].fn;
	}
}

void at_set(Tensor* tensor, int* shape, int rank, void* change_with) {
	TensorType tensor_dtype = tensor->dtype;
	
	for(int i = 0; i < rank; i++) {
		if(shape[i] == -1) {
			for(int shape_i = 0; shape_i < tensor->shape[i]; shape_i++) {
				shape[i] = shape_i;
				at_set(tensor, shape, rank, change_with);
			}
			
			shape[i] = -1;
			return;
		}
	}
	
	if(rank == tensor->rank) {
		switch(tensor_dtype) {
			case FLOAT64: {
				double* change_elem = (double*)change_with;
				double* elem = (double*)at(tensor, shape, rank);
				
				*elem = *change_elem;
					
				break;
			}
				
			case FLOAT32: {
				float* change_elem = (float*)change_with;
				float* elem = (float*)at(tensor, shape, rank);
					
				*elem = *change_elem;
					
				break;
			}
				
			case INT32: {
				int32_t* change_elem = (int32_t*)change_with;
				int32_t* elem = (int32_t*)at(tensor, shape, rank);
					
				*elem = *change_elem;
					
				break;
			}
				
			default: {
				fprintf(stderr, "Error (Hello from C!): Bad data-type (tref setter)\n");
				return;
			}
		}
	} else {
		for(int i = 0; i < tensor->rank; i++) {
			int new_rank = rank + 1;
			int* subshape = (int*)malloc(sizeof(int) * new_rank);  

			if (subshape == NULL) {
				fprintf(stderr, "Error (Hello from C!): Memory allocation failed (tref-setter)\n");
				return;
			}

			memcpy(subshape, shape, sizeof(int) * rank);

			for (int i = 0; i < tensor->shape[rank]; i++) {
				subshape[rank] = i; 
				at_set(tensor, subshape, new_rank, change_with); 
			}
	
			free(subshape);
			return;
		}
	}
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
	{naive_scaleinplace, 10, true, "NAIVE"},
};	

scaleinplacefn scaleinplace;

void init_scaleinplace() {
	for(size_t i = 0; i < sizeof(scaleinplace_backends) / sizeof(scaleinplace_backends[0]); i++) {
		if (scaleinplace_backends[i].available) scaleinplace = scaleinplace_backends[i].fn;
	}
}

Tensor* naive_scale(const Tensor* tensor, float multiplier) {
	Tensor* result = empty(tensor->shape, tensor->rank, tensor->dtype);
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
	{naive_scale, 10, true, "NAIVE"},
};	

scalefn scale;

void init_scale() {
	for(size_t i = 0; i < sizeof(scale_backends) / sizeof(scale_backends[0]); i++) {
		if (scale_backends[i].available) scale = scale_backends[i].fn;
	}
}

Tensor* empty_like(const Tensor* tensor) {
	return empty(tensor->shape, tensor->rank, tensor->dtype);
}

Tensor* zeros_like(const Tensor* tensor) {
	return zeros(tensor->shape, tensor->rank, tensor->dtype);
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
	{naive_maxinplace, 10, true, "NAIVE"},
};	

maxinplacefn maxinplace;

void init_maxinplace() {
	for(size_t i = 0; i < sizeof(maxinplace_backends) / sizeof(maxinplace_backends[0]); i++) {
		if (maxinplace_backends[i].available) maxinplace = maxinplace_backends[i].fn;
	}
}

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
	{naive_mininplace, 10, true, "NAIVE"},
};	

mininplacefn mininplace;

void init_mininplace() {
	for(size_t i = 0; i < sizeof(mininplace_backends) / sizeof(mininplace_backends[0]); i++) {
		if (mininplace_backends[i].available) mininplace = mininplace_backends[i].fn;
	}
}

Tensor* naive_max(const Tensor* tensora, const Tensor* tensorb) {
	TensorType typea = tensora->dtype, typeb = tensorb->dtype;
	
	if(typea != typeb) {
		fprintf(stderr, "Error (Hello from C!): Data types are different (max in-place)\n");
		return NULL;
	}
	
	int total_elems = product(tensora->shape, tensora->rank);
	
	Tensor* result = empty(tensora->shape, tensora->rank, typea);
	
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
	{naive_max, 10, true, "NAIVE"},
};	

maxfn nnl2_max;

void init_max() {
	for(size_t i = 0; i < sizeof(max_backends) / sizeof(max_backends[0]); i++) {
		if (max_backends[i].available) nnl2_max = max_backends[i].fn;
	}
}

Tensor* naive_min(const Tensor* tensora, const Tensor* tensorb) {
	TensorType typea = tensora->dtype, typeb = tensorb->dtype;
	
	if(typea != typeb) {
		fprintf(stderr, "Error (Hello from C!): Data types are different (min in-place)\n");
		return NULL;
	}
	
	int total_elems = product(tensora->shape, tensora->rank);
	
	Tensor* result = empty(tensora->shape, tensora->rank, typea);
	
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
	{naive_min, 10, true, "NAIVE"},
};	

minfn nnl2_min;

void init_min() {
	for(size_t i = 0; i < sizeof(min_backends) / sizeof(min_backends[0]); i++) {
		if (min_backends[i].available) nnl2_min = min_backends[i].fn;
	}
}

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
	{naive_absinplace, 10, true, "NAIVE"},
};	

absinplacefn absinplace;

void init_absinplace() {
	for(size_t i = 0; i < sizeof(absinplace_backends) / sizeof(absinplace_backends[0]); i++) {
		if (absinplace_backends[i].available) absinplace = absinplace_backends[i].fn;
	}
}

Tensor* naive_abs(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	Tensor* result = empty(tensor->shape, tensor->rank, tensor->dtype);
	
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
	{naive_abs, 10, true, "NAIVE"},
};	

absfn nnl2_abs;

void init_abs() {
	for(size_t i = 0; i < sizeof(abs_backends) / sizeof(abs_backends[0]); i++) {
		if (abs_backends[i].available) nnl2_abs = abs_backends[i].fn;
	}
}

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
		result = empty(shapec, 1, typea);
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

		result = empty(shapec, ranka, typea);
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
	{naive_hstack, 10, true, "NAIVE"},
};	

hstackfn hstack;

void init_hstack() {
	for(size_t i = 0; i < sizeof(hstack_backends) / sizeof(hstack_backends[0]); i++) {
		if (hstack_backends[i].available) hstack = hstack_backends[i].fn;
	}
}

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
		result = empty(shapec, 2, typea);
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
		
		result = empty(shapec, 2, typea);
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
		
		result = empty(shapec, 2, typea);
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

		result = empty(shapec, ranka, typea);
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
	{naive_vstack, 10, true, "NAIVE"},
};	

vstackfn vstack;

void init_vstack() {
	for(size_t i = 0; i < sizeof(vstack_backends) / sizeof(vstack_backends[0]); i++) {
		if (vstack_backends[i].available) vstack = vstack_backends[i].fn;
	}
}

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
	{naive_reluinplace, 10, true, "NAIVE"},
};	

reluinplacefn reluinplace;

void init_reluinplace() {
	for(size_t i = 0; i < sizeof(reluinplace_backends) / sizeof(reluinplace_backends[0]); i++) {
		if (reluinplace_backends[i].available) reluinplace = reluinplace_backends[i].fn;
	}
}

Tensor* naive_relu(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	Tensor* result = empty(tensor->shape, tensor->rank, tensor->dtype);
	
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
	{naive_relu, 10, true, "NAIVE"},
};	

relufn relu;

void init_relu() {
	for(size_t i = 0; i < sizeof(relu_backends) / sizeof(relu_backends[0]); i++) {
		if (relu_backends[i].available) relu = relu_backends[i].fn;
	}
}

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
	{naive_leakyreluinplace, 10, true, "NAIVE"},
};	

leakyreluinplacefn leakyreluinplace;

void init_leakyreluinplace() {
	for(size_t i = 0; i < sizeof(leakyreluinplace_backends) / sizeof(leakyreluinplace_backends[0]); i++) {
		if (leakyreluinplace_backends[i].available) leakyreluinplace = leakyreluinplace_backends[i].fn;
	}
}

Tensor* naive_leakyrelu(Tensor* tensor, float alpha) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	Tensor* result = empty(tensor->shape, tensor->rank, tensor->dtype);
	
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
				free_tensor(result);
				
				result = empty(tensor->shape, tensor->rank, FLOAT64);
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
	{naive_leakyrelu, 10, true, "NAIVE"},
};	

leakyrelufn leakyrelu;

void init_leakyrelu() {
	for(size_t i = 0; i < sizeof(leakyrelu_backends) / sizeof(leakyrelu_backends[0]); i++) {
		if (leakyrelu_backends[i].available) leakyrelu = leakyrelu_backends[i].fn;
	}
}

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
	{naive_sigmoidinplace, 10, true, "NAIVE"},
};	

sigmoidinplacefn sigmoidinplace;

void init_sigmoidinplace() {
	for(size_t i = 0; i < sizeof(sigmoidinplace_backends) / sizeof(sigmoidinplace_backends[0]); i++) {
		if (sigmoidinplace_backends[i].available) sigmoidinplace = sigmoidinplace_backends[i].fn;
	}
}

Tensor* naive_sigmoid(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = empty(tensor->shape, tensor->rank, dtype);
	
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
	{naive_sigmoid, 10, true, "NAIVE"},
};	

sigmoidfn sigmoid;

void init_sigmoid() {
	for(size_t i = 0; i < sizeof(sigmoid_backends) / sizeof(sigmoid_backends[0]); i++) {
		if (sigmoid_backends[i].available) sigmoid = sigmoid_backends[i].fn;
	}
}

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
	{naive_tanhinplace, 10, true, "NAIVE"},
};	

tanhinplacefn tanhinplace;

void init_tanhinplace() {
	for(size_t i = 0; i < sizeof(tanhinplace_backends) / sizeof(tanhinplace_backends[0]); i++) {
		if (tanhinplace_backends[i].available) tanhinplace = tanhinplace_backends[i].fn;
	}
}

Tensor* naive_tanh(Tensor* tensor) {	
	int total_elems = product(tensor->shape, tensor->rank);	
	
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = empty(tensor->shape, tensor->rank, dtype);
	
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
	{naive_tanh, 10, true, "NAIVE"},
};	

tanhfn nnl2_tanh;

void init_tanh() {
	for(size_t i = 0; i < sizeof(tanh_backends) / sizeof(tanh_backends[0]); i++) {
		if (tanh_backends[i].available) nnl2_tanh = tanh_backends[i].fn;
	}
}

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
    
    Tensor* result = empty(shapec, ranka, typea);
    
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
	{naive_concat, 10, true, "NAIVE"},
};	

concatfn nnl2_concat;

void init_concat() {
	for(size_t i = 0; i < sizeof(concat_backends) / sizeof(concat_backends[0]); i++) {
		if (concat_backends[i].available) nnl2_concat = concat_backends[i].fn;
	}
}

Tensor* naive_randn(int* shape, int rank, TensorType dtype, void* from, void* to) {
	Tensor* result = empty(shape, rank, dtype);
	
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
	{naive_randn, 10, true, "NAIVE"},
};	

randnfn randn;

void init_randn() {
	for(size_t i = 0; i < sizeof(randn_backends) / sizeof(randn_backends[0]); i++) {
		if (randn_backends[i].available) randn = randn_backends[i].fn;
	}
}

Tensor* randn_like(const Tensor* tensor, void* from, void* to) {
	return randn(tensor->shape, tensor->rank, tensor->dtype, from, to);
}

Tensor* naive_xavier(int* shape, int rank, TensorType dtype, int in, int out, float gain, float distribution) {
	if(dtype == INT32) {
		fprintf(stderr, "Error (Hello from C!): Provided an not supported type (xavier-naive)\n");
		return NULL;
	}
	
	Tensor* result = empty(shape, rank, dtype);
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
	{naive_xavier, 10, true, "NAIVE"},
};	

xavierfn xavier;

void init_xavier() {
	for(size_t i = 0; i < sizeof(xavier_backends) / sizeof(xavier_backends[0]); i++) {
		if (xavier_backends[i].available) xavier = xavier_backends[i].fn;
	}
}

#endif
