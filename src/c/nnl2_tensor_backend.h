#include "nnl2_core.h"

#ifndef NNL2_TENSOR_BACKEND
#define NNL2_TENSOR_BACKEND

#include <string.h>
#include <stdlib.h>

// NNL2

/**
 * @file nnl2_tensor_backend.h
 * @brief Contains the tensor and implemenets structures
 */

/** @brief 
 * Enumerations of available tensor types (INT32, FLOAT32/FLOAT, FLOAT64/DOUBLE)
 *
 */
typedef enum {
	INT32,
	FLOAT32,
	FLOAT64
} TensorType;

/** @brief 
 * Tensor structure with an arbitrary number of dimensions
 *
 *** dtype :
 *
 * Type of tensor (INT32, FLOAT32/FLOAT, FLOAT64/DOUBLE) 
 *
 *** data :
 *
 * Void pointer to tensor data
 *
 *** shape :
 *
 * Pointer to an int array of tensor dimensions
 *
 *** rank :
 *
 * The number of dimensions of the tensor
 *
 */
typedef struct {
	TensorType dtype;
	void* data;
	int* shape;
	int rank;
} Tensor;

typedef struct {
    void* fn;
    int speed_priority;
    bool available;
    const char* name;
} Implementation;

typedef enum {
	nnl2ColMajor=101,
	nnl2RowMajor=102
} nnl2_order;

typedef enum {
	nnl2NoTrans=111,
	nnl2Trans=112,
} nnl2_transpose;

typedef void (*fn_inplace_fill)(Tensor*, void*, TensorType);
typedef Tensor* (*fn_empty)(const int*, int, TensorType);
typedef Tensor* (*fn_zeros)(const int*, int, TensorType);
typedef Tensor* (*fn_ones)(const int*, int, TensorType);

typedef void (*sgemminplacefn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
							   const int, const int, const int, const float, const Tensor*, 
							   const int, const Tensor*, const int, const float, Tensor*, const int);

typedef void (*dgemminplacefn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
							   const int, const int, const int, const double, const Tensor*, 
							   const int, const Tensor*, const int, const double, Tensor*, const int);

typedef Tensor* (*sgemmfn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
					       const int, const int, const int, const float, const Tensor*, 
					       const int, const Tensor*, const int, const float);

typedef Tensor* (*dgemmfn)(const nnl2_order, const nnl2_transpose, const nnl2_transpose, 
					       const int, const int, const int, const double, const Tensor*, 
					       const int, const Tensor*, const int, const double);

typedef void (*addinplacefn)(Tensor*, const Tensor*);			
typedef void (*subinplacefn)(Tensor*, const Tensor*);	
typedef Tensor* (*addfn)(const Tensor*, const Tensor*);		
typedef Tensor* (*subfn)(const Tensor*, const Tensor*);		
typedef void (*mulinplacefn)(Tensor*, const Tensor*);		
typedef void (*divinplacefn)(Tensor*, const Tensor*);		
typedef Tensor* (*mulfn)(const Tensor*, const Tensor*);	   
typedef Tensor* (*divfn)(const Tensor*, const Tensor*);	
typedef void (*powinplacefn)(Tensor*, const Tensor*);
typedef void (*expinplacefn)(Tensor*);
typedef Tensor* (*powfn)(const Tensor*, const Tensor*);
typedef Tensor* (*expfn)(const Tensor*);
typedef void (*loginplacefn)(Tensor*);
typedef Tensor* (*logfn)(const Tensor*);
typedef void (*transposeinplacefn)(Tensor*);
typedef void (*transposefn)(const Tensor*);

char* get_tensortype_name(TensorType dtype) {
	switch(dtype) {
		case INT32:
			return "INT32";
			
		case FLOAT32:
			return "FLOAT32";
			
		case FLOAT64:
			return "FLOAT64";
			
		default:
			return "BAD DATA TYPE";
	}	
}

int* concat_int_arr(int* arr1, int size1, int* arr2, int size2) {
	int* result = malloc((size1 + size2) * sizeof(int));
	
	if(!result) {
		fprintf(stderr, "Error (Hello from C!): Memory Allocation Error (concat_int_arr)\n");
		return NULL;
	}
	
	int* p = result;
	
	for(int i = 0; i < size1; ++i) *p++ = arr1[i];
	for(int i = 0; i < size2; ++i) *p++ = arr2[i];
	
	return result;
}

int* append_int_arr(int* arr, int size, int new_element) {
    int* new_arr = (int*)realloc(arr, (size + 1) * sizeof(int));

    if (new_arr == NULL) {
        fprintf(stderr, "Error (Hello from C!): Failed to realloc (append_int_arr)\n");
        return NULL;
    }

    new_arr[size] = new_element;

    return new_arr; 
}


#endif