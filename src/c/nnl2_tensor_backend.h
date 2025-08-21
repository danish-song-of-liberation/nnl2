#include "nnl2_core.h"

#ifndef NNL2_TENSOR_BACKEND
#define NNL2_TENSOR_BACKEND

#include <string.h>
#include <stdlib.h>

#define REGISTER_BACKEND(fn, speed, name) { fn, speed, true, name }
#define INIT_BACKEND(fn_var, backends_array) fn_var = init_backend(backends_array, sizeof(backends_array)/sizeof(backends_array[0]))
#define EINIT_BACKEND(fn_var, backends_array, cur_pntr) fn_var = einit_backend(backends_array, sizeof(backends_array)/sizeof(backends_array[0]), cur_pntr)
#define SET_BACKEND_BY_NAME(backend, fn, backend_name) set_backend_by_name(backend, sizeof(backend)/sizeof(backend[0]), (void**)&fn, backend_name)
#define GET_BACKENDS(backend) get_backends(backend, sizeof(backend)/sizeof(backend[0]))
#define ESET_BACKEND_BY_NAME(backend, fn, backend_name, cur_pntr) eset_backend_by_name(backend, sizeof(backend)/sizeof(backend[0]), (void**)&fn, backend_name, cur_pntr)
#define DEFINE_GET_BACKENDS_FUNCTION(name) const char** get_##name##_backends() { return GET_BACKENDS(name##_backends); }
#define DEFINE_GET_NUMS_BACKENDS_FUNCTION(name) size_t get_##name##_num_backends() { return sizeof(name##_backends) / sizeof(name##_backends[0]); }
#define MAX_BACKEND_NAME_LENGTH 32

// NNL2

/**
 * @file nnl2_tensor_backend.h
 * @brief Contains the tensor and implemenets structures
 */

/** @brief 
 * Enumerations of available tensor types (INT32/INT, FLOAT32/FLOAT, FLOAT64/DOUBLE)
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

typedef enum {
	nnl2_naive,
	nnl2_avx128,
	nnl2_avx256,
	nnl2_blas,
	nnl2_implver_count
} implver;

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
typedef Tensor* (*transposefn)(const Tensor*);
typedef void (*scaleinplacefn)(Tensor*, float);
typedef Tensor* (*scalefn)(const Tensor*, float);
typedef void (*maxinplacefn)(Tensor*, const Tensor*);
typedef void (*mininplacefn)(Tensor*, const Tensor*);
typedef Tensor* (*maxfn)(const Tensor*, const Tensor*);
typedef Tensor* (*minfn)(const Tensor*, const Tensor*);
typedef void (*absinplacefn)(Tensor*);
typedef Tensor* (*absfn)(const Tensor*);
typedef Tensor* (*hstackfn)(const Tensor*, const Tensor*);
typedef Tensor* (*vstackfn)(const Tensor*, const Tensor*);
typedef void (*reluinplacefn)(Tensor*);
typedef Tensor* (*relufn)(const Tensor*);
typedef void (*leakyreluinplacefn)(Tensor*, float);
typedef Tensor* (*leakyrelufn)(const Tensor*, float);
typedef void (*sigmoidinplacefn)(Tensor*);
typedef Tensor* (*sigmoidfn)(const Tensor*);
typedef void (*tanhinplacefn)(Tensor*);
typedef Tensor* (*tanhfn)(const Tensor*);
typedef Tensor* (*concatfn)(const Tensor*, const Tensor*, int);
typedef Tensor* (*randnfn)(const int*, int, TensorType, void*, void*);
typedef Tensor* (*xavierfn)(int*, int, TensorType, int, int, float, float);
typedef void (*sumfn)(const Tensor*, int*, int);
typedef void (*l2normfn)(const Tensor*, int*, int);
typedef Tensor* (*copyfn)(const Tensor*);
typedef void (*addincfinplacefn)(Tensor*, void*);
typedef Tensor* (*addincffn)(const Tensor*, void*);
typedef void (*subdecfinplacefn)(Tensor*, void*);
typedef Tensor* (*subdecffn)(const Tensor*, void*);
typedef void (*mulmulfinplacefn)(Tensor*, void*);
typedef Tensor* (*mulmulffn)(const Tensor*, void*);
typedef void (*divdivfinplacefn)(Tensor*, void*);
typedef Tensor* (*divdivffn)(const Tensor*, void*);
typedef void (*powpowfinplacefn)(Tensor*, void*);
typedef Tensor* (*powpowffn)(const Tensor*, void*);
typedef void (*maxmaxfinplacefn)(Tensor*, void*);
typedef Tensor* (*maxmaxffn)(Tensor*, void*);
typedef void (*minminfinplacefn)(Tensor*, void*);
typedef Tensor* (*minminffn)(Tensor*, void*);
typedef void (*addbroadcastinginplacefn)(Tensor*, const Tensor*);
typedef Tensor* (*addbroadcastingfn)(const Tensor*, const Tensor*);
typedef void (*subbroadcastinginplacefn)(Tensor*, const Tensor*);
typedef Tensor* (*subbroadcastingfn)(const Tensor*, const Tensor*);
typedef void (*mulbroadcastinginplacefn)(Tensor*, const Tensor*);
typedef Tensor* (*mulbroadcastingfn)(const Tensor*, const Tensor*);
typedef void (*divbroadcastinginplacefn)(Tensor*, const Tensor*);
typedef Tensor* (*divbroadcastingfn)(const Tensor*, const Tensor*);
typedef void (*powbroadcastinginplacefn)(Tensor*, const Tensor*);
typedef Tensor* (*powbroadcastingfn)(const Tensor*, const Tensor*);
typedef void (*maxbroadcastinginplacefn)(Tensor*, const Tensor*);
typedef void (*minbroadcastinginplacefn)(Tensor*, const Tensor*);
typedef Tensor* (*maxbroadcastingfn)(const Tensor*, const Tensor*);
typedef Tensor* (*minbroadcastingfn)(const Tensor*, const Tensor*);
typedef void (*filltensorwithdatafn)(Tensor*, void*, size_t);
typedef void (*axpyinplacefn)(Tensor*, const Tensor*, float); 
typedef Tensor* (*axpyfn)(const Tensor*, const Tensor*, float); 
typedef void (*axpfinplacefn)(Tensor*, void*, float);
typedef Tensor* (*axpffn)(const Tensor*, void*, float);
typedef void (*axpybroadcastinginplacefn)(Tensor*, const Tensor*, float);
typedef Tensor* (*axpybroadcastingfn)(const Tensor*, const Tensor*, float);
typedef void* (*trefgetterfn)(Tensor*, const int32_t*, uint8_t);
typedef void* (*trefsetterfn)(Tensor*, int*, int, void*, bool);

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

void* concat_arr(const void* arr1, size_t size1, const void* arr2, size_t size2, size_t element_size) {
	if (!arr1 || !arr2 || element_size == 0) {
        fprintf(stderr, "Error (Hello from C!): Invalid input parameters (concat_arr)\n");
        return NULL;
    }
	
	void* result = malloc((size1 + size2) * element_size);
	
	if(!result) {
		fprintf(stderr, "Error (Hello from C!): Memory Allocation Error (concat_arr)\n");
		return NULL;
	}
	
	memcpy(result, arr1, size1 * element_size);
    memcpy((char*)result + size1 * element_size, arr2, size2 * element_size);

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

void set_backend_by_name(Implementation* backends, size_t count, void** target_fn, const char* backend_name) {
    for (size_t i = 0; i < count; i++) {
        if (strcmp(backends[i].name, backend_name) == 0) {
            *target_fn = backends[i].fn;
            return;
        }
    }
}

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

#endif
