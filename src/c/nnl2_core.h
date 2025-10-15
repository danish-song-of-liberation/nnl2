#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <time.h>

#include "nnl2_log.h"
#include "nnl2_type_backend.h"

#ifndef NNL2_CORE_H
#define NNL2_CORE_H

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline nnl2_int32 nnl2_relu_int32(nnl2_int32 a) {
	return MAX(a, 0);
}

static inline nnl2_float32 nnl2_relu_float32(nnl2_float32 a) {
	return MAX(a, 0.0f);
}

static inline nnl2_float64 nnl2_relu_float64(nnl2_float64 a) {
	return MAX(a, 0.0);
}

static inline void nnl2_relu_int32_inplace(nnl2_int32* a) {
	if(*a < 0) *a = 0;
}

static inline void nnl2_relu_float32_inplace(nnl2_float32* a) {
	if(*a < 0.0f) *a = 0.0f;
}

static inline void nnl2_relu_float64_inplace(nnl2_float64* a) {
	if(*a < 0.0) *a = 0.0;
}

static inline nnl2_int32 nnl2_leaky_relu_int32(nnl2_int32 a, float alpha) {
    if (a > 0) {
        return a;
    } else {
        return (nnl2_int32)(a * alpha);
    }
}

static inline nnl2_float32 nnl2_leaky_relu_float32(nnl2_float32 a, float alpha) {
    if (a > 0.0f) {
        return a;
    } else {
        return a * alpha;  
    }
}

static inline nnl2_float64 nnl2_leaky_relu_float64(nnl2_float64 a, float alpha) {
    if (a > 0.0) {
        return a;
    } else {
        return a * alpha;
    }
}

static inline void nnl2_leaky_relu_int32_inplace(nnl2_int32* a, float alpha) {
    if(*a < 0) {
        float result = (*a * alpha);
        float remainder = fmodf(fabsf(result), 1.0f);
        
        if(remainder > 1e-5f && (1.0f - remainder) > 1e-5f) {
            NNL2_FATAL("Leaky ReLU cannot be applied to the provided tensor");     
            exit(EXIT_FAILURE);
        } else {
            *a = (nnl2_int32)result;
        }
    }
}

static inline void nnl2_leaky_relu_float32_inplace(nnl2_float32* a, float alpha) {
	if(*a < 0.0f) *a = (*a * alpha);
}

static inline void nnl2_leaky_relu_float64_inplace(nnl2_float64* a, float alpha) {
	if(*a < 0.0f) *a = (*a * alpha);
}

static inline nnl2_float32 nnl2_sigmoid_float32(nnl2_float32 x) {
	return 1.0f / (1.0f + expf(-x));
}

static inline nnl2_float64 nnl2_sigmoid_float64(nnl2_float64 x) {
	return 1.0 / (1.0 + exp(-x));
}

static inline void nnl2_sigmoid_float32_inplace(nnl2_float32* x) {
	*x = 1.0f / (1.0f + expf(-(*x)));
}

static inline void nnl2_sigmoid_float64_inplace(nnl2_float64* x) {
	*x = 1.0 / (1.0 + exp(-(*x)));
}

uint32_t __nnl2_test_1(void);
uint32_t __nnl2_test_2(void);

#endif
