#include <stdint.h>

#ifndef NNL2_CORE_H
#define NNL2_CORE_H

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static inline int32_t nnl2_relu_int32(int32_t a) {
	return MAX(a, 0);
}

static inline float nnl2_relu_float32(float a) {
	return MAX(a, 0.0f);
}

static inline double nnl2_relu_float64(double a) {
	return MAX(a, 0.0);
}

static inline void nnl2_relu_int32_inplace(int32_t* a) {
	if(*a < 0) *a = 0;
}

static inline void nnl2_relu_float32_inplace(float* a) {
	if(*a < 0.0f) *a = 0.0f;
}

static inline void nnl2_relu_float64_inplace(double* a) {
	if(*a < 0.0) *a = 0.0;
}

static inline int32_t nnl2_leaky_relu_int32(int32_t a, float alpha) {
    if (a > 0) {
        return a;
    } else {
        return (int32_t)(a * alpha);
    }
}

static inline float nnl2_leaky_relu_float32(float a, float alpha) {
    if (a > 0.0f) {
        return a;
    } else {
        return a * alpha;
    }
}

static inline double nnl2_leaky_relu_float64(double a, float alpha) {
    if (a > 0.0) {
        return a;
    } else {
        return a * alpha;
    }
}

static inline void nnl2_leaky_relu_int32_inplace(int32_t* a, float alpha) {
	if(*a < 0) *a = (int32_t)(*a * alpha);
}

static inline void nnl2_leaky_relu_float32_inplace(float* a, float alpha) {
	if(*a < 0.0f) *a = (*a * alpha);
}

static inline void nnl2_leaky_relu_float64_inplace(double* a, float alpha) {
	if(*a < 0.0f) *a = (*a * alpha);
}

uint32_t __nnl2_test_1(void);
uint32_t __nnl2_test_2(void);

#endif
