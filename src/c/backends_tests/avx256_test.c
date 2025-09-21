#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

/**
 * @brief Adding two vectors using AVX256 (8 elements)
 */
int main() {
    float x[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float y[8] = {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
	
    float result[8] = {0};
    
    __m256 vec_x = _mm256_loadu_ps(x);
    __m256 vec_y = _mm256_loadu_ps(y);

    __m256 vec_result = _mm256_add_ps(vec_x, vec_y);

    _mm256_storeu_ps(result, vec_result);
	
    return 0;
}
