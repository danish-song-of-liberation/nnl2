#include <stdio.h>
#include <stdint.h>
#include <immintrin.h> 

/**
 * @brief Adding two vectors using AVX128
 */
int main() {
    int32_t vector_size = 4;
    
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[4] = {0};
    
    __m128 vec_x = _mm_loadu_ps(x);
    __m128 vec_y = _mm_loadu_ps(y);
    
    __m128 vec_result = _mm_add_ps(vec_x, vec_y);
    
    _mm_storeu_ps(result, vec_result);
    
    return 0;
}
