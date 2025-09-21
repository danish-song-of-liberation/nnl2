#include <stdio.h>
#include <stdint.h>
#include <immintrin.h> 

/**
 * @brief Adding two vectors using AVX512
 */
int main() {
    int32_t vector_size = 16; 
    
    float x[16] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float y[16] = {17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f};
	
    float result[16] = {0};
    
    __m512 vec_x = _mm512_loadu_ps(x);
    __m512 vec_y = _mm512_loadu_ps(y);

    __m512 vec_result = _mm512_add_ps(vec_x, vec_y);

    _mm512_storeu_ps(result, vec_result);
    
    return 0;
}
