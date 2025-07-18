#include <stdio.h>
#include <stdint.h>
#include <cblas.h>

/**
 * @brief Testing openblas work
 */
int main() {
	int32_t vector_size = 4;
	
	float x[] = {1.0, 2.0, 3.0, 4.0};
	float y[] = {5.0, 6.0, 7.0, 8.0};
	
	float dot = cblas_sdot(vector_size, x, 1, y, 1); // (1 * 5) + (2 * 6) + (3 * 6) + (4 * 8) 
	
	return 0;
}
