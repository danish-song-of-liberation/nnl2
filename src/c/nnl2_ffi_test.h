#include <stdio.h>
#include <stdint.h>

#include "nnl2_core.h"

/** @brief 
 * First test to check the correctness of the (cffi) common lisp binding with c.
 */
uint32_t __nnl2_test_1 (void) {	
	return 0;
}

/** @brief 
 * Second test to check the correctness of the (cffi) common lisp binding with c.
 */
uint32_t __nnl2_test_2 (void) {
	const uint32_t foo = 3;
	const uint32_t bar = 4;
	
	const uint32_t baz = foo + bar;
	
	return baz;
}
