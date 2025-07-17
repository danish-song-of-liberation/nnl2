#include "nnl2_core.h"
#include "nnl2_ffi_test.h"
#include "backends_status/nnl2_status.h"
#include "nnl2_tensor_core.h"
 
void init_system() {
	init_inplace_fill();
	init_empty();
	init_zeros();
	init_ones();
	init_sgemminplace();
	init_dgemminplace();
	init_sgemm();
	init_dgemm();
} 
 