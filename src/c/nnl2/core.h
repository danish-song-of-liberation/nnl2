#ifndef NNL2_CORE_INCLUDE_H
#define NNL2_CORE_INCLUDE_H

#include "../nnl2_tensor_backend.h"
#include "../nnl2_config.h"

#ifdef NNL2_PTHREAD_AVAILABLE
	#include "../nnl2_parallel_backend.h"
	#include <pthread.h>
#endif

#include "../nnl2_convert.h"
#include "../nnl2_product.h"
#include "../nnl2_ffi_accessors.h"
#include "../nnl2_foreign_shape.h"
#include "../nnl2_rng.h"

#endif /** NNL2_CORE_INCLUDE_H **/
