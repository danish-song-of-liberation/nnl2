#include <stdint.h>

/**
 * @brief Test to check openblas0330woa64static status (0 if available else 1)
 */
int32_t openblas0330woa64static_status(void) {
	return 
	#ifdef OPENBLAS0330WOA64STATIC_AVAILABLE
	    0;
	#else
		1;
	#endif
}
	