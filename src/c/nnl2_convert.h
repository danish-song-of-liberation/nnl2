#ifndef NNL2_CONVERT_H
#define NNL2_CONVERT_H

#include <limits.h>
#include <float.h>

/** @brief
 * Converts an arbitrary type value to nnl2_float64
 *
 ** @param value
 * Void pointer to the value to convert
 *
 ** @param dtype 
 * Source data type
 *
 ** @return 
 * Converted value of the float64 type
 *
 ** @note 
 * For unsupported types, it returns NAN and generates a fatal type error
 */
NNL2_FORCE_INLINE static nnl2_float64 nnl2_convert_to_float64(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: return *((nnl2_float64*)value);
		case FLOAT32: return (nnl2_float64)(*((nnl2_float32*)value)); 
		case INT32:   return (nnl2_float64)(*((nnl2_int32*)value)); 
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return NAN;
		}
	}
}

/** @brief
 * Converts an arbitrary type value to float (float32)
 *
 ** @param value
 * Void pointer to the value to convert
 *
 ** @param dtype 
 * Source data type
 *
 ** @return 
 * Converted value of the float32 type
 *
 ** @warning
 * When converting nnl2_float64->float32, it checks for out-of-range float values
 *
 ** @note 
 * For unsupported types, it returns 0.0f and generates a fatal type error
 *
 */
NNL2_FORCE_INLINE static nnl2_float32 nnl2_convert_to_float32(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: {
            nnl2_float64 casted_nnl2_float64 = *((nnl2_float64*)value);
            
			// Checking for overflow of the float range
            if (casted_nnl2_float64 < FLT_MIN || casted_nnl2_float64 > FLT_MAX) {
                NNL2_FATAL("FLOAT64 value out of FLOAT32 range (Point Overflow)");
                return INFINITY;
            }
			
            return (nnl2_float32)casted_nnl2_float64; 
        }
		
		case FLOAT32: return *((nnl2_float32*)value);
		case INT32:   return (nnl2_float32)(*((nnl2_int32*)value)); 
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return 0.0f;
		}
	}
}

/** @brief
 * Converts an arbitrary type value to int32
 *
 ** @param value
 * Void pointer to the value to convert
 *
 ** @param dtype 
 * Source data type
 *
 ** @return 
 * Converted value of the int32 type
 *
 ** @warning 
 * Checks that the value is within the range of nnl2_int32
 *
 ** @warning
 * Checks that fractional numbers do not have a fractional part before conversion
 *
 ** @note 
 * For unsupported types, it returns 0 and generates a fatal error
 */
NNL2_FORCE_INLINE static nnl2_int32 nnl2_convert_to_int32(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: {
			nnl2_float64 casted_nnl2_float64 = *((nnl2_float64*)value);
			
			// Checking that the number does not have a fractional part
			if(casted_nnl2_float64 != trunc(casted_nnl2_float64)) {
				NNL2_FATAL("Cannot convert FLOAT64 to INT32");
				return 0;
			}
			
			// Checking for overflow of the int32 range
			if (casted_nnl2_float64 < INT32_MIN || casted_nnl2_float64 > INT32_MAX) {
                NNL2_FATAL("FLOAT64 value out of INT32 range (Point Overflow)");
                return 0;
            }
			
			return (nnl2_int32)casted_nnl2_float64;
		}
		
		case FLOAT32: {
			nnl2_float32 casted_float = *((nnl2_float32*)value);
			
			// Checking that the number does not have a fractional part
			if(casted_float != truncf(casted_float)) {
				NNL2_FATAL("Cannot convert FLOAT32 to INT32");
				return 0;
			}
			
			// Checking for overflow of the int32 range
			if (casted_float < INT32_MIN || casted_float > INT32_MAX) {
                NNL2_FATAL("FLOAT32 value out of INT32 range");
                return 0;
            }
			
			return (nnl2_int32)casted_float;
		}
		
		case INT32: return *((nnl2_int32*)value);
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return 0;
		}
	}
}

#endif /** NNL2_CONVERT_H **/
