#ifndef NNL2_CONVERT_H
#define NNL2_CONVERT_H

#include <limits.h>
#include <float.h>

/** @brief
 * Converts an arbitrary type value to double (float64)
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
NNL2_FORCE_INLINE static double nnl2_convert_to_float64(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: return *((double*)value);
		case FLOAT32: return (double)(*((float*)value)); 
		case INT32:   return (double)(*((int32_t*)value)); 
		
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
 * When converting double->float, it checks for out-of-range float values
 *
 ** @note 
 * For unsupported types, it returns 0.0f and generates a fatal type error
 *
 */
NNL2_FORCE_INLINE static float nnl2_convert_to_float32(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: {
            double casted_double = *((double*)value);
            
			// Checking for overflow of the float range
            if (casted_double < FLT_MIN || casted_double > FLT_MAX) {
                NNL2_FATAL("FLOAT64 value out of FLOAT32 range (Point Overflow)");
                return INFINITY;
            }
			
            return (float)casted_double; 
        }
		
		case FLOAT32: return *((float*)value);
		case INT32:   return (float)(*((int32_t*)value)); 
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return 0.0f;
		}
	}
}

/** @brief
 * Converts an arbitrary type value to int (int32)
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
 * Checks that the value is within the range of int32_t
 *
 ** @warning
 * Checks that fractional numbers do not have a fractional part before conversion
 *
 ** @note 
 * For unsupported types, it returns 0 and generates a fatal error
 */
NNL2_FORCE_INLINE static int32_t nnl2_convert_to_int32(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT64: {
			double casted_double = *((double*)value);
			
			// Checking that the number does not have a fractional part
			if(casted_double != trunc(casted_double)) {
				NNL2_FATAL("Cannot convert FLOAT64 to INT32");
				return 0;
			}
			
			// Checking for overflow of the int32 range
			if (casted_double < INT32_MIN || casted_double > INT32_MAX) {
                NNL2_FATAL("FLOAT64 value out of INT32 range (Point Overflow)");
                return 0;
            }
			
			return (int32_t)casted_double;
		}
		
		case FLOAT32: {
			float casted_float = *((float*)value);
			
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
			
			return (int32_t)casted_float;
		}
		
		case INT32: return *((int32_t*)value);
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return 0;
		}
	}
}

#endif /** NNL2_CONVERT_H **/
