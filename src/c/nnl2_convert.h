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
		case FLOAT64:  return *((nnl2_float64*)value);
		case FLOAT32:  return (nnl2_float64)(*((nnl2_float32*)value)); 
		case FLOAT128: return (nnl2_float64)(*((nnl2_float128*)value));
		case INT8:     return (nnl2_float64)(*((nnl2_int8*)value));
		case INT16:    return (nnl2_float64)(*((nnl2_int16*)value));
		case INT32:    return (nnl2_float64)(*((nnl2_int32*)value)); 
		case INT64:    return (nnl2_float64)(*((nnl2_int64*)value));
		case INT128:   return (nnl2_float64)(*((nnl2_int128*)value));
		case UINT8:    return (nnl2_float64)(*((nnl2_uint8*)value));
		case UINT16:   return (nnl2_float64)(*((nnl2_uint16*)value));
		case UINT32:   return (nnl2_float64)(*((nnl2_uint32*)value));
		case UINT64:   return (nnl2_float64)(*((nnl2_uint64*)value));
		case UINT128:  return (nnl2_float64)(*((nnl2_uint128*)value));
		case BOOL:     return (nnl2_float64)(*((nnl2_bool*)value));
		
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
 * When converting larger types to float32, it checks for out-of-range values
 * and warns about potential precision loss
 *
 ** @note 
 * For unsupported types, it returns 0.0f and generates a fatal type error
 */
NNL2_FORCE_INLINE static nnl2_float32 nnl2_convert_to_float32(void* value, TensorType dtype) {
	switch(dtype) {
		case FLOAT32:  return *((nnl2_float32*)value);
		
		case FLOAT64: {
			nnl2_float64 f64_val = *((nnl2_float64*)value);
			
			// Checking for overflow of the float range
			if (f64_val < -FLT_MAX || f64_val > FLT_MAX) {
				NNL2_FATAL("FLOAT64 value out of FLOAT32 range (Point Overflow)");
				return (f64_val > 0) ? INFINITY : -INFINITY;
			}
			
			// Warning for precision loss with very small values
			if (f64_val != 0.0 && fabs(f64_val) < FLT_MIN) {
				NNL2_WARN("FLOAT64 value may lose precision (underflow) when converting to FLOAT32");
			}
			
			return (nnl2_float32)f64_val; 
		}
		
		case FLOAT128: {
			nnl2_float128 f128_val = *((nnl2_float128*)value);
			nnl2_float64 f64_val = (nnl2_float64)f128_val;
			
			// Use same checks as for FLOAT64
			if (f64_val < -FLT_MAX || f64_val > FLT_MAX) {
				NNL2_FATAL("FLOAT128 value out of FLOAT32 range (Point Overflow)");
				return (f64_val > 0) ? INFINITY : -INFINITY;
			}
			
			return (nnl2_float32)f64_val;
		}
		
		case INT8:     return (nnl2_float32)(*((nnl2_int8*)value));
		case INT16:    return (nnl2_float32)(*((nnl2_int16*)value));
		case INT32:    return (nnl2_float32)(*((nnl2_int32*)value));
		
		case INT64: {
			nnl2_int64 int64_val = *((nnl2_int64*)value);
			
			// Check if value fits exactly in 24-bit mantissa of float32
			if(int64_val < -(1LL << 24) || int64_val > (1LL << 24)) {
				NNL2_WARN("INT64 value may lose precision when converting to FLOAT32");
			}
			
			return (nnl2_float32)int64_val;	
		}
		
		case INT128: {
			nnl2_int128 int128_val = *((nnl2_int128*)value);			
			return (nnl2_float32)((nnl2_float64)int128_val);
		}
		
		case UINT8:    return (nnl2_float32)(*((nnl2_uint8*)value));
		case UINT16:   return (nnl2_float32)(*((nnl2_uint16*)value));
		case UINT32:   return (nnl2_float32)(*((nnl2_uint32*)value));
		
		case UINT64: {
			nnl2_uint64 uint64_val = *((nnl2_uint64*)value);
			
			// Check if value fits exactly in 24-bit mantissa of float32
			if(uint64_val > (1ULL << 24)) {
				NNL2_WARN("UINT64 value may lose precision when converting to FLOAT32");
			}
			
			return (nnl2_float32)uint64_val;
		}
		
		case UINT128: {
			nnl2_uint128 uint128_val = *((nnl2_uint128*)value);
			
			// 128-bit integers can't be represented exactly in float32
			if(uint128_val != 0) {
				NNL2_WARN("UINT128 value will lose precision when converting to FLOAT32");
			}
			
			return (nnl2_float32)((nnl2_float64)uint128_val);
		}
		
		case BOOL:     return (nnl2_float32)(*((nnl2_bool*)value));
		
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
		case INT8:    return (nnl2_int32)(*((nnl2_int8*)value));
		case INT16:   return (nnl2_int32)(*((nnl2_int16*)value));
		case INT32:   return *((nnl2_int32*)value);
		
		case INT64: {
			nnl2_int64 int64_val = *((nnl2_int64*)value);
			
			// Checking for overflow of the int32 range
			if (int64_val < INT32_MIN || int64_val > INT32_MAX) {
				NNL2_FATAL("INT64 value out of INT32 range");
				return 0;
			}
			
			return (nnl2_int32)int64_val;
		}
		
		case INT128: {
			nnl2_int128 int128_val = *((nnl2_int128*)value);
			
			// Checking for overflow of the int32 range
			if (int128_val < INT32_MIN || int128_val > INT32_MAX) {
				NNL2_FATAL("INT128 value out of INT32 range");
				return 0;
			}
			
			return (nnl2_int32)int128_val;
		}
		
		case UINT8:   return (nnl2_int32)(*((nnl2_uint8*)value));
		case UINT16:  return (nnl2_int32)(*((nnl2_uint16*)value));
		case UINT32: {
			nnl2_uint32 uint32_val = *((nnl2_uint32*)value);
			
			// Checking for overflow of the int32 range
			if (uint32_val > INT32_MAX) {
				NNL2_FATAL("UINT32 value out of INT32 range");
				return 0;
			}
			
			return (nnl2_int32)uint32_val;
		}
		
		case UINT64: {
			nnl2_uint64 uint64_val = *((nnl2_uint64*)value);
			
			// Checking for overflow of the int32 range
			if (uint64_val > INT32_MAX) {
				NNL2_FATAL("UINT64 value out of INT32 range");
				return 0;
			}
			
			return (nnl2_int32)uint64_val;
		}
		
		case UINT128: {
			nnl2_uint128 uint128_val = *((nnl2_uint128*)value);
			
			// Checking for overflow of the int32 range
			if (uint128_val > INT32_MAX) {
				NNL2_FATAL("UINT128 value out of INT32 range");
				return 0;
			}
			
			return (nnl2_int32)uint128_val;
		}
		
		case FLOAT32: {
			nnl2_float32 float_val = *((nnl2_float32*)value);
			
			if(float_val != truncf(float_val)) {
				NNL2_FATAL("Cannot convert FLOAT32 to INT32: fractional part present");
				return 0;
			}
			
			if (float_val < INT32_MIN || float_val > INT32_MAX) {
				NNL2_FATAL("FLOAT32 value out of INT32 range");
				return 0;
			}
			
			if (!isfinite(float_val)) {
				NNL2_FATAL("Cannot convert non-finite FLOAT32 to INT32");
				return 0;
			}
			
			return (nnl2_int32)float_val;
		}
		
		case FLOAT64: {
			nnl2_float64 float_val = *((nnl2_float64*)value);

			if(float_val != trunc(float_val)) {
				NNL2_FATAL("Cannot convert FLOAT64 to INT32: fractional part present");
				return 0;
			}
			
			if (float_val < INT32_MIN || float_val > INT32_MAX) {
				NNL2_FATAL("FLOAT64 value out of INT32 range");
				return 0;
			}
			
			if (!isfinite(float_val)) {
				NNL2_FATAL("Cannot convert non-finite FLOAT64 to INT32");
				return 0;
			}
			
			return (nnl2_int32)float_val;
		}
		
		case FLOAT128: {
			nnl2_float128 float_val = *((nnl2_float128*)value);
			
			#if defined(NNL2_USE_FLOAT128_QUADMATH)
				if(float_val != truncq(float_val)) {
					NNL2_FATAL("Cannot convert FLOAT128 to INT32. fractional part present");
					return 0;
				}
			#else
				// Fallback: convert to double first
				nnl2_float64 dbl_val = (nnl2_float64)float_val;
				if(dbl_val != trunc(dbl_val)) {
					NNL2_FATAL("Cannot convert FLOAT128 to INT32. fractional part present");
					return 0;
				}
			#endif
			
			// Convert to double for range checking
			nnl2_float64 dbl_val = (nnl2_float64)float_val;
			
			// Checking for overflow of the int32 range
			if (dbl_val < INT32_MIN || dbl_val > INT32_MAX) {
				NNL2_FATAL("FLOAT128 value out of INT32 range");
				return 0;
			}
			
			if(!isfinite(dbl_val)) {
				NNL2_FATAL("Cannot convert non-finite FLOAT128 to INT32");
				return 0;
			}
			
			return (nnl2_int32)dbl_val;
		}
		
		case BOOL:    return (nnl2_int32)(*((nnl2_bool*)value));
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return 0;
		}
	}
}

/** @brief
 * Converts an arbitrary type value to int64
 *
 ** @param value
 * Void pointer to the value to convert
 *
 ** @param dtype 
 * Source data type
 *
 ** @return 
 * Converted value of the int64 type
 *
 ** @warning 
 * Checks that the value is within the range of nnl2_int64
 *
 ** @warning
 * Checks that fractional numbers do not have a fractional part before conversion
 *
 ** @note 
 * For unsupported types, it returns 0 and generates a fatal error
 */
NNL2_FORCE_INLINE static nnl2_int64 nnl2_convert_to_int64(void* value, TensorType dtype) {
	switch(dtype) {
		case INT8:     return (nnl2_int64)(*((nnl2_int8*)value));
		case INT16:    return (nnl2_int64)(*((nnl2_int16*)value));
		case INT32:    return (nnl2_int64)(*((nnl2_int32*)value));
		case INT64:    return *((nnl2_int64*)value);
		
		case INT128: {
			nnl2_int128 int128_val = *((nnl2_int128*)value);
			
			// Checking for overflow of the int64 range
			if (int128_val < INT64_MIN || int128_val > INT64_MAX) {
				NNL2_FATAL("INT128 value out of INT64 range");
				return 0;
			}
			
			return (nnl2_int64)int128_val;
		}
		
		case UINT8:    return (nnl2_int64)(*((nnl2_uint8*)value));
		case UINT16:   return (nnl2_int64)(*((nnl2_uint16*)value));
		case UINT32:   return (nnl2_int64)(*((nnl2_uint32*)value));
		
		case UINT64: {
			nnl2_uint64 uint64_val = *((nnl2_uint64*)value);
			
			// Checking for overflow of the int64 range
			if (uint64_val > INT64_MAX) {
				NNL2_FATAL("UINT64 value out of INT64 range");
				return 0;
			}
			
			return (nnl2_int64)uint64_val;
		}
		
		case UINT128: {
			nnl2_uint128 uint128_val = *((nnl2_uint128*)value);
			
			// Checking for overflow of the int64 range
			if (uint128_val > INT64_MAX) {
				NNL2_FATAL("UINT128 value out of INT64 range");
				return 0;
			}
			
			return (nnl2_int64)uint128_val;
		}
		
		case FLOAT32: {
			nnl2_float32 float_val = *((nnl2_float32*)value);
			
			// Checking that the number does not have a fractional part
			if(float_val != truncf(float_val)) {
				NNL2_FATAL("Cannot convert FLOAT32 to INT64. fractional part present");
				return 0;
			}
			
			// Checking for overflow of the int64 range
			if (float_val < INT64_MIN || float_val > INT64_MAX) {
				NNL2_FATAL("FLOAT32 value out of INT64 range");
				return 0;
			}
			
			// Check for NaN or infinity
			if (!isfinite(float_val)) {
				NNL2_FATAL("Cannot convert non-finite FLOAT32 to INT64");
				return 0;
			}
			
			return (nnl2_int64)float_val;
		}
		
		case FLOAT64: {
			nnl2_float64 float_val = *((nnl2_float64*)value);
			
			if(float_val != trunc(float_val)) {
				NNL2_FATAL("Cannot convert FLOAT64 to INT64. fractional part present");
				return 0;
			}
			
			if (float_val < INT64_MIN || float_val > INT64_MAX) {
				NNL2_FATAL("FLOAT64 value out of INT64 range");
				return 0;
			}
			
			if (!isfinite(float_val)) {
				NNL2_FATAL("Cannot convert non-finite FLOAT64 to INT64");
				return 0;
			}
			
			return (nnl2_int64)float_val;
		}
		
		case FLOAT128: {
			nnl2_float128 float_val = *((nnl2_float128*)value);
			
			#if defined(NNL2_USE_FLOAT128_QUADMATH)
				if(float_val != truncq(float_val)) {
					NNL2_FATAL("Cannot convert FLOAT128 to INT64. fractional part present");
					return 0;
				}
			#else
				nnl2_float64 dbl_val = (nnl2_float64)float_val;
				if(dbl_val != truncq(dbl_val)) {
					NNL2_FATAL("Cannot convert FLOAT128 to INT64. fractional part present");
					return 0;
				}
			#endif
			
			// Convert to double for range checking
			nnl2_float64 dbl_val = (nnl2_float64)float_val;
			
			if (dbl_val < INT64_MIN || dbl_val > INT64_MAX) {
				NNL2_FATAL("FLOAT128 value out of INT64 range");
				return 0;
			}
			
			if (!isfinite(dbl_val)) {
				NNL2_FATAL("Cannot convert non-finite FLOAT128 to INT64");
				return 0;
			}
			
			return (nnl2_int64)dbl_val;
		}
		
		case BOOL:     return (nnl2_int64)(*((nnl2_bool*)value));
		
		default: {
			NNL2_TYPE_FATAL(dtype); // Fatal error for unsupported types
			return 0;
		}
	}
}

#endif /** NNL2_CONVERT_H **/
