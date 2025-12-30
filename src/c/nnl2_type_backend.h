#ifndef NNL2_TYPE_BACKEND
#define NNL2_TYPE_BACKEND

///@{

/** @typedef nnl2_bool
 ** @brief Boolean type (true/false)
 ** @details Equivalent to C99 _Bool or C++ bool
 **/
#ifdef __cplusplus
    typedef bool nnl2_bool;
#else
    #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
        typedef _Bool nnl2_bool;
    #else
        typedef unsigned char nnl2_bool;
    #endif
#endif

/** @typedef nnl2_int8
 ** @brief 8-bit signed integer
 ** @details Always exactly 8 bits, uses C99 int8_t when available
 **/
typedef int8_t nnl2_int8;

/** @typedef nnl2_uint8
 ** @brief 8-bit unsigned integer
 ** @details Always exactly 8 bits, uses C99 uint8_t when available
 **/
typedef uint8_t nnl2_uint8;

/** @typedef nnl2_int16
 ** @brief 16-bit signed integer
 ** @details Always exactly 16 bits, uses C99 int16_t when available
 **/
typedef int16_t nnl2_int16;

/** @typedef nnl2_uint16
 ** @brief 16-bit unsigned integer
 ** @details Always exactly 16 bits, uses C99 uint16_t when available
 **/
typedef uint16_t nnl2_uint16;

/** @typedef nnl2_int32
 ** @brief 32-bit signed integer
 ** @details Always exactly 32 bits, uses C99 int32_t when available
 **/
typedef int32_t nnl2_int32;

/** @typedef nnl2_uint32
 ** @brief 32-bit unsigned integer
 ** @details Always exactly 32 bits, uses C99 uint32_t when available
 **/
typedef uint32_t nnl2_uint32;

/** @typedef nnl2_int64
 ** @brief 64-bit signed integer
 ** @details Always exactly 64 bits, uses C99 int64_t when available
 **/
typedef int64_t nnl2_int64;

/** @typedef nnl2_uint64
 ** @brief 64-bit unsigned integer
 ** @details Always exactly 64 bits, uses C99 uint64_t when available
 **/
typedef uint64_t nnl2_uint64;

/** @typedef nnl2_float64
 ** @brief 64-bit double-precision floating-point number
 **/
typedef double nnl2_float64;

/** @typedef nnl2_float32 
 ** @brief 32-bit single-precision floating-point number
 **/
typedef float nnl2_float32;

/** @typedef nnl2_float16
 ** @brief 16-bit half-precision floating-point (when available)
 **/
#if defined(__clang__) && defined(__ARM_FP16_FORMAT_IEEE)
    typedef __fp16 nnl2_float16;
    #define NNL2_HALF_SUPPORTED 1
#elif defined(__GNUC__) && defined(__FLOAT16__)
    typedef _Float16 nnl2_float16;
    #define NNL2_HALF_SUPPORTED 1
#else
    typedef uint16_t nnl2_float16;  // Raw storage
    #define NNL2_HALF_SUPPORTED 0
#endif

/** @typedef nnl2_int32
 ** @brief 32-bit signed integer
 **/
typedef int32_t nnl2_int32;

/** @typedef nnl2_int64
 ** @brief 64-bit signed integer
 **/
typedef int64_t nnl2_int64;
	
/** @typedef nnl2_int128/nnl2_uint128
 ** @brief 128-bit signed/unsigned integer 
 ** @warning On non-GCC/Clang compilers, this may be only 64-bit despite the name.
 **/ 
#if defined(__GNUC__) || defined(__clang__) || defined(__ICC) || defined(__INTEL_COMPILER)
    #if defined(__SIZEOF_INT128__) || (defined(__GNUC__) && __GNUC__ >= 4)
        typedef __int128 nnl2_int128;
        typedef unsigned __int128 nnl2_uint128;
        #define NNL2_INT128_SUPPORTED 1
        #define NNL2_UINT128_SUPPORTED 1
    #else
        #define NNL2_INT128_SUPPORTED 0
        #define NNL2_UINT128_SUPPORTED 0
    #endif
#else
	typedef long long int nnl2_int128;
	typedef unsigned long long int nnl2_uint128;
    #define NNL2_INT128_SUPPORTED 0
    #define NNL2_UINT128_SUPPORTED 0
#endif

/** @typedef nnl2_float128
 ** @brief 128-bit quadruple-precision floating-point number 
 **/
#if defined(__GNUC__) || defined(__clang__) || defined(__ICC) || defined(__INTEL_COMPILER)
    #if defined(__SIZEOF_FLOAT128__) || \
        (defined(__GNUC__) && __GNUC__ >= 7) || \
        (defined(__clang__) && __clang_major__ >= 6)
        
        #if defined(__FLT128_MAX__) && !defined(__STRICT_ANSI__)
            typedef __float128 nnl2_float128;
            #define NNL2_FLOAT128_SUPPORTED 1
        #elif defined(_Float128)
            typedef _Float128 nnl2_float128;
            #define NNL2_FLOAT128_SUPPORTED 1
        #else
            #define NNL2_FLOAT128_SUPPORTED 0
        #endif
        
    #elif defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
        typedef __float128 nnl2_float128;
        #define NNL2_FLOAT128_SUPPORTED 1
        
    #else
        #define NNL2_FLOAT128_SUPPORTED 0
    #endif
#else
    #define NNL2_FLOAT128_SUPPORTED 0
#endif

#if !NNL2_FLOAT128_SUPPORTED && !defined(nnl2_float128)
    typedef long double nnl2_float128;
#endif

///@}

#endif /** NNL2_TYPE_BACKEND **/
