#ifndef NNL2_RNG_H
#define NNL2_RNG_H

#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

/** @file nnl2_rng.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief High-performance random number generator with xorshift128+ 
 *
 * Invention of the bicycle. 
 * 2.5 times slower than rand, but I'm sorry to delete it
 *
 ** Filepath: src/c/nnl2_rng.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2	
 **/
 
 
 
///@{ [macros]
 
/** @brief Maximum value for 32-bit unsigned integer **/ 
#define NNL2_RNG_U32_MAX 0xFFFFFFFFU

/** @brief Maximum value for 32-bit signed integer (compatible with rand()) **/
#define NNL2_RNG_I32_MAX 0x7FFFFFFF

/** @brief Inverse of 2^32 for float conversion (1.0 / 4294967296.0) **/
#define NNL2_RNG_FLOAT_DIV 2.3283064365386963e-10f

/** @brief Inverse of 2^64 for double conversion (1.0 / 18446744073709551616.0) **/
#define NNL2_RNG_DOUBLE_DIV 5.42101086242752217e-20

///@{ [nnl2_xs128p_init]

/** @brief Constant for xorshift128+ state initialization (sqrt(2) in hex) **/
#define NNL2_XS128P_INIT_A 0x6A09E667F3BCC909ULL
#define NNL2_XS128P_INIT_B 0xBB67AE8584CAA73BULL

///@} [nnl2_xs128p_init]

/** @brief Number of warmup iterations for initialization **/
#define NNL2_RNG_WARMUP_ITERATIONS 10 

/** @brief TLS configuration **/
#if defined(NNL2_PTHREAD_AVAILABLE) && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define NNL2_USE_THREAD_LOCAL 1
#elif defined(_MSC_VER)
    #define NNL2_USE_THREAD_LOCAL 1
#elif defined(__GNUC__) || defined(__clang__)
    #define NNL2_USE_THREAD_LOCAL 1
#else
    #define NNL2_USE_THREAD_LOCAL 0
#endif

/** @brief Default seed mixing value for additional entropy **/
#define NNL2_SEED_MIXER 0x9E3779B97F4A7C15ULL

/** @brief Maximum number of threads for fallback **/
#define NNL2_MAX_THREADS 256

/** @brief Buffer size for bulk random generation **/
#define NNL2_BUFFER_SIZE 1024

///@} [macros]



///@{ [structures]

/** @struct nnl2_rng_state
 ** @brief 
 * Xorshift128+ random number generator state
 */
typedef struct {
    uint64_t s[2];      ///< Generator state
    uint32_t counter;   ///< Number of generated values 
} nnl2_rng_state;

///@} [structures]



///@{ [core]

#if NNL2_USE_THREAD_LOCAL && defined(NNL2_PTHREAD_AVAILABLE)

/** @brief Thread-local storage declaration **/
#ifdef _MSC_VER
    static __declspec(thread) nnl2_rng_state tls_rng_state = {0};
    static __declspec(thread) int tls_rng_initialized = 0;
#else
    static _Thread_local nnl2_rng_state tls_rng_state = {0};
    static _Thread_local int tls_rng_initialized = 0;
#endif

/** @brief Global atomic counter for unique thread seeds **/
static volatile uint32_t global_thread_counter = 0;

/** @brief 
 * Generate a unique seed for each thread
 * 
 ** @return uint64_t 
 * High-quality unique seed
 */
static uint64_t generate_thread_seed(void) {
    uint64_t seed = (uint64_t)time(NULL);
    
    // Mix with global counter 
    uint32_t counter = __sync_fetch_and_add(&global_thread_counter, 1);
    seed ^= ((uint64_t)counter << 32);
    
    // Mix with thread identifier 
    seed ^= (uint64_t)pthread_self();
    seed ^= ((uint64_t)getpid() << 16);
    
    // Final mixing 
    seed ^= NNL2_SEED_MIXER;
    seed ^= (uint64_t)((uintptr_t)&tls_rng_state);
    
    return seed;
}

/** @brief 
 * Initialize thread-local RNG state
 * 
 ** @param state 
 * Pointer to RNG state structure
 *
 ** @param seed 
 * Seed value (0 for automatic generation)
 */
static void init_thread_rng(nnl2_rng_state* state, uint64_t seed) {
    if(seed == 0) 
        seed = generate_thread_seed();
    
    // Initialize state with mixing constants 
    state->s[0] = seed ^ NNL2_XS128P_INIT_A;
    state->s[1] = seed ^ NNL2_XS128P_INIT_B;
    state->counter = 0;
    
    // Warmup generator to avoid correlation 
    for(int i = 0; i < NNL2_RNG_WARMUP_ITERATIONS; i++) {
        uint64_t x = state->s[0];
        uint64_t const y = state->s[1];
        state->s[0] = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        state->s[1] = x;
        (void)(x + y); // Discard result 
    }
}

/** @brief 
 * Get thread-local RNG state 
 * 
 ** @return nnl2_rng_state* 
 * Pointer to thread's RNG state
 */
static nnl2_rng_state* get_thread_rng(void) {
    if(!tls_rng_initialized) {
        init_thread_rng(&tls_rng_state, 0);
        tls_rng_initialized = 1;
    }
	
    return &tls_rng_state;
}

#else /** NNL2_USE_THREAD_LOCAL && defined(NNL2_PTHREAD_AVAILABLE) **/

/// Fallback
/// Muted-protected global generator

/** @brief Global generator protected by mutex **/
static nnl2_rng_state global_rng_state = {0};
static pthread_mutex_t rng_mutex = PTHREAD_MUTEX_INITIALIZER;
static int global_rng_initialized = 0;

/** @brief 
 * Initialize global RNG state
 * 
 ** @param seed 
 * Seed value (0 for automatic generation)
 */
static void init_global_rng(uint64_t seed) {
    if(seed == 0) {
        seed = (uint64_t)time(NULL);
        seed ^= NNL2_SEED_MIXER;
        seed ^= (uint64_t)getpid();
    }
        
    global_rng_state.s[0] = seed ^ NNL2_XS128P_INIT_A;
    global_rng_state.s[1] = seed ^ NNL2_XS128P_INIT_B;
    global_rng_state.counter = 0;
        
    /// Warmup generator 
    for(int i = 0; i < NNL2_RNG_WARMUP_ITERATIONS; i++) {
        uint64_t x = global_rng_state.s[0];
        uint64_t const y = global_rng_state.s[1];
        global_rng_state.s[0] = y;
        x ^= x << 23;
        x ^= x >> 17;
        x ^= y ^ (y >> 26);
        global_rng_state.s[1] = x;
    }
        
    global_rng_initialized = 1;
}

/** @brief 
 * Get global RNG state 
 * 
 ** @return nnl2_rng_state* 
 * Pointer to global RNG state
 */
static nnl2_rng_state* get_global_rng(void) {
    pthread_mutex_lock(&rng_mutex);
        
    if(!global_rng_initialized) 
        init_global_rng(0);
        
    pthread_mutex_unlock(&rng_mutex);
    return &global_rng_state;
}
	
#endif /** NNL2_USE_THREAD_LOCAL && defined(NNL2_PTHREAD_AVAILABLE) **/

/** @brief 
 * Xorshift128+ algorithm core 
 * 
 ** @param state 
 * RNG state to update
 *
 ** @return uint64_t 
 * Next random 64-bit value
 */
static inline uint64_t xorshift128plus_core(nnl2_rng_state* state) {
    uint64_t x = state->s[0];
    uint64_t const y = state->s[1];
    
    // State updating 
    state->s[0] = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    state->s[1] = x;
    
    state->counter++;
    return x + y;
}
 
///@} [core]



///@{ [api]

uint64_t nnl2_random_u64() {
	#if NNL2_USE_THREAD_LOCAL && defined(NNL2_PTHREAD_AVAILABLE)
		nnl2_rng_state* state = get_thread_rng();
	#else
		nnl2_rng_state* state = get_global_rng();
		pthread_mutex_lock(&rng_mutex);
	#endif
    
    uint64_t result = xorshift128plus_core(state);
    
	#if !(NNL2_USE_THREAD_LOCAL && defined(NNL2_PTHREAD_AVAILABLE))
		pthread_mutex_unlock(&rng_mutex);
	#endif
    
    return result;
}

uint32_t nnl2_random_u32(void) {
    return (uint32_t)(nnl2_random_u64() >> 32);
}

int32_t nnl2_random_i32(void) {
    return (int32_t)(nnl2_random_u64() & NNL2_RNG_I32_MAX);
}

float nnl2_random_float(void) {
    uint32_t r = nnl2_random_u32();
    return (float)r * NNL2_RNG_FLOAT_DIV;
}

double nnl2_random_double(void) {
    uint64_t r = nnl2_random_u64();
    return (double)r * NNL2_RNG_DOUBLE_DIV;
}

int32_t nnl2_random_range(int32_t min, int32_t max) {
    if(min > max) {
        int32_t tmp = min;
        min = max;
        max = tmp;
    }
    
    if(min == max)  return min;
    
    uint64_t range = (uint64_t)max - (uint64_t)min + 1;

    uint64_t limit = NNL2_RNG_U32_MAX - (NNL2_RNG_U32_MAX % range);
    uint64_t r;
    
    do {
        r = nnl2_random_u32();
    } while (r >= limit);
    
    return min + (int32_t)(r % range);
}

void nnl2_random_bytes(void* buffer, size_t size) {
    uint8_t* buf = (uint8_t*)buffer;

    while (size >= 8) {
        uint64_t r = nnl2_random_u64();
        memcpy(buf, &r, 8);
        buf += 8;
        size -= 8;
    }

    if (size > 0) {
        uint64_t r = nnl2_random_u64();
        memcpy(buf, &r, size);
    }
}

void nnl2_rng_init(uint64_t seed) {
	#if NNL2_USE_THREAD_LOCAL && defined(NNL2_PTHREAD_AVAILABLE)
		nnl2_rng_state* state = &tls_rng_state;
		init_thread_rng(state, seed);
		tls_rng_initialized = 1;
	#else
		pthread_mutex_lock(&rng_mutex);
		init_global_rng(seed);
		pthread_mutex_unlock(&rng_mutex);
	#endif
}

///@} [api]

#endif /** NNL2_RNG_H **/
