#ifndef NNL2_AD_TOPO_H
#define NNL2_AD_TOPO_H

/** @file nnl2_ad_topo.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Topological sort implementation for automatic differentiation computational graphs
 **/

// For self-sufficiency of the file
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h> 

/** @brief Prefetches data at the specified address into the cache for topological sort **/
#if defined(__GNUC__) || defined(__clang__)
#  define NNL2_HAVE_TOPO_PREFETCH 1
#  define NNL2_TOPO_PREFETCH(addr) __builtin_prefetch((addr))
#else
#  define NNL2_HAVE_TOPO_PREFETCH 0
#  define NNL2_TOPO_PREFETCH(addr) ((void)0)
#endif

/** @brief Thread-local generation counter for topology tracking 
 ** @note Initialized to 1 by default **/
#if defined(__STDC_NO_THREADS__) && !defined(_MSC_VER)
static uint64_t nnl2_ad_current_gen = 1;
#else
static _Thread_local uint64_t nnl2_ad_current_gen = 1;
#endif

///@{
	
/** @brief 
 * Arena allocation and topological 
 * sort configuration constants 
 **/

#define NNL2_ARENA_GROWTH_THRESHOLD 16384        ///< Threshold for switching from 2x to 1.5x growth factor 
#define NNL2_TOPO_INITIAL_ARENA_CAPACITY 4096    ///< Initial arena capacity for topological sort operations 
#define NNL2_ARENA_INITIAL_CAPACITY 64			 ///< Default initial capacity for arena allocator 
#define NNL2_TOPO_INITIAL_STACK_CAPACITY 64		 ///< Initial stack capacity for DFS in topological sort 
#define NNL2_TOPO_INITIAL_TOPO_CAPACITY 64		 ///< Initial buffer capacity for topological ordering 

///@}
	
	
	
///@{ [nnl2_arena_t]

/** @struct nnl2_arena_t
 ** @brief Arena allocator structure for efficient memory management in topological sort **/
typedef struct {
    unsigned char *buf;    ///< Pointer to the arena's memory buffer
    size_t size;		   ///< Current used size of the arena in bytes
    size_t capacity;	   ///< Total capacity of the arena in bytes
} nnl2_arena_t;

///@} [nnl2_arena_t]



///@{ [nnl2_stack_frame]

/** @struct nnl2_stack_frame
 ** @brief Stack frame structure for tracking tensor traversal state **/
typedef struct {
    struct nnl2_ad_tensor *tensor;	  ///< Pointer to the tensor being processed
    size_t index;					  ///< Current index in the tensor's dependency list
} nnl2_stack_frame;

///@} [nnl2_stack_frame]

/** @brief 
 * Initializes an arena allocator with the specified initial capacity
 *
 ** @param a 
 * Pointer to the arena structure to initialize
 *
 ** @param initial_cap 
 * Initial capacity in bytes for the arena buffer
 * 
 ** @return 
 * true if initialization succeeded, false if memory allocation failed
 *
 ** @exception NNL2Error
 * If memory allocation of initial_cap fails 
 *
 ** @exception NNL2Error [nnl2_safety_max+]
 * If a (input pointer) is NULL
 */
static inline bool nnl2_arena_init(nnl2_arena_t *a, size_t initial_cap) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(a, "In function nnl2_arena_init, nnl2_arena_t *a is NULL", false);
	#endif
	
    a->buf = malloc(initial_cap);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!a->buf) {
			NNL2_MALLOC_ERROR();
			return false;
		}
	#endif
	
    a->size = 0;
    a->capacity = initial_cap;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
	
    return true;
}

/** @brief 
 * Releases all resources associated with an arena allocator
 *
 ** @param a 
 * Pointer to the arena structure to free
 *
 ** @exception NNL2Error [nnl2_safety_max+]
 * If a (input pointer) is NULL
 */ 
static inline void nnl2_arena_free(nnl2_arena_t *a) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(a, "In function nnl2_arena_free, nnl2_arena_t *a is NULL");
	#endif
	
    free(a->buf);
    a->buf = NULL;
    a->size = a->capacity = 0;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief
 * Ensures the arena has enough capacity for an additional n bytes
 *
 ** @param a 
 * Pointer to the arena structure
 *
 ** @param n 
 * Number of additional bytes required
 *
 ** @return 
 * true if the arena has or successfully obtained sufficient 
 * capacity, false if memory reallocation failed
 *
 ** @note 
 * Initial capacity is 64 bytes if arena was zero-initialized
 *
 ** @exception NNL2Error [nnl2_safety_max+]
 * If a (input pointer) is NULL
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If failed to malloc new buffer
 *
 ** @see NNL2_ARENA_INITIAL_CAPACITY 64
 ** @see NNL2_ARENA_GROWTH_THRESHOLD 16384
 **/
static inline bool nnl2_arena_ensure(nnl2_arena_t *a, size_t n) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(a, "In function nnl2_arena_free, nnl2_arena_t *a is NULL", false);
	#endif
	
	// current usage + requested bytes
    size_t needed = a->size + n;
	
	// Fast path. If already have enough capacity, return immediately
    if(needed <= a->capacity) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
			NNL2_FUNC_EXIT();
		#endif
	
		return true;
	}
	
	// Start with current capacity or initial size if arena is empty
    size_t new_cap = a->capacity ? a->capacity : NNL2_ARENA_INITIAL_CAPACITY;
	
	// Geometric growth. double for small sizes, 1.5x for larger sizes
    while(new_cap < needed) {
        new_cap = (new_cap < NNL2_ARENA_GROWTH_THRESHOLD) 
			? (new_cap * 2) 					 // Double until threshold
			: (new_cap + (new_cap >> 1));		 // Then grow by 50% (new_cap * 1.5)
    }
	
	// Attempt to resize the memory buffer to the new calculated capacity
    unsigned char *new_buf = realloc(a->buf, new_cap);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!new_buf) {
			NNL2_ERROR("Failed to reallocate new buffer in nnl2_arena_ensure");
			return false;
		}
	#endif
	
    a->buf = new_buf;
    a->capacity = new_cap;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
	
    return true;
}

/** @brief 
 * Allocates aligned memory from the arena
 *
 ** @param a 
 * Pointer to the arena structure
 *
 ** @param n 
 * Number of bytes to allocate
 *
 ** @return 
 * Pointer to the allocated memory, or NULL if allocation failed
 *
 ** @note 
 * Memory is aligned to pointer size boundaries
 *
 ** @note
 * Automatically grows the arena if insufficient space is available
 *
 ** @exception NNL2Error [nnl2_safety_max+]
 * If a (input pointer) is NULL
 *
 ** @exception NNL2Error
 * If nnl2_arena_ensure failed
 **/
static inline void* nnl2_arena_alloc(nnl2_arena_t *a, size_t n) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(a, "In function nnl2_arena_free, nnl2_arena_t *a is NULL", false);
	#endif
	
	// Align to pointer size boundary for optimal performance
    const size_t align = sizeof(void*);
	
	// (align - 1) Creates a mask of lower bits to clear (e.g., 0x07 for 8-byte alignment)
	// ~(align - 1) Creates a mask of upper bits to keep (e.g., 0xFFFFFFF8 for 8-byte alignment)
	// (a->size + (align - 1)) Adds enough to push value to next alignment boundary if not already aligned
	// & ~(align - 1) Clears the lower bits, rounding down to alignment boundary
    size_t off = (a->size + (align - 1)) & ~(align - 1);
	
	// Total space needed
	// aligned offset + requested size
    size_t needed = off + n;
	
	// Check if arena needs to grow
    if(needed > a->capacity) {
		size_t additional_bytes = n + align;
		
		// Ensure arena has enough space for requested size + alignment padding
        if(!nnl2_arena_ensure(a, additional_bytes)) {
			NNL2_ERROR("Failed to ensure arena has enough capacity for an additional %zu bytes", additional_bytes);
			return NULL;
		}
		
		// Recalculate alignment since buffer might have moved during realloc
        // Same bit magic as above, but with potentially updated a->size
        off = (a->size + (align - 1)) & ~(align - 1);
        needed = off + n;
    }
	
	// Calculate pointer to allocated memory in the buffer
    void *ptr = a->buf + off;
	
	// Update arena's current size to mark this space as used
    a->size = needed;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
	
    return ptr;
}

/** @brief 
 * Builds a topological ordering of tensors starting from the root tensor
 *
 ** @param root 
 * The root tensor from which to start the topological sort
 *
 ** @param out_topo_size 
 * Pointer to store the size of the resulting topological array
 *
 ** @return 
 * Array of tensors in topological order, or NULL if allocation fails
 *
 ** @note 
 * Employs arena allocation for efficient memory management during traversal
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If root is NULL
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If out_topo_size is NULL
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If failed to allocate stack (see *stack = (nnl2_stack_frame*)nnl2_arena_alloc(...))
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If failed to allocate bufferr (see **topo_buf = (struct nnl2_ad_tensor**)nnl2_arena_alloc(...))
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If failed to ensure arena capacity during stack growth (see nnl2_arena_ensure)
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If failed to reallocate stack during capacity expansion (see realloc stack)
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If failed to reallocate topological buffer during capacity expansion (see realloc topo_buf)
 *
 ** @exception NNL2Error [nnl2_safety_min+]
 * If failed to allocate final result array (see malloc result)
 *
 ** @see NNL2_TOPO_INITIAL_ARENA_CAPACITY 4096
 ** @see NNL2_TOPO_INITIAL_STACK_CAPACITY 64
 ** @see NNL2_TOPO_INITIAL_TOPO_CAPACITY 64
 **/
static NNL2_FORCE_INLINE nnl2_ad_tensor** nnl2_ad_build_topo(struct nnl2_ad_tensor* root, int* out_topo_size) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(root, "In function nnl2_arena_free, struct nnl2_ad_tensor* root is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(out_topo_size, "In function nnl2_arena_free, int* out_topo_size root is NULL", NULL);
	#endif

	// Increment generation counter for this traversal, handle 0 wrap-around
    ++nnl2_ad_current_gen;
    if(nnl2_ad_current_gen == 0) ++nnl2_ad_current_gen;

	// Initialize arena for efficient memory management during graph traversal
    nnl2_arena_t arena;
    if(!nnl2_arena_init(&arena, NNL2_TOPO_INITIAL_ARENA_CAPACITY)) {
		NNL2_ERROR("Failed to initialize arena in topological sort (backpropagation core)");
		return NULL;
	}

	// Set initial capacities for topological buffer and stack
    size_t topo_capacity = NNL2_TOPO_INITIAL_TOPO_CAPACITY;
    size_t stack_capacity = NNL2_TOPO_INITIAL_STACK_CAPACITY;

	// Allocate stack frames from arena for iterative DFS
    nnl2_stack_frame *stack = (nnl2_stack_frame*)nnl2_arena_alloc(&arena, stack_capacity * sizeof(nnl2_stack_frame));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!stack) { 
			NNL2_ERROR("Failed to allocate stack in topological sort (backpropagation core)");
			nnl2_arena_free(&arena); 
			return NULL; 
		}
	#endif

	// Allocate topological result buffer from arena
    struct nnl2_ad_tensor **topo_buf = (struct nnl2_ad_tensor**)nnl2_arena_alloc(&arena, topo_capacity * sizeof(struct nnl2_ad_tensor*));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!topo_buf) { 
		    NNL2_ERROR("Failed to allocate buffer in topological sort (backpropagation core)");
			nnl2_arena_free(&arena); 
			return NULL; 
		}
	#endif

	// Initialize traversal state
    int topo_size = 0;	// Count of tensors in topological order
    int sp = 0;			// Stack pointer for DFS

	// Start DFS from root tensor with initial index 0 (first dependency)
    stack[sp++] = (nnl2_stack_frame){root, 0};
    root->visited_gen = nnl2_ad_current_gen;  // Mark root as visited in current generation

	// Iterative depth-first search loop
    while(sp > 0) {
		// Get current stack frame (peek without popping)
        nnl2_stack_frame *frame = &stack[sp - 1];
        struct nnl2_ad_tensor *t = frame->tensor;

		// Prefetch next child tensor if available
		#if NNL2_HAVE_TOPO_PREFETCH
			if (t->num_roots > 0 && t->roots) {
				NNL2_TOPO_PREFETCH(&t->roots[frame->index < t->num_roots ? frame->index : 0]);
			}
		#endif

		// Process current tensor's dependencies (children)
        if (frame->index < t->num_roots) {
			// Get next child dependency and advance index
            struct nnl2_ad_tensor *child = t->roots[frame->index++];
			
			// Process child if it exists and hasn't been visited in this generation
            if (child && child->visited_gen != nnl2_ad_current_gen) {
				// Calculate memory needed for potential stack growth
                size_t needed_stack = (size_t)(sp + 1) * sizeof(nnl2_stack_frame);
				
			    // Ensure arena has capacity for stack growth and topological buffer
                if (arena.size + needed_stack >= arena.capacity) {
                    if (!nnl2_arena_ensure(&arena, needed_stack + topo_capacity * sizeof(struct nnl2_ad_tensor*))) {
						NNL2_ERROR("Failed to ensure arena has enough capacity for an additional memory in nnl2_ad_build_topo");
                        nnl2_arena_free(&arena);
                        return NULL;
                    }
                }
				
				// Grow stack array if at capacity
                if ((size_t)sp >= stack_capacity) {
                    size_t new_stack_cap = stack_capacity * 2; // Double stack capacity
                    nnl2_stack_frame *new_stack = realloc(stack, new_stack_cap * sizeof(nnl2_stack_frame));
					
					#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
						if (!new_stack) { 
							NNL2_ERROR("Failed to reallocate new stack in nnl2_ad_build_topo");
							nnl2_arena_free(&arena); 
							return NULL; 
						}
					#endif
					
                    stack = new_stack;
                    stack_capacity = new_stack_cap;
                }
				
				// Mark child as visited and push onto stack for processing
                child->visited_gen = nnl2_ad_current_gen;
                stack[sp++] = (nnl2_stack_frame){child, 0};  // Start with index 0 for child's dependencies
            }
        } else {
			// Grow topological buffer if at capacity
            if ((size_t)topo_size >= topo_capacity) {
				// Geometric growth, double for small sizes, 1.5x for larger sizes
                size_t new_capacity = (topo_capacity < NNL2_ARENA_GROWTH_THRESHOLD) ? (topo_capacity * 2)  					     // Double capacity
																					: (topo_capacity + (topo_capacity >> 1));    // Grow by 50% (>> 1 is divide by 2)
																					
                struct nnl2_ad_tensor **new_topo = realloc(topo_buf, new_capacity * sizeof(struct nnl2_ad_tensor*));
				
				#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
					if (!new_topo) { 
						NNL2_ERROR("Failed to reallocate new topo in nnl2_ad_build_topo");
						nnl2_arena_free(&arena); 
						return NULL; 
					}
				#endif
				
                topo_buf = new_topo;
                topo_capacity = new_capacity;
            }
			
			// Add tensor to topological order 
            topo_buf[topo_size++] = t;
            sp--;  // Pop current tensor from stack
        }
    }

	// Allocate final result array
    struct nnl2_ad_tensor **result = malloc((size_t)topo_size * sizeof(struct nnl2_ad_tensor*));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!result) { 
			NNL2_MALLOC_ERROR();
			nnl2_arena_free(&arena); 
			return NULL; 
		}
	#endif
	
	// Copy topological order from arena buffer to final result
    memcpy(result, topo_buf, (size_t)topo_size * sizeof(struct nnl2_ad_tensor*));

	// Automatically frees stack and topo_buf
    nnl2_arena_free(&arena);
	
	// Return results to caller
    *out_topo_size = topo_size;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
    return result;
}

#endif /* NNL2_AD_TOPO_H */
