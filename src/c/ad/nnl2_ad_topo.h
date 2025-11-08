#ifndef NNL2_AD_TOPO_H
#define NNL2_AD_TOPO_H

#include <stdlib.h>
#include <stdbool.h>

static void nnl2_ad_build_topo_internal(nnl2_ad_tensor* restrict tensor, nnl2_ad_tensor*** restrict topo, int* topo_size, int* topo_capacity) {
	if(!tensor || tensor->visited) return;
	
	tensor->visited = true;
	
    for (size_t i = 0; i < tensor->num_roots; ++i) {
        nnl2_ad_build_topo_internal(tensor->roots[i], topo, topo_size, topo_capacity);
    }

    if (*topo_size >= *topo_capacity) {
        int new_capacity = (*topo_capacity < 16384) ? (*topo_capacity * 2) : (*topo_capacity + (*topo_capacity >> 1));
        nnl2_ad_tensor** new_topo = realloc(*topo, new_capacity * sizeof(nnl2_ad_tensor*));
        if (!new_topo) {
            NNL2_FATAL("todo");
        }
		
        *topo = new_topo;
        *topo_capacity = new_capacity;
    }

    (*topo)[(*topo_size)++] = tensor;
}

static inline nnl2_ad_tensor** nnl2_ad_build_topo(nnl2_ad_tensor* root, int* restrict topo_size) {
    if (!root || !topo_size) return NULL;
	
	int capacity = 64;
    nnl2_ad_tensor** topo = malloc(capacity * sizeof(nnl2_ad_tensor*));  
	
    *topo_size = 0;
    nnl2_ad_build_topo_internal(root, &topo, topo_size, &capacity);
	
	for(int i = 0; i < *topo_size; ++i) topo[i]->visited = false;
    
    return topo;
}

#endif /** NNL2_AD_TOPO_H **/
