#ifndef NNL2_AD_TOPO_H
#define NNL2_AD_TOPO_H

void nnl2_ad_build_topo_internal(nnl2_ad_tensor* tensor, nnl2_ad_tensor*** topo, int* topo_size, int* topo_capacity, bool* visited, int visited_size) {
	for(int i = 0; i < *topo_size; i++) {
		if((*topo)[i] == tensor) return;
	}
	
	for (size_t i = 0; i < tensor->num_roots; i++) {
        nnl2_ad_build_topo_internal(tensor->roots[i], topo, topo_size, topo_capacity, visited, visited_size);
    }
	
	if (*topo_size >= *topo_capacity) {
		*topo_capacity *= 2;
        *topo = realloc(*topo, *topo_capacity * sizeof(nnl2_ad_tensor*));
    }
	
    (*topo)[(*topo_size)++] = tensor;
}

nnl2_ad_tensor** nnl2_ad_build_topo(nnl2_ad_tensor* tensor, int* topo_size) {
	int capacity = 16;
	nnl2_ad_tensor** topo = malloc(capacity * sizeof(nnl2_ad_tensor*));	
	*topo_size = 0;
	bool* visited = NULL;
	
	nnl2_ad_build_topo_internal(tensor, &topo, topo_size, &capacity, visited, 0);
    
	NNL2_WARN("Don't forget to add optimizations to nnl2_ad_build_topo in the future");
	
    return topo;
}

#endif /** NNL2_AD_TOPO_H **/
