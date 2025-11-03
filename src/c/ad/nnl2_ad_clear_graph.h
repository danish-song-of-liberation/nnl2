#ifndef NNL2_AD_CLEAR_GRAPH_H
#define NNL2_AD_CLEAR_GRAPH_H

void nnl2_ad_clear_graph(nnl2_ad_tensor** topo, int topo_size) {
    for (int i = 0; i < topo_size; i++) {
        nnl2_ad_tensor* node = topo[i];
        node->backward_fn = NULL;
		
        if (node->roots) {
            for (size_t j = 0; j < node->num_roots; j++) {
                node->roots[j] = NULL;
            }
			
            free(node->roots);
			
            node->roots = NULL;
            node->num_roots = 0;
        }
        
        node->extra_multiplier = 1.0f;
        node->extra_bool = false;
        node->extra_correspondence = NULL;
    }
}

#endif /** NNL2_AD_CLEAR_GRAPH_H **/
