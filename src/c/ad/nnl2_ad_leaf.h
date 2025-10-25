#ifndef NNL2_AD_LEAF_H
#define NNL2_AD_LEAF_H

nnl2_ad_tensor* nnl2_ad_find_leaf(nnl2_ad_tensor* tensor) {
	if(tensor->is_leaf) return tensor;
	
	for(size_t i = 0; i < tensor->num_roots; i++) {
		nnl2_ad_tensor* potential_leaf = nnl2_ad_find_leaf(tensor->roots[i]);
		if(potential_leaf != NULL) {
			return potential_leaf;
		}
		
		nnl2_free_ad_tensor(potential_leaf);
	}
		
	NNL2_WARN("Don't forget to add optimizations to nnl2_ad_find_leaf in the future");
	NNL2_WARN("Don't forget to add tensor freeing to nnl2_ad_find_leaf in the future");
	
	return NULL;
}

#endif /** NNL2_AD_LEAF_H **/
