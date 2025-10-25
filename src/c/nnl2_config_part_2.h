#ifndef NNL2_CONFIG_PART_2_H
#define NNL2_CONFIG_PART_2_H

nnl2_object_type get_nnl2_object_type(void* obj) {
    if (obj == NULL) return nnl2_type_unknown;
    
    nnl2_object_type* type_pntr = (nnl2_object_type*)obj;
    return *type_pntr;
}

#endif /** NNL2_CONFIG_PART_2_H **/
