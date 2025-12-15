#ifndef NNL2_VECTOR_AS_PARAMETER_H
#define NNL2_VECTOR_AS_PARAMETER_H

nnl2_tensor* nnl2_vector_as_parameter(int32_t* shape, int rank, size_t start, nnl2_tensor* vector) {
	nnl2_tensor* result = (nnl2_tensor*)malloc(sizeof(nnl2_tensor));
	
	result -> ts_type = vector -> ts_type;  
    result -> dtype = vector -> dtype;     
    result -> rank = rank;      
    result -> magic_number = TENSOR_MAGIC_ALIVE;
	result -> is_view = true; 	 
	 
	result -> shape = shape;
    result -> strides = (int32_t*)malloc(rank * sizeof(int32_t));
		
	result -> data = vector -> data + start * get_dtype_size(vector -> dtype);
	
	return result;
}

#endif /** NNL2_VECTOR_AS_PARAMETER_H **/
