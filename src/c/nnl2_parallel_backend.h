#ifndef NNL2_PARALLEL_BACKEND_H
#define NNL2_PARALLEL_BACKEND_H

///@{

typedef struct {
	void* data;    ///< Pointer to an array with tensor data
	size_t start;  ///< Index for entering the data array
	size_t end;    ///< Index for the end of the data entry into the array
} single_arr_ptask;  

///@}	

#endif /** NNL2_PARALLEL_BACKEND_H **/
