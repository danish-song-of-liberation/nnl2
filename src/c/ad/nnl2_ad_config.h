#ifndef NNL2_AD_CONFIG_H
#define NNL2_AD_CONFIG_H

// NNL2

/** @file nnl2_ad_config.h
 ** @brief Automatic differentiation configuration and macros
 ** @date 2025
 ** @copyright MIT
 **/
 
///@{
	
/** @brief 
 * Macro for fatal error when in-place operations are attempted on AD tensors
 *
 ** @param operation
 * Name of the operation that triggered the error
 *
 ** @param ad_tensor  
 * AD tensor on which the operation was attempted
 */
#define NNL2_AD_INPLACE_FATAL(operation, ad_tensor)  \
  if(ad_tensor->name) {				\
	  NNL2_FATAL("In operation `" operation "`, on tensor %x (namely %s), inplace operations are not allowed in AD as they break the computational graph. Try to put :track-graph nil into operation (or :requires-grad nil into leaf tensor)", ad_tensor, ad_tensor->name); \
  } else { \
      NNL2_FATAL("In operation `" operation "`, on tensor %x, inplace operations are not allowed in AD as they break the computational graph. Try to put :track-graph nil into operation (or :requires-grad nil into leaf tensor)", ad_tensor); \
  }
  
///@}

#endif /** NNL2_AD_CONFIG_H **/
