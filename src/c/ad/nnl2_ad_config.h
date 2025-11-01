#ifndef NNL2_AD_CONFIG_H
#define NNL2_AD_CONFIG_H

#define NNL2_AD_INPLACE_FATAL(operation, ad_tensor)  \
  if(ad_tensor->name) {				\
	  NNL2_FATAL("In operation `" operation "`, on tensor %x (namely %s), inplace operations are not allowed in AD as they break the computational graph. Try to put :requires-grad nil", ad_tensor, ad_tensor->name); \
  } else { \
      NNL2_FATAL("In operation `" operation "`, on tensor %x, inplace operations are not allowed in AD as they break the computational graph. Try to put :requires-grad nil", ad_tensor); \
  }

#endif /** NNL2_AD_CONFIG_H **/
