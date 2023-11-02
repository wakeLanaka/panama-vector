#ifndef SHARE_PRIMS_OPENCLHELPER_HPP
#define SHARE_PRIMS_OPENCLHELPER_HPP

#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"

const char *getOpenCLError(cl_int error);
void handleError(cl_int value, const char * errorString);
unsigned int verifyZeroCopyPtr(void *ptr, unsigned int sizeOfContentsOfPtr);

#endif // SHARE_PRIMS_OPENCLHELPER_HPP
