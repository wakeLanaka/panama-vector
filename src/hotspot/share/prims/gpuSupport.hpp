#ifndef SHARE_PRIMS_GPUSUPPORT_HPP
#define SHARE_PRIMS_GPUSUPPORT_HPP

#include "code/debugInfo.hpp"
#include "jni.h"
#include "memory/allStatic.hpp"
#include "oops/typeArrayOop.hpp"
#include "runtime/registerMap.hpp"
#include "utilities/exceptions.hpp"

extern "C" {
  void JNICALL JVM_RegisterGPUSupportMethods(JNIEnv* env, jclass vsclass);
}

#endif // SHARE_PRIMS_GPUSUPPORT_HPP
