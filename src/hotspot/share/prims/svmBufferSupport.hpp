#ifndef SHARE_PRIMS_SVMBUFFERSUPPORT_HPP
#define SHARE_PRIMS_SVMBUFFERSUPPORT_HPP

#include "code/debugInfo.hpp"
#include "jni.h"
#include "memory/allStatic.hpp"
#include "oops/typeArrayOop.hpp"
#include "runtime/registerMap.hpp"
#include "utilities/exceptions.hpp"

extern "C" {
  void JNICALL JVM_RegisterSVMBufferSupportMethods(JNIEnv* env, jclass vsclass);
}

#endif // SHARE_PRIMS_SVMBUFFERSUPPORT_HPP
