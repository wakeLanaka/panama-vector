#include "prims/svmBufferSupport.hpp"
#include "runtime/interfaceSupport.inline.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include "openclHelper.hpp"

cl_int svmError = 0;

JVM_ENTRY(long, SVMBufferSupport_createContext(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_device_id clDevice = (cl_device_id) jDevice;
  cl_context clContext = clCreateContext(0, 1, &clDevice, NULL, NULL, &svmError);
  handleError(svmError, "clCreateContext");

  return (jlong) clContext;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseContext(JNIEnv *env, jclass vsclazz, jlong jContext)) {
  cl_context clContext = (cl_context) jContext;
  if (clContext) {
    svmError = clReleaseContext(clContext);
    handleError(svmError, "ReleaseContext");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createProgram(JNIEnv *env, jclass vsclazz, jlong jContext, jstring jKernelString)) {
  cl_context clContext = (cl_context)jContext;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);
  const char * clKernelString[] = {kernel};
  const size_t clKernelLength = (size_t)env->GetStringLength(jKernelString);

  cl_program clProgram = clCreateProgramWithSource(clContext, 1, clKernelString, &clKernelLength, &svmError);
  handleError(svmError, "clCreateProgramWithSource");

  svmError = clBuildProgram(clProgram, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
  handleError(svmError, "clBuildProgram");
  env->ReleaseStringUTFChars(jKernelString, kernel);

  return (jlong)clProgram;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseProgram(JNIEnv *env, jclass vsclazz, jlong jProgram)) {
  cl_program clProgram = (cl_program) jProgram;
  if (clProgram) {
    svmError = clReleaseProgram(clProgram);
    handleError(svmError, "ReleaseProgram");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createCommandQueue(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jDevice)) {
  cl_context clContext = (cl_context)jContext;
  cl_device_id clDevice = (cl_device_id)jDevice;
  cl_command_queue_properties properties[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue clCommandQueue = clCreateCommandQueueWithProperties(clContext, clDevice, properties, &svmError);
  handleError(svmError, "createCommandQueue");
  return (jlong)clCommandQueue;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseCommandQueue(JNIEnv *env, jclass vsclazz, jlong jCommandQueue)) {
  cl_command_queue clCommandQueue = (cl_command_queue) jCommandQueue;
  if (clCommandQueue) {
    svmError = clReleaseCommandQueue(clCommandQueue);
    handleError(svmError, "clReleaseCommandQueue");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createDevice(JNIEnv *env, jclass vsclazz)) {
  cl_platform_id clPlatform = NULL;
  cl_device_id clDevice = NULL;

  svmError = clGetPlatformIDs(1, &clPlatform, NULL);
  handleError(svmError, "clGetPlatformIDs");

  svmError = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);
  handleError(svmError, "clGetDeviceIDs");

  return (jlong)clDevice;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseDevice(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_device_id clDevice = (cl_device_id)jDevice;
  if (clDevice) {
    svmError = clReleaseDevice(clDevice);
    handleError(svmError, "clReleaseDevice");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createReadSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jint length)) {
  cl_context clContext = (cl_context)jContext;
  int * svmBuffer = (int *) clSVMAlloc(clContext, CL_MEM_READ_ONLY, sizeof(int) * (int)length, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createWriteSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jint length)) {
  cl_context clContext = (cl_context)jContext;
  int * svmBuffer = (int *)clSVMAlloc(clContext, CL_MEM_WRITE_ONLY, sizeof(int) * (int)length, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_writeSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jCommandQueue, jlong jBuffer, jint index, jint length, jint value)) {
  int * svmBuffer = (int *)jBuffer;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_WRITE, svmBuffer, sizeof(int) * (int)length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap");
  svmBuffer[index] = value;
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap");
} JVM_END

JVM_ENTRY(jint, SVMBufferSupport_readSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jCommandQueue, jlong jBuffer, jint index, jint length)) {
  int * svmBuffer = (int *)jBuffer;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * (int)length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap");
  int value = svmBuffer[index];
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap");
  return value;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_addSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  int* clBuffer3 = (int *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_add", NULL);

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 3, sizeof(int), &clLength);
  handleError(svmError, "clSetKernelArg");


  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_matrixFmaSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint K, jint N, jint k, jint length)) {
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  cl_program clProgram = (cl_program)jProgram;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clK = (int)K;
  int clN = (int)N;
  int clk = (int)k;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "matrix_fma", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 3, sizeof(int), &clK);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 4, sizeof(int), &clN);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 5, sizeof(int), &clk);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_fmaSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  cl_program clProgram = (cl_program)jProgram;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_fma", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(jfloat, SVMBufferSupport_reduceAdd(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jProgram, jlong jCommandQueue, jlong jBuffer, jint length)) {
  cl_context clContext = (cl_context)jContext;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer = (float *)jBuffer;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_reduce", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer);
  handleError(svmError, "clSetKernelArg");

  cl_mem c_mem_obj = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(float), NULL, &svmError);
  handleError(svmError, "createBuffer");

  svmError = clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_mem_obj);
  handleError(svmError, "setArgs");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  float sum = 0;
  clEnqueueReadBuffer(clCommandQueue, c_mem_obj, CL_TRUE, 0, sizeof(float), &sum, 0, NULL, NULL);

  clReleaseKernel(kernel);
  return sum;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_copyFromArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jintArray jArray)) {
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

  int * svmBuffer = (int *) clSVMAlloc((cl_context)clContext, CL_MEM_READ_ONLY, sizeof(int) * length, 0);
  int * jarrayElements = env->GetIntArrayElements(jArray, 0);
  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap");
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, svmBuffer, jarrayElements, sizeof(int) * length, 0, 0, 0);
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap");
  env->ReleaseIntArrayElements(jArray, jarrayElements, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_copyFromFloatArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jfloatArray jArray)) {
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

  float * svmBuffer = (float *) clSVMAlloc((cl_context)clContext, CL_MEM_READ_WRITE, sizeof(float) * length, 0);
  float * jarrayElements = env->GetFloatArrayElements(jArray, 0);
  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap");
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, svmBuffer, jarrayElements, sizeof(float) * length, 0, 0, 0);
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap");
  env->ReleaseFloatArrayElements(jArray, jarrayElements, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jintArray jArray)) {
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

  int* svmBuffer = (int *)jBuffer;

  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap");
  int * jarrayBody = env->GetIntArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, svmBuffer, sizeof(int) * length, 0, 0, 0);
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap");
  env->ReleaseIntArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToFloatArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jfloatArray jArray)) {
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

  float* clBuffer = (float *)jBuffer;

  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap");
  float * jarrayBody = env->GetFloatArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, clBuffer, sizeof(float) * length, 0, 0, 0);
  svmError = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap");
  env->ReleaseFloatArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jBuffer)) {
  cl_context clContext = (cl_context)jContext;
  int* svmBuffer = (int *)jBuffer;
  clSVMFree(clContext, svmBuffer);
} JVM_END

#define CC (char*)  /*cast a literal from (const char*)*/
#define FN_PTR(f) CAST_FROM_FN_PTR(void*, &f)

static JNINativeMethod jdk_internal_vm_vector_SVMBufferSupport_methods[] = {
    {CC "CreateContext",   CC "(J)J", FN_PTR(SVMBufferSupport_createContext)},
    {CC "ReleaseContext",   CC "(J)V", FN_PTR(SVMBufferSupport_releaseContext)},
    {CC "CreateProgram",   CC "(JLjava/lang/String;)J", FN_PTR(SVMBufferSupport_createProgram)},
    {CC "ReleaseProgram",   CC "(J)V", FN_PTR(SVMBufferSupport_releaseProgram)},
    {CC "CreateCommandQueue",   CC "(JJ)J", FN_PTR(SVMBufferSupport_createCommandQueue)},
    {CC "ReleaseCommandQueue",   CC "(J)V", FN_PTR(SVMBufferSupport_releaseCommandQueue)},
    {CC "CreateDevice",   CC "()J", FN_PTR(SVMBufferSupport_createDevice)},
    {CC "ReleaseDevice",   CC "(J)V", FN_PTR(SVMBufferSupport_releaseDevice)},
    {CC "CreateReadSVMBuffer",   CC "(JI)J", FN_PTR(SVMBufferSupport_createReadSVMBuffer)},
    {CC "CreateWriteSVMBuffer",   CC "(JI)J", FN_PTR(SVMBufferSupport_createWriteSVMBuffer)},
    {CC "WriteSVMBuffer",   CC "(JJIII)V", FN_PTR(SVMBufferSupport_writeSVMBuffer)},
    {CC "ReadSVMBuffer",   CC "(JJII)I", FN_PTR(SVMBufferSupport_readSVMBuffer)},
    {CC "AddSVMBuffer",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_addSVMBuffer)},
    {CC "CopyToArray",   CC "(JJJ[I)V", FN_PTR(SVMBufferSupport_copyToArray)},
    {CC "CopyToFloatArray",   CC "(JJJ[F)V", FN_PTR(SVMBufferSupport_copyToFloatArray)},
    {CC "CopyFromArray",   CC "(JJ[I)J", FN_PTR(SVMBufferSupport_copyFromArray)},
    {CC "CopyFromFloatArray",   CC "(JJ[F)J", FN_PTR(SVMBufferSupport_copyFromFloatArray)},
    {CC "ReleaseSVMBuffer",   CC "(JJ)V", FN_PTR(SVMBufferSupport_releaseSVMBuffer)},
    {CC "MatrixFmaSVMBuffer",   CC "(JJJJJIIII)V", FN_PTR(SVMBufferSupport_matrixFmaSVMBuffer)},
    {CC "FmaSVMBuffer",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_fmaSVMBuffer)},
    {CC "ReduceAdd",   CC "(JJJJI)F", FN_PTR(SVMBufferSupport_reduceAdd)},
};

JVM_ENTRY(void, JVM_RegisterSVMBufferSupportMethods(JNIEnv* env, jclass vsclass)) {
  ThreadToNativeFromVM ttnfv(thread);

  int ok = env->RegisterNatives(vsclass, jdk_internal_vm_vector_SVMBufferSupport_methods, sizeof(jdk_internal_vm_vector_SVMBufferSupport_methods)/sizeof(JNINativeMethod));
  guarantee(ok == 0, "register jdk.internal.vm.vector.SVMBufferSupport natives");
} JVM_END
