#include "precompiled.hpp"
#include "prims/svmBufferSupport.hpp"
#include "runtime/interfaceSupport.inline.hpp"
#include "openclHelper.hpp"
#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include <cmath>

JVM_ENTRY(jlong, SVMBufferSupport_createContext(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_int error = 0;
  const cl_device_id clDevice = (cl_device_id) jDevice;
  const cl_context clContext = clCreateContext(0, 1, &clDevice, NULL, NULL, &error);
  handleError(error, "clCreateContext", "createContext");

  return (jlong) clContext;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseContext(JNIEnv *env, jclass vsclazz, jlong jContext)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context) jContext;
  if (clContext) {
    error = clReleaseContext(clContext);
    handleError(error, "ReleaseContext", "releaseContext");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createProgram(JNIEnv *env, jclass vsclazz, jlong jContext, jstring jKernelString)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);
  const char * clKernelString[] = {kernel};
  const size_t clKernelLength = (size_t)env->GetStringLength(jKernelString);

  const cl_program clProgram = clCreateProgramWithSource(clContext, 1, clKernelString, &clKernelLength, &error);
  handleError(error, "clCreateProgramWithSource", "createProgram");

  error = clBuildProgram(clProgram, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
  handleError(error, "clBuildProgram", "createProgram");
  env->ReleaseStringUTFChars(jKernelString, kernel);

  return (jlong)clProgram;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseProgram(JNIEnv *env, jclass vsclazz, jlong jProgram)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program) jProgram;
  if (clProgram) {
    error = clReleaseProgram(clProgram);
    handleError(error, "ReleaseProgram", "ReleaseProgram");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createCommandQueue(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jDevice)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_device_id clDevice = (cl_device_id)jDevice;
  const cl_command_queue clCommandQueue = clCreateCommandQueueWithProperties(clContext, clDevice, NULL, &error);
  handleError(error, "createCommandQueue", "CreateCommandQueue");
  return (jlong)clCommandQueue;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseCommandQueue(JNIEnv *env, jclass vsclazz, jlong jCommandQueue)) {
  cl_int error = 0;
  const cl_command_queue clCommandQueue = (cl_command_queue) jCommandQueue;
  if (clCommandQueue) {
    error = clReleaseCommandQueue(clCommandQueue);
    handleError(error, "clReleaseCommandQueue", "releaseCommandQueue");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createDevice(JNIEnv *env, jclass vsclazz)) {
  cl_int error = 0;
  cl_platform_id clPlatform = NULL;
  cl_device_id clDevice = NULL;

  error = clGetPlatformIDs(1, &clPlatform, NULL);
  handleError(error, "clGetPlatformIDs", "createDevice");

  error = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);
  handleError(error, "clGetDeviceIDs", "createDevice");

  return (jlong)clDevice;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseDevice(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_int error = 0;
  const cl_device_id clDevice = (cl_device_id)jDevice;
  if (clDevice) {
    error = clReleaseDevice(clDevice);
    handleError(error, "clReleaseDevice", "releaseDevice");
  }
} JVM_END

JVM_ENTRY(jint, SVMBufferSupport_getMaxWorkGroupSize(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  const cl_device_id clDevice = (cl_device_id)jDevice;
  size_t maxWorkGroupSize = 0;
  clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
  return (jint)maxWorkGroupSize;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createReadWriteFloatSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jint length)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const int clLength = (int)length;

  float * svmBuffer = (float *)clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(float) * clLength, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createReadWriteIntSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jint length)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const int clLength = (int)length;
  int * svmBuffer = (int *)clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(int) * clLength, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_matrixFmaSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint K, jint N, jint k, jint length)) {
  cl_int error = 0;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  const cl_program clProgram = (cl_program)jProgram;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clK = (int)K;
  const int clN = (int)N;
  const int clk = (int)k;
  const int clLength = (int)length;

  const char * kernelName = "matrix_fma";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg2", kernelName);

  error = clSetKernelArg(kernel, 3, sizeof(int), &clK);
  handleError(error, "clSetKernelArg3", kernelName);

  error = clSetKernelArg(kernel, 4, sizeof(int), &clN);
  handleError(error, "clSetKernelArg4", kernelName);

  error = clSetKernelArg(kernel, 5, sizeof(int), &clk);
  handleError(error, "clSetKernelArg5", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_fmaSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jlong jBuffer4, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  float* clBuffer4 = (float *)jBuffer4;
  const int clLength = (int)length;

  const char * kernelName = "vector_fma";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg2", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 3, clBuffer4);
  handleError(error, "clSetKernelArg3", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(jfloat, SVMBufferSupport_sumReduceFLOAT(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jProgram, jlong jCommandQueue, jlong jBuffer, jint maxWorkGroupSize, jint length)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer = (float *)jBuffer;
  const int clLength = (int)length;
  const int localSize = (int)maxWorkGroupSize;

  const jsize workGroups = (jsize)std::ceil(clLength / (float)localSize);
  float* sums = (float * )clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(float) * workGroups, 0);

  const char * kernelName = "sumreduceFLOAT";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, sums);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArg(kernel, 2, sizeof(float) * localSize, NULL);
  handleError(error, "clSetKernelArg2", kernelName);

  size_t local_item_size[] = {(size_t)localSize};
  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, local_item_size, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, sums, sizeof(float) * workGroups, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", kernelName);
  jfloatArray jSums = env->NewFloatArray(workGroups);
  jfloat * jsums = env->GetFloatArrayElements(jSums, NULL);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jsums, sums, sizeof(float) * workGroups, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, sums, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", kernelName);

  float sum = 0.0f;
  for(jsize i = 0; i < workGroups; i++) {
    sum += (float)jsums[i];
  }

  env->ReleaseFloatArrayElements(jSums, jsums, 0);
  clReleaseKernel(kernel);
  clSVMFree(clContext, sums);
  return (jfloat)sum;
} JVM_END

JVM_ENTRY(jint, SVMBufferSupport_sumReduceINT(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jProgram, jlong jCommandQueue, jlong jBuffer, jint maxWorkGroupSize, jint length)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer = (int *)jBuffer;
  const int clLength = (int)length;
  const int localSize = (int)maxWorkGroupSize;

  const jsize workGroups = (jsize)std::ceil(clLength / (float)localSize);
  int* sums = (int * )clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(int) * workGroups, 0);

  const char * kernelName = "sumreduceINT";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, sums);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArg(kernel, 2, sizeof(int) * localSize, NULL);
  handleError(error, "clSetKernelArg2", kernelName);

  size_t local_item_size[] = {(size_t)localSize};
  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, local_item_size, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, sums, sizeof(int) * workGroups, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", kernelName);
  jintArray jSums = env->NewIntArray(workGroups);
  jint * jsums = env->GetIntArrayElements(jSums, NULL);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jsums, sums, sizeof(int) * workGroups, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, sums, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", kernelName);

  int sum = 0;
  for(jsize i = 0; i < workGroups; i++) {
    sum += (int)jsums[i];
  }

  env->ReleaseIntArrayElements(jSums, jsums, 0);
  clReleaseKernel(kernel);
  clSVMFree(clContext, sums);
  return (jint)sum;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_copyFromArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jintArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  const jsize length = env->GetArrayLength(jArray);
  const char * name = "copyFromArray";

  int * svmBuffer = (int *) clSVMAlloc(clContext, CL_MEM_READ_ONLY, sizeof(int) * length, 0);
  int * jarrayElements = env->GetIntArrayElements(jArray, 0);
  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", name);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, svmBuffer, jarrayElements, sizeof(int) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", name);
  env->ReleaseIntArrayElements(jArray, jarrayElements, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_copyFromFloatArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jfloatArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  const jsize length = env->GetArrayLength(jArray);
  const char * name = "copyFromFloatArray";

  float * svmBuffer = (float *) clSVMAlloc((cl_context)clContext, CL_MEM_READ_WRITE, sizeof(float) * length, 0);
  float * jarrayElements = env->GetFloatArrayElements(jArray, 0);
  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", name);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, svmBuffer, jarrayElements, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", name);
  env->ReleaseFloatArrayElements(jArray, jarrayElements, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_fill(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jfloatArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer = (float *)jBuffer;

  const jsize length = env->GetArrayLength(jArray);
  const char * name = "fill";

  float * jarrayElements = env->GetFloatArrayElements(jArray, 0);
  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", name);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, clBuffer, jarrayElements, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", name);
  env->ReleaseFloatArrayElements(jArray, jarrayElements, 0);

  return (jlong)clBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_fillInt(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jintArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer = (int *)jBuffer;

  const jsize length = env->GetArrayLength(jArray);
  const char * name = "fillInt";

  int * jarrayElements = env->GetIntArrayElements(jArray, 0);
  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", name);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, clBuffer, jarrayElements, sizeof(int) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", name);
  env->ReleaseIntArrayElements(jArray, jarrayElements, 0);

  return (jlong)clBuffer;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToIntArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jintArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  const jsize length = env->GetArrayLength(jArray);
  const char * name = "copyToIntArray";

  int* svmBuffer = (int *)jBuffer;

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", name);
  int * jarrayBody = env->GetIntArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, svmBuffer, sizeof(int) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", name);
  env->ReleaseIntArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToFloatArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jfloatArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  const jsize length = env->GetArrayLength(jArray);
  const char * name = "copyToFloatArray";

  float* clBuffer = (float *)jBuffer;

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", name);
  float * jarrayBody = env->GetFloatArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, clBuffer, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", name);
  env->ReleaseFloatArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyRangeToFloatArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jfloatArray jArray, jint bufferLength, jint length, jint offset)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  const char * name = "copyRangeToFloatArray";

  float* clBuffer = (float *)jBuffer;

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * bufferLength, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", name);
  float * jarrayBody = env->GetFloatArrayElements(jArray, 0);
  float * offsetPointer = jarrayBody + offset;
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, offsetPointer, clBuffer, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", name);
  env->ReleaseFloatArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyRangeToIntArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jintArray jArray, jint bufferLength, jint length, jint offset)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  const char * name = "copyRangeToIntArray";

  int* clBuffer = (int *)jBuffer;

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(int) * bufferLength, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap", name);
  int * jarrayBody = env->GetIntArrayElements(jArray, 0);
  int * offsetPointer = jarrayBody + offset;
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, offsetPointer, clBuffer, sizeof(int) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap", name);
  env->ReleaseIntArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* svmBuffer = (float *)jBuffer;
  const char * name = "releaseSVMBuffer";
  error = clFinish(clCommandQueue);
  handleError(error, "clFinish", name);
  error = clFlush(clCommandQueue);
  handleError(error, "clFlush", name);
  clSVMFree(clContext, svmBuffer);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseSVMBufferInt(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* svmBuffer = (int *)jBuffer;
  const char * name = "releaseSVMBufferInt";
  error = clFinish(clCommandQueue);
  handleError(error, "clFinish", name);
  error = clFlush(clCommandQueue);
  handleError(error, "clFlush", name);
  clSVMFree(clContext, svmBuffer);
} JVM_END

template<typename T1, typename T2, typename T3>
void executeKernelValue(cl_program clProgram, cl_command_queue clCommandQueue, T1 clBuffer1, T2 clFactor, T3 clBuffer2, int clLength, const char * kernelName){
  cl_int error = 0;

  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArg(kernel, 1, sizeof(T2), &clFactor);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer2);
  handleError(error, "clSetKernelArg2", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
}

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesFFF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jfloat jFactor, jlong jBuffer2, jint length, jstring jKernelString)) {
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  jfloat* clBuffer1 = (jfloat *)jBuffer1;
  jfloat clFactor = (jfloat)jFactor;
  jfloat* clBuffer2 = (jfloat *)jBuffer2;
  const jint clLength = (jint)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelValue(clProgram, clCommandQueue, clBuffer1, clFactor, clBuffer2, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesIFF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jfloat jFactor, jlong jBuffer2, jint length, jstring jKernelString)) {
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  jint* clBuffer1 = (jint *)jBuffer1;
  const jfloat clFactor = (jfloat)jFactor;
  jfloat* clBuffer2 = (jfloat *)jBuffer2;
  const jint clLength = (jint)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelValue(clProgram, clCommandQueue, clBuffer1, clFactor, clBuffer2, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesFIF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jint jFactor, jlong jBuffer2, jint length, jstring jKernelString)) {
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  jfloat* clBuffer1 = (jfloat *)jBuffer1;
  const jint clFactor = (jint)jFactor;
  jfloat* clBuffer2 = (jfloat *)jBuffer2;
  const jint clLength = (jint)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelValue(clProgram, clCommandQueue, clBuffer1, clFactor, clBuffer2, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesIII(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jint jFactor, jlong jBuffer2, jint length, jstring jKernelString)) {
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  jint* clBuffer1 = (jint *)jBuffer1;
  const jint clFactor = (jint)jFactor;
  jint* clBuffer2 = (jint *)jBuffer2;
  const jint clLength = (jint)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelValue(clProgram, clCommandQueue, clBuffer1, clFactor, clBuffer2, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

template<typename T1, typename T2, typename T3>
void executeKernelBuffer(cl_program clProgram, cl_command_queue clCommandQueue, T1 clBuffer1, T2 clBuffer2, const T3 clResult, int clLength, const char * kernelName){
  cl_int error = 0;

  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clResult);
  handleError(error, "clSetKernelArg2", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
}

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferFFF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferFIF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferIFF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferIII(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  int* clBuffer3 = (int *)jBuffer3;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

template<typename T1, typename T2>
void executeKernelBuffer(cl_program clProgram, cl_command_queue clCommandQueue, T1 clBuffer1, T2 clBuffer2, int clLength, const char * kernelName){
  cl_int error = 0;

  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
}

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferFF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferFI(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferIF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferII(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernelBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyRange(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jint index1, jlong jBuffer2, jint index2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clIndex1 = (int)index1;
  const int clIndex2 = (int)index2;
  const int clLength = (int)length;

  const char * kernelName = "multiplyrange";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg2", kernelName);

  error = clSetKernelArg(kernel, 3, sizeof(int), &clIndex1);
  handleError(error, "clSetKernelArg3", kernelName);

  error = clSetKernelArg(kernel, 4, sizeof(int), &clIndex2);
  handleError(error, "clSetKernelArg4", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_blend(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jMask, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clMask = (float *)jMask;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;

  const char * kernelName = "blend";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clMask);
  handleError(error, "clSetKernelArg2", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 3, clBuffer3);
  handleError(error, "clSetKernelArg3", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
} JVM_END

template<typename T1, typename T2, typename T3, typename T4>
void executeKernel(cl_program clProgram, cl_command_queue clCommandQueue, T1 clBuffer1, T2 clBuffer2, T3 clBuffer3, T4 value, int clLength, const char * kernelName){
  cl_int error = 0;

  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg2", kernelName);

  error = clSetKernelArg(kernel, 3, sizeof(T4), &value);
  handleError(error, "clSetKernelArg3", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
}


JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferFFFI(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint size, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clBuffer1 = (float *)jBuffer1;
  float * clBuffer2 = (float *)jBuffer2;
  float * clBuffer3 = (float *)jBuffer3;
  const int clSize = (int)size;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernel(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clSize, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferIFFI(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint size, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int * clBuffer1 = (int *)jBuffer1;
  float * clBuffer2 = (float *)jBuffer2;
  float * clBuffer3 = (float *)jBuffer3;
  const int clSize = (int)size;
  const int clLength = (int)length;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernel(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clSize, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferFIFI(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint size, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clBuffer1 = (float *)jBuffer1;
  int * clBuffer2 = (int *)jBuffer2;
  float * clBuffer3 = (float *)jBuffer3;
  const int clSize = (int)size;
  const int clLength = (int)length;

  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernel(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clSize, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_executeKernelWithTypesBufferIIII(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint size, jint length, jstring jKernelString)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int * clBuffer1 = (int *)jBuffer1;
  int * clBuffer2 = (int *)jBuffer2;
  int * clBuffer3 = (int *)jBuffer3;
  const int clSize = (int)size;
  const int clLength = (int)length;

  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);

  executeKernel(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clSize, clLength, kernel);

  env->ReleaseStringUTFChars(jKernelString, kernel);
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_executeKernel(JNIEnv *env, jclass vsclazz, jlong jKernel, jlong jCommandQueue, jint length)) {
  cl_int error = 0;
  const cl_kernel clKernel = (cl_kernel)jKernel;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  const int clLength = (int)length;

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", "exec");
  clFlush(clCommandQueue);
  clFinish(clCommandQueue);

  clReleaseKernel(clKernel);
} JVM_END

JVM_ENTRY(jlong, ExecBufferSupport_createExecKernel(JNIEnv *env, jclass vsclazz, jlong jProgram)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const char * kernelName = "exec";
  const cl_kernel clKernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);
  return (jlong)clKernel;
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jKernel, jlong buffer, jint argumentNumber)) {
  cl_int error = 0;
  const cl_kernel clKernel = (cl_kernel)jKernel;
  float* clBuffer = (float *)buffer;
  const int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArgSVMPointer(clKernel, clArgumentNumber, clBuffer);
  handleError(error, "clSetKernelArg", "kernelBuilder");
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentInteger(JNIEnv *env, jclass vsclazz, jlong jKernel, jint value, jint argumentNumber)) {
  cl_int error = 0;
  const cl_kernel clKernel = (cl_kernel)jKernel;
  const int clValue = (int)value;
  const int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArg(clKernel, clArgumentNumber, sizeof(int), &clValue);
  handleError(error, "clSetKernelArg", "exec");
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentFloat(JNIEnv *env, jclass vsclazz, jlong jKernel, jfloat value, jint argumentNumber)) {
  cl_int error = 0;
  const cl_kernel clKernel = (cl_kernel)jKernel;
  const float clValue = (float)value;
  const int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArg(clKernel, clArgumentNumber, sizeof(float), &clValue);
  handleError(error, "clSetKernelArg", "exec");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_toInt(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "toInt", &error);
  handleError(error, "clCreateKernel", "exec");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg", "exec");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg", "exec");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", "exec");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_toFloat(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  const int clLength = (int)length;

  const char * kernelName = "toFloat";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyArea(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint offset, jint thisWidth, jint factorWidth, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clOffset = (int)offset;
  const int clThisWidth = (int)thisWidth;
  const int clFactorWidth = (int)factorWidth;
  const int clLength = (int)length;

  const char * kernelName = "multArea";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg2", kernelName);

  error = clSetKernelArg(kernel, 3, sizeof(int), &clOffset);
  handleError(error, "clSetKernelArg3", kernelName);

  error = clSetKernelArg(kernel, 4, sizeof(int), &clThisWidth);
  handleError(error, "clSetKernelArg4", kernelName);

  error = clSetKernelArg(kernel, 5, sizeof(int), &clFactorWidth);
  handleError(error, "clSetKernelArg5", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_eachAreaFMA(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint width, jint kernelWidth, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clWidth = (int)width;
  const int clKernelWidth = (int)kernelWidth;
  const int clLength = (int)length;

  const char * kernelName = "eachAreaFMA";
  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg0", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg1", kernelName);

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg22", kernelName);

  error = clSetKernelArg(kernel, 3, sizeof(int), &clWidth);
  handleError(error, "clSetKernelArg3", kernelName);

  error = clSetKernelArg(kernel, 4, sizeof(int), &clKernelWidth);
  handleError(error, "clSetKernelArg4", kernelName);

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel", kernelName);

  clReleaseKernel(kernel);
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
    {CC "GetMaxWorkGroupSize",   CC "(J)I", FN_PTR(SVMBufferSupport_getMaxWorkGroupSize)},
    {CC "CreateReadWriteFloatSVMBuffer",   CC "(JI)J", FN_PTR(SVMBufferSupport_createReadWriteFloatSVMBuffer)},
    {CC "CreateReadWriteIntSVMBuffer",   CC "(JI)J", FN_PTR(SVMBufferSupport_createReadWriteIntSVMBuffer)},
    {CC "CopyToArray",   CC "(JJJ[I)V", FN_PTR(SVMBufferSupport_copyToIntArray)},
    {CC "CopyToArray",   CC "(JJJ[F)V", FN_PTR(SVMBufferSupport_copyToFloatArray)},
    {CC "CopyToArray",   CC "(JJJ[FIII)V", FN_PTR(SVMBufferSupport_copyRangeToFloatArray)},
    {CC "CopyToArray",   CC "(JJJ[IIII)V", FN_PTR(SVMBufferSupport_copyRangeToIntArray)},
    {CC "CopyFromArray",   CC "(JJ[I)J", FN_PTR(SVMBufferSupport_copyFromArray)},
    {CC "CopyFromArray",   CC "(JJ[F)J", FN_PTR(SVMBufferSupport_copyFromFloatArray)},
    {CC "Fill",   CC "(JJJ[F)J", FN_PTR(SVMBufferSupport_fill)},
    {CC "Fill",   CC "(JJJ[I)J", FN_PTR(SVMBufferSupport_fillInt)},
    {CC "ReleaseSVMBuffer",   CC "(JJJ)V", FN_PTR(SVMBufferSupport_releaseSVMBuffer)},
    {CC "ReleaseSVMBufferInt",   CC "(JJJ)V", FN_PTR(SVMBufferSupport_releaseSVMBufferInt)},
    {CC "MatrixFmaSVMBuffer",   CC "(JJJJJIIII)V", FN_PTR(SVMBufferSupport_matrixFmaSVMBuffer)},
    {CC "FmaSVMBuffer",   CC "(JJJJJJI)V", FN_PTR(SVMBufferSupport_fmaSVMBuffer)},
    {CC "SumReduceFLOAT",   CC "(JJJJII)F", FN_PTR(SVMBufferSupport_sumReduceFLOAT)},
    {CC "SumReduceINT",   CC "(JJJJII)I", FN_PTR(SVMBufferSupport_sumReduceINT)},
    {CC "executeKernelWithTypesFFF",   CC "(JJJFJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesFFF)},
    {CC "executeKernelWithTypesIFF",   CC "(JJJFJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesIFF)},
    {CC "executeKernelWithTypesFIF",   CC "(JJJIJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesFIF)},
    {CC "executeKernelWithTypesIII",   CC "(JJJIJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesIII)},
    {CC "executeKernelWithTypesBufferFFF",   CC "(JJJJJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferFFF)},
    {CC "executeKernelWithTypesBufferIFF",   CC "(JJJJJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferIFF)},
    {CC "executeKernelWithTypesBufferFIF",   CC "(JJJJJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferFIF)},
    {CC "executeKernelWithTypesBufferIII",   CC "(JJJJJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferIII)},
    {CC "executeKernelWithTypesBufferFF",   CC "(JJJJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferFF)},
    {CC "executeKernelWithTypesBufferIF",   CC "(JJJJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferIF)},
    {CC "executeKernelWithTypesBufferFI",   CC "(JJJJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferFI)},
    {CC "executeKernelWithTypesBufferII",   CC "(JJJJILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferII)},
    {CC "MultiplyRange",   CC "(JJJIJIJI)V", FN_PTR(SVMBufferSupport_multiplyRange)},
    {CC "Blend",   CC "(JJJJJJI)V", FN_PTR(SVMBufferSupport_blend)},
    {CC "executeKernelWithTypesBufferIIII",   CC "(JJJJJIILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferIIII)},
    {CC "executeKernelWithTypesBufferFFFI",   CC "(JJJJJIILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferFFFI)},
    {CC "executeKernelWithTypesBufferFIFI",   CC "(JJJJJIILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferFIFI)},
    {CC "executeKernelWithTypesBufferIFFI",   CC "(JJJJJIILjava/lang/String;)V", FN_PTR(SVMBufferSupport_executeKernelWithTypesBufferIFFI)},
    {CC "ExecuteKernel",   CC "(JJI)V", FN_PTR(ExecBufferSupport_executeKernel)},
    {CC "CreateKernel",   CC "(J)J", FN_PTR(ExecBufferSupport_createExecKernel)},
    {CC "SetKernelArgument",   CC "(JJI)V", FN_PTR(ExecBufferSupport_setKernelArgumentSVMBuffer)},
    {CC "SetKernelArgument",   CC "(JII)V", FN_PTR(ExecBufferSupport_setKernelArgumentInteger)},
    {CC "SetKernelArgument",   CC "(JFI)V", FN_PTR(ExecBufferSupport_setKernelArgumentFloat)},
    {CC "ToInt",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_toInt)},
    {CC "ToFloat",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_toFloat)},
    {CC "MultiplyArea",   CC "(JJJJJIIII)V", FN_PTR(SVMBufferSupport_multiplyArea)},
    {CC "EachAreaFMA",   CC "(JJJJJIII)V", FN_PTR(SVMBufferSupport_eachAreaFMA)},
};

JVM_ENTRY(void, JVM_RegisterSVMBufferSupportMethods(JNIEnv* env, jclass vsclass)) {
  ThreadToNativeFromVM ttnfv(thread);

  int ok = env->RegisterNatives(vsclass, jdk_internal_vm_vector_SVMBufferSupport_methods, sizeof(jdk_internal_vm_vector_SVMBufferSupport_methods)/sizeof(JNINativeMethod));
  guarantee(ok == 0, "register jdk.internal.vm.vector.SVMBufferSupport natives");
} JVM_END
