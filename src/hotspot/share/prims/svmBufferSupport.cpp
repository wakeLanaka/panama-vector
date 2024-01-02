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
  handleError(error, "clCreateContext");

  return (jlong) clContext;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseContext(JNIEnv *env, jclass vsclazz, jlong jContext)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context) jContext;
  if (clContext) {
    error = clReleaseContext(clContext);
    handleError(error, "ReleaseContext");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createProgram(JNIEnv *env, jclass vsclazz, jlong jContext, jstring jKernelString)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);
  const char * clKernelString[] = {kernel};
  const size_t clKernelLength = (size_t)env->GetStringLength(jKernelString);

  const cl_program clProgram = clCreateProgramWithSource(clContext, 1, clKernelString, &clKernelLength, &error);
  handleError(error, "clCreateProgramWithSource");

  error = clBuildProgram(clProgram, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
  handleError(error, "clBuildProgram");
  env->ReleaseStringUTFChars(jKernelString, kernel);

  return (jlong)clProgram;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseProgram(JNIEnv *env, jclass vsclazz, jlong jProgram)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program) jProgram;
  if (clProgram) {
    error = clReleaseProgram(clProgram);
    handleError(error, "ReleaseProgram");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createCommandQueue(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jDevice)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_device_id clDevice = (cl_device_id)jDevice;
  const cl_command_queue clCommandQueue = clCreateCommandQueueWithProperties(clContext, clDevice, NULL, &error);
  handleError(error, "createCommandQueue");
  return (jlong)clCommandQueue;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseCommandQueue(JNIEnv *env, jclass vsclazz, jlong jCommandQueue)) {
  cl_int error = 0;
  const cl_command_queue clCommandQueue = (cl_command_queue) jCommandQueue;
  if (clCommandQueue) {
    error = clReleaseCommandQueue(clCommandQueue);
    handleError(error, "clReleaseCommandQueue");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createDevice(JNIEnv *env, jclass vsclazz)) {
  cl_int error = 0;
  cl_platform_id clPlatform = NULL;
  cl_device_id clDevice = NULL;

  error = clGetPlatformIDs(1, &clPlatform, NULL);
  handleError(error, "clGetPlatformIDs");

  error = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);
  handleError(error, "clGetDeviceIDs");

  return (jlong)clDevice;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseDevice(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_int error = 0;
  const cl_device_id clDevice = (cl_device_id)jDevice;
  if (clDevice) {
    error = clReleaseDevice(clDevice);
    handleError(error, "clReleaseDevice");
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

JVM_ENTRY(void, SVMBufferSupport_addSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "vector_add", NULL);

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
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

  const cl_kernel kernel = clCreateKernel(clProgram, "matrix_fma", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 3, sizeof(int), &clK);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 4, sizeof(int), &clN);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 5, sizeof(int), &clk);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

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

  const cl_kernel kernel = clCreateKernel(clProgram, "vector_fma", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 3, clBuffer4);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(jfloat, SVMBufferSupport_sumReduce(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jProgram, jlong jCommandQueue, jlong jBuffer, jint maxWorkGroupSize, jint length)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer = (float *)jBuffer;
  const int clLength = (int)length;
  const int localSize = (int)maxWorkGroupSize;

  const jsize workGroups = (jsize)std::ceil(clLength / (float)localSize);
  float* sums = (float * )clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(float) * workGroups, 0);

  const cl_kernel kernel = clCreateKernel(clProgram, "sumreduce", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, sums);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(float) * localSize, NULL);
  handleError(error, "clSetKernelArg");

  size_t local_item_size[] = {(size_t)localSize};
  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, local_item_size, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, sums, sizeof(float) * workGroups, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");
  jfloatArray jSums = env->NewFloatArray(workGroups);
  jfloat * jsums = env->GetFloatArrayElements(jSums, NULL);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jsums, sums, sizeof(float) * workGroups, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, sums, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");

  float sum = 0.0f;
  for(jsize i = 0; i < workGroups; i++) {
    sum += (float)jsums[i];
  }

  env->ReleaseFloatArrayElements(jSums, jsums, 0);
  clReleaseKernel(kernel);
  clSVMFree(clContext, sums);
  return (jfloat)sum;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_copyFromArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jintArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  const jsize length = env->GetArrayLength(jArray);

  int * svmBuffer = (int *) clSVMAlloc((cl_context)clContext, CL_MEM_READ_ONLY, sizeof(int) * length, 0);
  int * jarrayElements = env->GetIntArrayElements(jArray, 0);
  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap4");
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, svmBuffer, jarrayElements, sizeof(int) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap4");
  env->ReleaseIntArrayElements(jArray, jarrayElements, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_copyFromFloatArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jfloatArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  const jsize length = env->GetArrayLength(jArray);

  float * svmBuffer = (float *) clSVMAlloc((cl_context)clContext, CL_MEM_READ_WRITE, sizeof(float) * length, 0);
  float * jarrayElements = env->GetFloatArrayElements(jArray, 0);
  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap1");
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, svmBuffer, jarrayElements, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap1");
  env->ReleaseFloatArrayElements(jArray, jarrayElements, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_fill(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jfloatArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer = (float *)jBuffer;

  const jsize length = env->GetArrayLength(jArray);

  float * jarrayElements = env->GetFloatArrayElements(jArray, 0);
  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, clBuffer, jarrayElements, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");
  env->ReleaseFloatArrayElements(jArray, jarrayElements, 0);

  return (jlong)clBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_fillInt(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jintArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer = (int *)jBuffer;

  const jsize length = env->GetArrayLength(jArray);

  int * jarrayElements = env->GetIntArrayElements(jArray, 0);
  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, clBuffer, jarrayElements, sizeof(int) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");
  env->ReleaseIntArrayElements(jArray, jarrayElements, 0);

  return (jlong)clBuffer;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToIntArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jintArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  const jsize length = env->GetArrayLength(jArray);

  int* svmBuffer = (int *)jBuffer;

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");
  int * jarrayBody = env->GetIntArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, svmBuffer, sizeof(int) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");
  env->ReleaseIntArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToFloatArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jfloatArray jArray)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  const jsize length = env->GetArrayLength(jArray);

  float* clBuffer = (float *)jBuffer;


  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");
  float * jarrayBody = env->GetFloatArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, clBuffer, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");
  env->ReleaseFloatArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToFloatArrayII(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jfloatArray jArray, jint bufferLength, jint length, jint offset)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  float* clBuffer = (float *)jBuffer;

  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * bufferLength, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");
  float * jarrayBody = env->GetFloatArrayElements(jArray, 0);
  float * offsetPointer = jarrayBody + offset;
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, offsetPointer, clBuffer, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");
  env->ReleaseFloatArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* svmBuffer = (float *)jBuffer;
  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  error = clFlush(clCommandQueue);
  handleError(error, "clFlush");
  clSVMFree(clContext, svmBuffer);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseSVMBufferInt(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer)) {
  cl_int error = 0;
  const cl_context clContext = (cl_context)jContext;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* svmBuffer = (int *)jBuffer;
  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  error = clFlush(clCommandQueue);
  handleError(error, "clFlush");
  clSVMFree(clContext, svmBuffer);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_subtract(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float*)jBuffer3;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "subtract", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_subtractMinuend(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jfloat jMinuend, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float clMinuend = (float)jMinuend;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "subtraction_subtrahend", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(float), &clMinuend);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

template<typename T1, typename T2, typename T3>
void executeMultiplyValue(cl_program clProgram, cl_command_queue clCommandQueue, T1 clBuffer1, const T3 clFactor, T2 clBuffer2, int clLength, const char * kernelName){
  cl_int error = 0;

  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 1, sizeof(T2), &clFactor);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
}

JVM_ENTRY(void, SVMBufferSupport_multiplyFFF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jfloat jFactor, jlong jBuffer2, jint length)) {
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  jfloat* clBuffer1 = (jfloat *)jBuffer1;
  const jfloat clFactor = (jfloat)jFactor;
  jfloat* clBuffer2 = (jfloat *)jBuffer2;
  const jint clLength = (jint)length;

  executeMultiplyValue(clProgram, clCommandQueue, clBuffer1, clFactor, clBuffer2, clLength, "multiplyFFF");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyIFF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jfloat jFactor, jlong jBuffer2, jint length)) {
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  jint* clBuffer1 = (jint *)jBuffer1;
  const jfloat clFactor = (jfloat)jFactor;
  jfloat* clBuffer2 = (jfloat *)jBuffer2;
  const jint clLength = (jint)length;

  executeMultiplyValue(clProgram, clCommandQueue, clBuffer1, clFactor, clBuffer2, clLength, "multiplyIFF");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyIII(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jint jFactor, jlong jBuffer2, jint length)) {
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  jint* clBuffer1 = (jint *)jBuffer1;
  const jint clFactor = (jint)jFactor;
  jint* clBuffer2 = (jint *)jBuffer2;
  const jint clLength = (jint)length;

  executeMultiplyValue(clProgram, clCommandQueue, clBuffer1, clFactor, clBuffer2, clLength, "multiplyIII");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyFIF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jint jFactor, jlong jBuffer2, jint length)) {
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  jfloat* clBuffer1 = (jfloat *)jBuffer1;
  const jint clFactor = (jint)jFactor;
  jfloat* clBuffer2 = (jfloat *)jBuffer2;
  const jint clLength = (jint)length;

  executeMultiplyValue(clProgram, clCommandQueue, clBuffer1, clFactor, clBuffer2, clLength, "multiplyFIF");
} JVM_END

template<typename T1, typename T2, typename T3>
void executeMultiplyBuffer(cl_program clProgram, cl_command_queue clCommandQueue, T1 clBuffer1, T2 clBuffer2, const T3 clResult, int clLength, const char * kernelName){
  cl_int error = 0;

  const cl_kernel kernel = clCreateKernel(clProgram, kernelName, &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clResult);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
}

JVM_ENTRY(void, SVMBufferSupport_multiplyBufferFFF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;

  executeMultiplyBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clLength, "mulBufferFFF");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyBufferFIF(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;

  executeMultiplyBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clLength, "mulBufferFIF");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyBufferIII(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  int* clBuffer3 = (int *)jBuffer3;
  const int clLength = (int)length;

  executeMultiplyBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clLength, "mulBufferIII");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyVector(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;

  executeMultiplyBuffer(clProgram, clCommandQueue, clBuffer1, clBuffer2, clBuffer3, clLength, "mulVector");
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

  const cl_kernel kernel = clCreateKernel(clProgram, "multiplyrange", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 3, sizeof(int), &clIndex1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 4, sizeof(int), &clIndex2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END


JVM_ENTRY(void, SVMBufferSupport_sqrt(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "sqrt", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_division(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jfloat jDivisor, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float clDivisor = (float)jDivisor;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "division", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(float), &clDivisor);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_divisionBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "vector_division", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_log(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "log", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_exp(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "exp", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_abs(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "abs", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_compareGT(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jfloat jComparee, jlong jBuffer2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float clComparee = (float)jComparee;
  float* clBuffer2 = (float *)jBuffer2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "compareGT", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(float), &clComparee);
  handleError(error, "clSetKernelArg");


  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

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

  const cl_kernel kernel = clCreateKernel(clProgram, "blend", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clMask);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 3, clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_blackscholes(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jfloat sig, jfloat r, jlong xBuffer, jlong callBuffer, jlong putBuffer, jlong tBuffer, jlong s0, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clXBuffer = (float *)xBuffer;
  float* clCallBuffer = (float *)callBuffer;
  float* clPutBuffer = (float *)putBuffer;
  float* clTBuffer = (float *)tBuffer;
  float * clS0Buffer = (float *)s0;
  float clSig = (float)sig;
  float clR = (float)r;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "blackscholes", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(float), &clSig);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 1, sizeof(float), &clR);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clXBuffer);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 3, clCallBuffer);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 4, clPutBuffer);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 5, clTBuffer);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 6, clS0Buffer);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_cos(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "cos", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_sin(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "sinx", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_dft(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jlong b3, jlong b4, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  float * clB3 = (float *)b3;
  float * clB4 = (float *)b4;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "dft", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clB3);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 3, clB4);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 4, sizeof(int), &clLength);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyRepeat(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint size, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  const int clSize = (int)size;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "multiplyRepeat", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clSize);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_executeKernel(JNIEnv *env, jclass vsclazz, jlong jKernel, jlong jCommandQueue, jint length)) {
  cl_int error = 0;
  const cl_kernel clKernel = (cl_kernel)jKernel;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  const int clLength = (int)length;

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");
  clFlush(clCommandQueue);
  clFinish(clCommandQueue);

  clReleaseKernel(clKernel);
} JVM_END

JVM_ENTRY(jlong, ExecBufferSupport_createExecKernel(JNIEnv *env, jclass vsclazz, jlong jProgram)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_kernel clKernel = clCreateKernel(clProgram, "exec", &error);
  handleError(error, "clCreateKernel");
  return (jlong)clKernel;
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jKernel, jlong buffer, jint argumentNumber)) {
  cl_int error = 0;
  const cl_kernel clKernel = (cl_kernel)jKernel;
  float* clBuffer = (float *)buffer;
  const int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArgSVMPointer(clKernel, clArgumentNumber, clBuffer);
  handleError(error, "clSetKernelArg");
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentInteger(JNIEnv *env, jclass vsclazz, jlong jKernel, jint value, jint argumentNumber)) {
  cl_int error = 0;
  const cl_kernel clKernel = (cl_kernel)jKernel;
  const int clValue = (int)value;
  const int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArg(clKernel, clArgumentNumber, sizeof(int), &clValue);
  handleError(error, "clSetKernelArg");
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentFloat(JNIEnv *env, jclass vsclazz, jlong jKernel, jfloat value, jint argumentNumber)) {
  cl_int error = 0;
  const cl_kernel clKernel = (cl_kernel)jKernel;
  const float clValue = (float)value;
  const int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArg(clKernel, clArgumentNumber, sizeof(float), &clValue);
  handleError(error, "clSetKernelArg");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_repeat1(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint repetition, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  const int clRepetition = (int)repetition;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "repeat1", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clRepetition);
  handleError(error, "clSetKernelArg");

                     // TODO use get_global_size()
  error = clSetKernelArg(kernel, 3, sizeof(int), &clLength);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_repeat2(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint repetition, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  const int clRepetition = (int)repetition;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "repeat2", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clRepetition);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_ashr(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint amount, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)b1;
  int* clBuffer2 = (int *)b2;
  const int clAmount = (int)amount;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "ashr", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clAmount);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_lshl(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint amount, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)b1;
  int* clBuffer2 = (int *)b2;
  const int clAmount = (int)amount;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "lshl", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clAmount);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_toInt(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  int* clBuffer2 = (int *)b2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "toInt", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_toFloat(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)b1;
  float* clBuffer2 = (float *)b2;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "toFloat", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_and(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint value, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)b1;
  int* clBuffer2 = (int *)b2;
  const int clValue = (int)value;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "bitwiseand", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clValue);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_or(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jlong b3, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)b1;
  int* clBuffer2 = (int *)b2;
  int* clBuffer3 = (int *)b3;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "vector_bitwiseor", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_max(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jfloat value, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  const float clValue = (float)value;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "vector_max", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(float), &clValue);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_min(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jfloat value, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  const float clValue = (float)value;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "vector_min", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(float), &clValue);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_ror(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint value, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  const int clValue = (int)value;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "ror", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clValue);
  handleError(error, "clSetKernelArg");

                     // TODO use get_global_size()
  error = clSetKernelArg(kernel, 3, sizeof(int), &clLength);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_rol(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint value, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  const int clValue = (int)value;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "rol", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clValue);
  handleError(error, "clSetKernelArg");

                         // TODO use get_global_size()
  error = clSetKernelArg(kernel, 3, sizeof(int), &clLength);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyArea(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jlong b3, jint offset, jint thisWidth, jint factorWidth, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)b1;
  float* clBuffer2 = (float *)b2;
  float* clBuffer3 = (float *)b3;
  const int clOffset = (int)offset;
  const int clThisWidth = (int)thisWidth;
  const int clFactorWidth = (int)factorWidth;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "multArea", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 3, sizeof(int), &clOffset);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 4, sizeof(int), &clThisWidth);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 5, sizeof(int), &clFactorWidth);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_eachAreaFMA(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jlong b3, jint width, jint kernelWidth, jint length)) {
  cl_int error = 0;
  const cl_program clProgram = (cl_program)jProgram;
  const cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  float* clBuffer3 = (float *)b3;
  const int clWidth = (int)width;
  const int clKernelWidth = (int)kernelWidth;
  const int clLength = (int)length;

  const cl_kernel kernel = clCreateKernel(clProgram, "eachAreaFMA", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 3, sizeof(int), &clWidth);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 4, sizeof(int), &clKernelWidth);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

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
    {CC "AddSVMBuffer",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_addSVMBuffer)},
    {CC "CopyToArray",   CC "(JJJ[I)V", FN_PTR(SVMBufferSupport_copyToIntArray)},
    {CC "CopyToArray",   CC "(JJJ[F)V", FN_PTR(SVMBufferSupport_copyToFloatArray)},
    {CC "CopyToArray",   CC "(JJJ[FIII)V", FN_PTR(SVMBufferSupport_copyToFloatArrayII)},
    {CC "CopyFromArray",   CC "(JJ[I)J", FN_PTR(SVMBufferSupport_copyFromArray)},
    {CC "CopyFromArray",   CC "(JJ[F)J", FN_PTR(SVMBufferSupport_copyFromFloatArray)},
    {CC "Fill",   CC "(JJJ[F)J", FN_PTR(SVMBufferSupport_fill)},
    {CC "Fill",   CC "(JJJ[I)J", FN_PTR(SVMBufferSupport_fillInt)},
    {CC "ReleaseSVMBuffer",   CC "(JJJ)V", FN_PTR(SVMBufferSupport_releaseSVMBuffer)},
    {CC "ReleaseSVMBufferInt",   CC "(JJJ)V", FN_PTR(SVMBufferSupport_releaseSVMBufferInt)},
    {CC "MatrixFmaSVMBuffer",   CC "(JJJJJIIII)V", FN_PTR(SVMBufferSupport_matrixFmaSVMBuffer)},
    {CC "FmaSVMBuffer",   CC "(JJJJJJI)V", FN_PTR(SVMBufferSupport_fmaSVMBuffer)},
    {CC "SumReduce",   CC "(JJJJII)F", FN_PTR(SVMBufferSupport_sumReduce)},
    {CC "Subtract",   CC "(JJJJFI)V", FN_PTR(SVMBufferSupport_subtractMinuend)},
    {CC "Subtract",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_subtract)},
    {CC "MulIFF",   CC "(JJJFJI)V", FN_PTR(SVMBufferSupport_multiplyIFF)},
    {CC "MulFFF",   CC "(JJJFJI)V", FN_PTR(SVMBufferSupport_multiplyFFF)},
    {CC "MulFIF",   CC "(JJJIJI)V", FN_PTR(SVMBufferSupport_multiplyFIF)},
    {CC "MulIII",   CC "(JJJIJI)V", FN_PTR(SVMBufferSupport_multiplyIII)},
    {CC "MulFFF",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_multiplyBufferFFF)},
    {CC "MulFIF",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_multiplyBufferFIF)},
    {CC "MulIII",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_multiplyBufferIII)},
    {CC "MulVector",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_multiplyVector)},
    {CC "MultiplyRange",   CC "(JJJIJIJI)V", FN_PTR(SVMBufferSupport_multiplyRange)},
    {CC "Sqrt",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_sqrt)},
    {CC "Division",   CC "(JJJJFI)V", FN_PTR(SVMBufferSupport_division)},
    {CC "Division",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_divisionBuffer)},
    {CC "Log",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_log)},
    {CC "Exp",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_exp)},
    {CC "Abs",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_abs)},
    {CC "CompareGT",   CC "(JJJFJI)V", FN_PTR(SVMBufferSupport_compareGT)},
    {CC "Blend",   CC "(JJJJJJI)V", FN_PTR(SVMBufferSupport_blend)},
    {CC "BlackScholes",   CC "(JJFFJJJJJI)V", FN_PTR(SVMBufferSupport_blackscholes)},
    {CC "Cos",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_cos)},
    {CC "Sin",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_sin)},
    {CC "MultiplyInPlaceRepeat",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_multiplyRepeat)},
    {CC "DFT",   CC "(JJJJJJI)V", FN_PTR(SVMBufferSupport_dft)},
    {CC "ExecuteKernel",   CC "(JJI)V", FN_PTR(ExecBufferSupport_executeKernel)},
    {CC "CreateKernel",   CC "(J)J", FN_PTR(ExecBufferSupport_createExecKernel)},
    {CC "SetKernelArgument",   CC "(JJI)V", FN_PTR(ExecBufferSupport_setKernelArgumentSVMBuffer)},
    {CC "SetKernelArgument",   CC "(JII)V", FN_PTR(ExecBufferSupport_setKernelArgumentInteger)},
    {CC "SetKernelArgument",   CC "(JFI)V", FN_PTR(ExecBufferSupport_setKernelArgumentFloat)},
    {CC "Repeat1",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_repeat1)},
    {CC "Repeat2",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_repeat2)},
    {CC "Ashr",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_ashr)},
    {CC "Lshl",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_lshl)},
    {CC "ToInt",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_toInt)},
    {CC "ToFloat",   CC "(JJJJI)V", FN_PTR(SVMBufferSupport_toFloat)},
    {CC "And",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_and)},
    {CC "Or",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_or)},
    {CC "Max",   CC "(JJJJFI)V", FN_PTR(SVMBufferSupport_max)},
    {CC "Min",   CC "(JJJJFI)V", FN_PTR(SVMBufferSupport_min)},
    {CC "Ror",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_ror)},
    {CC "Rol",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_rol)},
    {CC "MultiplyArea",   CC "(JJJJJIIII)V", FN_PTR(SVMBufferSupport_multiplyArea)},
    {CC "EachAreaFMA",   CC "(JJJJJIII)V", FN_PTR(SVMBufferSupport_eachAreaFMA)},
};

JVM_ENTRY(void, JVM_RegisterSVMBufferSupportMethods(JNIEnv* env, jclass vsclass)) {
  ThreadToNativeFromVM ttnfv(thread);

  int ok = env->RegisterNatives(vsclass, jdk_internal_vm_vector_SVMBufferSupport_methods, sizeof(jdk_internal_vm_vector_SVMBufferSupport_methods)/sizeof(JNINativeMethod));
  guarantee(ok == 0, "register jdk.internal.vm.vector.SVMBufferSupport natives");
} JVM_END
