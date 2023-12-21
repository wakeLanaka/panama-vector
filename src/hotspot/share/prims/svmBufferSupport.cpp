#include "precompiled.hpp"
#include "prims/svmBufferSupport.hpp"
#include "runtime/interfaceSupport.inline.hpp"
#include "openclHelper.hpp"
#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include <cmath>

JVM_ENTRY(jlong, SVMBufferSupport_createContext(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_int error = 0;
  cl_device_id clDevice = (cl_device_id) jDevice;
  cl_context clContext = clCreateContext(0, 1, &clDevice, NULL, NULL, &error);
  handleError(error, "clCreateContext");

  return (jlong) clContext;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseContext(JNIEnv *env, jclass vsclazz, jlong jContext)) {
  cl_int error = 0;
  cl_context clContext = (cl_context) jContext;
  if (clContext) {
    error = clReleaseContext(clContext);
    handleError(error, "ReleaseContext");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createProgram(JNIEnv *env, jclass vsclazz, jlong jContext, jstring jKernelString)) {
  cl_int error = 0;
  cl_context clContext = (cl_context)jContext;
  const char * kernel = env->GetStringUTFChars(jKernelString, NULL);
  const char * clKernelString[] = {kernel};
  const size_t clKernelLength = (size_t)env->GetStringLength(jKernelString);

  cl_program clProgram = clCreateProgramWithSource(clContext, 1, clKernelString, &clKernelLength, &error);
  handleError(error, "clCreateProgramWithSource");

  error = clBuildProgram(clProgram, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
  handleError(error, "clBuildProgram");
  env->ReleaseStringUTFChars(jKernelString, kernel);

  return (jlong)clProgram;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseProgram(JNIEnv *env, jclass vsclazz, jlong jProgram)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program) jProgram;
  if (clProgram) {
    error = clReleaseProgram(clProgram);
    handleError(error, "ReleaseProgram");
  }
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createCommandQueue(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jDevice)) {
  cl_int error = 0;
  cl_context clContext = (cl_context)jContext;
  cl_device_id clDevice = (cl_device_id)jDevice;
  cl_command_queue_properties properties[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue clCommandQueue = clCreateCommandQueueWithProperties(clContext, clDevice, properties, &error);
  handleError(error, "createCommandQueue");
  return (jlong)clCommandQueue;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseCommandQueue(JNIEnv *env, jclass vsclazz, jlong jCommandQueue)) {
  cl_int error = 0;
  cl_command_queue clCommandQueue = (cl_command_queue) jCommandQueue;
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
  cl_device_id clDevice = (cl_device_id)jDevice;
  if (clDevice) {
    error = clReleaseDevice(clDevice);
    handleError(error, "clReleaseDevice");
  }
} JVM_END

JVM_ENTRY(jint, SVMBufferSupport_getMaxWorkGroupSize(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_device_id clDevice = (cl_device_id)jDevice;
  size_t maxWorkGroupSize = 0;
  clGetDeviceInfo(clDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
  return (jint)maxWorkGroupSize;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createReadWriteFloatSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jint length)) {
  cl_int error = 0;
  cl_context clContext = (cl_context)jContext;
  int clLength = (int)length;

  float * svmBuffer = (float *)clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(float) * clLength, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_createReadWriteIntSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jint length)) {
  cl_int error = 0;
  cl_context clContext = (cl_context)jContext;
  int clLength = (int)length;
  int * svmBuffer = (int *)clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(int) * clLength, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_addSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_add", NULL);

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
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  cl_program clProgram = (cl_program)jProgram;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clK = (int)K;
  int clN = (int)N;
  int clk = (int)k;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "matrix_fma", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  float* clBuffer4 = (float *)jBuffer4;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_fma", &error);
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
  cl_context clContext = (cl_context)jContext;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer = (float *)jBuffer;
  int clLength = (int)length;
  int localSize = (int)maxWorkGroupSize;

  size_t workGroups = std::ceil(clLength / (float)localSize);
  float* sums = (float * )clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(float) * workGroups, 0);

  cl_kernel kernel = clCreateKernel(clProgram, "sumreduce", &error);
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
  float jarrayElements[workGroups];
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayElements, sums, sizeof(float) * workGroups, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, sums, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");

  float sum = 0.0f;
  for(size_t i = 0; i < workGroups; i++) {
    sum += jarrayElements[i];
  }

  clReleaseKernel(kernel);
  clSVMFree(clContext, sums);
  return (jfloat)sum;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_copyFromArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jintArray jArray)) {
  cl_int error = 0;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

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
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

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
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer = (float *)jBuffer;

  jsize length = env->GetArrayLength(jArray);

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
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer = (int *)jBuffer;

  jsize length = env->GetArrayLength(jArray);

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
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

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
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

  float* clBuffer = (float *)jBuffer;


  error = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap3");
  float * jarrayBody = env->GetFloatArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, clBuffer, sizeof(float) * length, 0, 0, 0);
  error = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap3");
  env->ReleaseFloatArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer)) {
  cl_int error = 0;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* svmBuffer = (float *)jBuffer;
  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  error = clFlush(clCommandQueue);
  handleError(error, "clFlush");
  clSVMFree(clContext, svmBuffer);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseSVMBufferInt(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer)) {
  cl_int error = 0;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* svmBuffer = (int *)jBuffer;
  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  error = clFlush(clCommandQueue);
  handleError(error, "clFlush");
  clSVMFree(clContext, svmBuffer);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_subtract(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float*)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "subtract", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float clMinuend = (float)jMinuend;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "subtraction_subtrahend", &error);
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

JVM_ENTRY(void, SVMBufferSupport_multiply(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jfloat jFactor, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float clFactor = (float)jFactor;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "multiply", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(float), &clFactor);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_multiply", &error);
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

JVM_ENTRY(void, SVMBufferSupport_multiplyBufferInt(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  int* clBuffer2 = (int *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_multiplyInt", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer1);
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

JVM_ENTRY(void, SVMBufferSupport_multiplyRange(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jint index1, jlong jBuffer2, jint index2, jlong jBuffer3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clIndex1 = (int)index1;
  int clIndex2 = (int)index2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "multiplyrange", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "sqrt", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float clDivisor = (float)jDivisor;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "division", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_division", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "log", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "exp", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "abs", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float clComparee = (float)jComparee;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "compareGT", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clMask = (float *)jMask;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "blend", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clXBuffer = (float *)xBuffer;
  float* clCallBuffer = (float *)callBuffer;
  float* clPutBuffer = (float *)putBuffer;
  float* clTBuffer = (float *)tBuffer;
  float * clS0Buffer = (float *)s0;
  float clSig = (float)sig;
  float clR = (float)r;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "blackscholes", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "cos", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "sinx", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  float * clB3 = (float *)b3;
  float * clB4 = (float *)b4;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "dft", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  int clSize = (int)size;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "multiplyRepeat", &error);
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

JVM_ENTRY(void, SVMBufferSupport_forSum(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jfloat v1, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  float clv1 = (float)v1;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "forSum", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(float), &clv1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 3, sizeof(int), &clLength);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_executeKernel(JNIEnv *env, jclass vsclazz, jlong jKernel, jlong jCommandQueue, jint length)) {
  cl_int error = 0;
  cl_kernel clKernel = (cl_kernel)jKernel;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int clLength = (int)length;

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
  cl_program clProgram = (cl_program)jProgram;
  cl_kernel clKernel = clCreateKernel(clProgram, "exec", &error);
  handleError(error, "clCreateKernel");
  return (jlong)clKernel;
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jKernel, jlong buffer, jint argumentNumber)) {
  cl_int error = 0;
  cl_kernel clKernel = (cl_kernel)jKernel;
  float* clBuffer = (float *)buffer;
  int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArgSVMPointer(clKernel, clArgumentNumber, clBuffer);
  handleError(error, "clSetKernelArg");
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentInteger(JNIEnv *env, jclass vsclazz, jlong jKernel, jint value, jint argumentNumber)) {
  cl_int error = 0;
  cl_kernel clKernel = (cl_kernel)jKernel;
  int clValue = (int)value;
  int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArg(clKernel, clArgumentNumber, sizeof(int), &clValue);
  handleError(error, "clSetKernelArg");
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentFloat(JNIEnv *env, jclass vsclazz, jlong jKernel, jfloat value, jint argumentNumber)) {
  cl_int error = 0;
  cl_kernel clKernel = (cl_kernel)jKernel;
  float clValue = (float)value;
  int clArgumentNumber = (int)argumentNumber;

  error = clSetKernelArg(clKernel, clArgumentNumber, sizeof(float), &clValue);
  handleError(error, "clSetKernelArg");
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_repeat1(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint repetition, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  int clRepetition = (int)repetition;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "repeat1", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 2, sizeof(int), &clRepetition);
  handleError(error, "clSetKernelArg");

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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  int clRepetition = (int)repetition;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "repeat2", &error);
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
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  int clAmount = (int)amount;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "ashr", &error);
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

JVM_ENTRY(void, SVMBufferSupport_and(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint value, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)b1;
  float* clBuffer2 = (float *)b2;
  int clValue = (int)value;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "bitwiseand", &error);
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

JVM_ENTRY(void, SVMBufferSupport_multiplyArea(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jlong b3, jint offset, jint thisWidth, jint factorWidth, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* clBuffer1 = (int *)b1;
  float* clBuffer2 = (float *)b2;
  float* clBuffer3 = (float *)b3;
  int clOffset = (int)offset;
  int clThisWidth = (int)thisWidth;
  int clFactorWidth = (int)factorWidth;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "multArea", &error);
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
    {CC "Multiply",   CC "(JJJJFI)V", FN_PTR(SVMBufferSupport_multiply)},
    {CC "Multiply",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_multiplyBuffer)},
    {CC "MultiplyInt",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_multiplyBufferInt)},
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
    {CC "ForSum",   CC "(JJJJFI)V", FN_PTR(SVMBufferSupport_forSum)},
    {CC "ExecuteKernel",   CC "(JJI)V", FN_PTR(ExecBufferSupport_executeKernel)},
    {CC "CreateKernel",   CC "(J)J", FN_PTR(ExecBufferSupport_createExecKernel)},
    {CC "SetKernelArgument",   CC "(JJI)V", FN_PTR(ExecBufferSupport_setKernelArgumentSVMBuffer)},
    {CC "SetKernelArgument",   CC "(JII)V", FN_PTR(ExecBufferSupport_setKernelArgumentInteger)},
    {CC "SetKernelArgument",   CC "(JFI)V", FN_PTR(ExecBufferSupport_setKernelArgumentFloat)},
    {CC "Repeat1",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_repeat1)},
    {CC "Repeat2",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_repeat2)},
    {CC "Ashr",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_ashr)},
    {CC "And",   CC "(JJJJII)V", FN_PTR(SVMBufferSupport_and)},
    {CC "MultiplyArea",   CC "(JJJJJIIII)V", FN_PTR(SVMBufferSupport_multiplyArea)},
};

JVM_ENTRY(void, JVM_RegisterSVMBufferSupportMethods(JNIEnv* env, jclass vsclass)) {
  ThreadToNativeFromVM ttnfv(thread);

  int ok = env->RegisterNatives(vsclass, jdk_internal_vm_vector_SVMBufferSupport_methods, sizeof(jdk_internal_vm_vector_SVMBufferSupport_methods)/sizeof(JNINativeMethod));
  guarantee(ok == 0, "register jdk.internal.vm.vector.SVMBufferSupport natives");
} JVM_END
