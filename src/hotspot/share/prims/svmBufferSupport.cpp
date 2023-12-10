#include "prims/svmBufferSupport.hpp"
#include "runtime/interfaceSupport.inline.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include "openclHelper.hpp"
#include <iostream>

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
  handleError(svmError, "clEnqueueSVMMap5");
  svmBuffer[index] = value;
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap5");
} JVM_END

JVM_ENTRY(jfloat, SVMBufferSupport_readSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jCommandQueue, jlong jBuffer, jint index, jint length)) {
  float * svmBuffer = (float *)jBuffer;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(float) * (int)length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap6");
  float value = svmBuffer[index];
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap6");
  return value;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_addSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_add", NULL);

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

JVM_ENTRY(void, SVMBufferSupport_fmaSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jlong jBuffer4, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  float* clBuffer4 = (float *)jBuffer4;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_fma", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 3, clBuffer4);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(jfloat, SVMBufferSupport_sumReduce(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jProgram, jlong jCommandQueue, jlong jBuffer, jint length)) {
  cl_context clContext = (cl_context)jContext;
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer = (float *)jBuffer;
  int clLength = (int)length;

  size_t localSize = 256;
  size_t workGroups = std::ceil(clLength / (float)localSize);
  float* sums = (float * )clSVMAlloc(clContext, CL_MEM_READ_WRITE, sizeof(float) * workGroups, 0);

  cl_kernel kernel = clCreateKernel(clProgram, "sumreduce", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, sums);
  handleError(svmError, "clSetKernelArg");

  cl_mem c_mem_obj = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(float), NULL, &svmError);
  handleError(svmError, "createBuffer");

  svmError = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem_obj);
  handleError(svmError, "setArgs");

  svmError = clSetKernelArg(kernel, 3, sizeof(int), &clLength);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 4, sizeof(float) * localSize, NULL);
  handleError(svmError, "clSetKernelArg");

  size_t local_item_size[] = {localSize};
  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, local_item_size, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  float sum = 0;
  clEnqueueReadBuffer(clCommandQueue, c_mem_obj, CL_TRUE, 0, sizeof(float), &sum, 0, NULL, NULL);

  clReleaseKernel(kernel);
  clReleaseMemObject(c_mem_obj);
  clSVMFree(clContext, sums);
  return sum;
} JVM_END

JVM_ENTRY(jlong, SVMBufferSupport_copyFromArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jintArray jArray)) {
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

  int * svmBuffer = (int *) clSVMAlloc((cl_context)clContext, CL_MEM_READ_ONLY, sizeof(int) * length, 0);
  int * jarrayElements = env->GetIntArrayElements(jArray, 0);
  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap4");
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, svmBuffer, jarrayElements, sizeof(int) * length, 0, 0, 0);
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap4");
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
  handleError(svmError, "clEnqueueSVMMap1");
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, svmBuffer, jarrayElements, sizeof(float) * length, 0, 0, 0);
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap1");
  env->ReleaseFloatArrayElements(jArray, jarrayElements, 0);

  return (jlong)svmBuffer;
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jintArray jArray)) {
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

  int* svmBuffer = (int *)jBuffer;

  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, svmBuffer, sizeof(int) * length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap2");
  int * jarrayBody = env->GetIntArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, svmBuffer, sizeof(int) * length, 0, 0, 0);
  svmError = clEnqueueSVMUnmap(clCommandQueue, svmBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap2");
  env->ReleaseIntArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_copyToFloatArray(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer, jfloatArray jArray)) {
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;

  jsize length = env->GetArrayLength(jArray);

  float* clBuffer = (float *)jBuffer;


  svmError = clEnqueueSVMMap(clCommandQueue, CL_TRUE, CL_MAP_READ, clBuffer, sizeof(float) * length, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMMap3");
  float * jarrayBody = env->GetFloatArrayElements(jArray, 0);
  clEnqueueSVMMemcpy(clCommandQueue, CL_TRUE, jarrayBody, clBuffer, sizeof(float) * length, 0, 0, 0);
  svmError = clEnqueueSVMUnmap(clCommandQueue, clBuffer, 0, 0, NULL);
  handleError(svmError, "clEnqueueSVMUnmap3");
  env->ReleaseFloatArrayElements(jArray, jarrayBody, 0);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_releaseSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jCommandQueue, jlong jBuffer)) {
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  int* svmBuffer = (int *)jBuffer;
  svmError = clFinish(clCommandQueue);
  handleError(svmError, "clFinish");
  clSVMFree(clContext, svmBuffer);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_subtract(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float*)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "subtract", &svmError);
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

JVM_ENTRY(void, SVMBufferSupport_subtractionMinuend(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jfloat jMinuend, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float clMinuend = (float)jMinuend;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "subtraction_minuend", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 2, sizeof(float), &clMinuend);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiply(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jfloat jFactor, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float clFactor = (float)jFactor;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "multiply", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 2, sizeof(float), &clFactor);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_multiply", &svmError);
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


JVM_ENTRY(void, SVMBufferSupport_sqrt(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "sqrt", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_division(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jfloat jDivisor, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float clDivisor = (float)jDivisor;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "division", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 2, sizeof(float), &clDivisor);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_divisionBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jBuffer3, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "vector_division", &svmError);
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

JVM_ENTRY(void, SVMBufferSupport_log(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "log", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_exp(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "exp", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_abs(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "abs", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_compareGT(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jfloat jComparee, jlong jBuffer2, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float clComparee = (float)jComparee;
  float* clBuffer2 = (float *)jBuffer2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "compareGT", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 2, sizeof(float), &clComparee);
  handleError(svmError, "clSetKernelArg");


  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_blend(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong jBuffer1, jlong jBuffer2, jlong jMask, jlong jBuffer3, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float* clBuffer1 = (float *)jBuffer1;
  float* clBuffer2 = (float *)jBuffer2;
  float* clMask = (float *)jMask;
  float* clBuffer3 = (float *)jBuffer3;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "blend", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clBuffer1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clBuffer2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 2, clMask);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 3, clBuffer3);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_blackscholes(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jfloat sig, jfloat r, jlong xBuffer, jlong callBuffer, jlong putBuffer, jlong tBuffer, jlong s0, jint length)) {
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

  cl_kernel kernel = clCreateKernel(clProgram, "blackscholes", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArg(kernel, 0, sizeof(float), &clSig);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 1, sizeof(float), &clR);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 2, clXBuffer);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 3, clCallBuffer);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 4, clPutBuffer);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 5, clTBuffer);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 6, clS0Buffer);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_cos(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "cos", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_sin(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "sinx", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_dft(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jlong b3, jlong b4, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  float * clB3 = (float *)b3;
  float * clB4 = (float *)b4;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "dft", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 2, clB3);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 3, clB4);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 4, sizeof(int), &clLength);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_multiplyRepeat(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jint size, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  int clSize = (int)size;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "multiplyRepeat", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 2, sizeof(int), &clSize);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, SVMBufferSupport_forSum(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jCommandQueue, jlong b1, jlong b2, jfloat v1, jint length)) {
  cl_program clProgram = (cl_program)jProgram;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * clB1 = (float *)b1;
  float * clB2 = (float *)b2;
  float clv1 = (float)v1;
  int clLength = (int)length;

  cl_kernel kernel = clCreateKernel(clProgram, "forSum", &svmError);
  handleError(svmError, "clCreateKernel");

  svmError = clSetKernelArgSVMPointer(kernel, 0, clB1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArgSVMPointer(kernel, 1, clB2);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 2, sizeof(float), &clv1);
  handleError(svmError, "clSetKernelArg");

  svmError = clSetKernelArg(kernel, 3, sizeof(int), &clLength);
  handleError(svmError, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  svmError = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(svmError, "clEnqueueNDRangeKernel");

  clReleaseKernel(kernel);
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_executeKernel(JNIEnv *env, jclass vsclazz, jlong jKernel, jlong jCommandQueue, jint length)) {
  int error = 0;
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
  int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_kernel clKernel = clCreateKernel(clProgram, "exec", &error);
  handleError(error, "clCreateKernel");
  return (long)clKernel;
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentSVMBuffer(JNIEnv *env, jclass vsclazz, jlong jKernel, jlong buffer, jint argumentNumber)) {
  cl_kernel clKernel = (cl_kernel)jKernel;
  float* clBuffer = (float *)buffer;
  int clArgumentNumber = (int)argumentNumber;

  svmError = clSetKernelArgSVMPointer(clKernel, clArgumentNumber, clBuffer);
  handleError(svmError, "clSetKernelArg");
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentInteger(JNIEnv *env, jclass vsclazz, jlong jKernel, jint value, jint argumentNumber)) {
  cl_kernel clKernel = (cl_kernel)jKernel;
  int clValue = (int)value;
  int clArgumentNumber = (int)argumentNumber;

  svmError = clSetKernelArg(clKernel, clArgumentNumber, sizeof(int), &clValue);
  handleError(svmError, "clSetKernelArg");
} JVM_END

JVM_ENTRY(void, ExecBufferSupport_setKernelArgumentFloat(JNIEnv *env, jclass vsclazz, jlong jKernel, jfloat value, jint argumentNumber)) {
  cl_kernel clKernel = (cl_kernel)jKernel;
  float clValue = (float)value;
  int clArgumentNumber = (int)argumentNumber;

  svmError = clSetKernelArg(clKernel, clArgumentNumber, sizeof(float), &clValue);
  handleError(svmError, "clSetKernelArg");
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
    {CC "ReadSVMBuffer",   CC "(JJII)F", FN_PTR(SVMBufferSupport_readSVMBuffer)},
    {CC "AddSVMBuffer",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_addSVMBuffer)},
    {CC "CopyToArray",   CC "(JJJ[I)V", FN_PTR(SVMBufferSupport_copyToArray)},
    {CC "CopyToFloatArray",   CC "(JJJ[F)V", FN_PTR(SVMBufferSupport_copyToFloatArray)},
    {CC "CopyFromArray",   CC "(JJ[I)J", FN_PTR(SVMBufferSupport_copyFromArray)},
    {CC "CopyFromArray",   CC "(JJ[F)J", FN_PTR(SVMBufferSupport_copyFromFloatArray)},
    {CC "ReleaseSVMBuffer",   CC "(JJJ)V", FN_PTR(SVMBufferSupport_releaseSVMBuffer)},
    {CC "MatrixFmaSVMBuffer",   CC "(JJJJJIIII)V", FN_PTR(SVMBufferSupport_matrixFmaSVMBuffer)},
    {CC "FmaSVMBuffer",   CC "(JJJJJJI)V", FN_PTR(SVMBufferSupport_fmaSVMBuffer)},
    {CC "SumReduce",   CC "(JJJJI)F", FN_PTR(SVMBufferSupport_sumReduce)},
    {CC "SubtractionMinuend",   CC "(JJJJFI)V", FN_PTR(SVMBufferSupport_subtractionMinuend)},
    {CC "Subtract",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_subtract)},
    {CC "Multiply",   CC "(JJJJFI)V", FN_PTR(SVMBufferSupport_multiply)},
    {CC "Multiply",   CC "(JJJJJI)V", FN_PTR(SVMBufferSupport_multiplyBuffer)},
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
};

JVM_ENTRY(void, JVM_RegisterSVMBufferSupportMethods(JNIEnv* env, jclass vsclass)) {
  ThreadToNativeFromVM ttnfv(thread);

  int ok = env->RegisterNatives(vsclass, jdk_internal_vm_vector_SVMBufferSupport_methods, sizeof(jdk_internal_vm_vector_SVMBufferSupport_methods)/sizeof(JNINativeMethod));
  guarantee(ok == 0, "register jdk.internal.vm.vector.SVMBufferSupport natives");
} JVM_END
