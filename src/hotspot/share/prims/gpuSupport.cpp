#include "prims/gpuSupport.hpp"
#include "precompiled.hpp"
#include "runtime/interfaceSupport.inline.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include "gc/x/xUtils.hpp"
#include "openclHelper.hpp"

JVM_ENTRY(jlong, GPUSupport_createContext(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_int error = 0;
  cl_device_id clDevice = (cl_device_id) jDevice;
  cl_context clContext = clCreateContext(0, 1, &clDevice, NULL, NULL, &error);
  handleError(error, "clCreateContext");

  return (jlong) clContext;
} JVM_END

JVM_ENTRY(void, GPUSupport_releaseContext(JNIEnv *env, jclass vsclazz, jlong jContext)) {
  cl_int error = 0;
  cl_context clContext = (cl_context) jContext;
  if (clContext) {
    error = clReleaseContext(clContext);
    handleError(error, "ReleaseContext");
  }
} JVM_END

JVM_ENTRY(jlong, GPUSupport_createProgram(JNIEnv *env, jclass vsclazz, jlong jContext, jstring jKernelString)) {
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

JVM_ENTRY(void, GPUSupport_releaseProgram(JNIEnv *env, jclass vsclazz, jlong jProgram)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program) jProgram;
  if (clProgram) {
    error = clReleaseProgram(clProgram);
    handleError(error, "ReleaseProgram");
  }
} JVM_END

JVM_ENTRY(jlong, GPUSupport_createCommandQueue(JNIEnv *env, jclass vsclazz, jlong jContext, jlong jDevice)) {
  cl_int error = 0;
  cl_context clContext = (cl_context)jContext;
  cl_device_id clDevice = (cl_device_id)jDevice;
  cl_command_queue_properties properties[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue clCommandQueue = clCreateCommandQueueWithProperties(clContext, clDevice, properties, &error);
  handleError(error, "createCommandQueue");
  return (jlong)clCommandQueue;
} JVM_END

JVM_ENTRY(void, GPUSupport_releaseCommandQueue(JNIEnv *env, jclass vsclazz, jlong jCommandQueue)) {
  cl_int error = 0;
  cl_command_queue clCommandQueue = (cl_command_queue) jCommandQueue;
  if (clCommandQueue) {
    error = clReleaseCommandQueue(clCommandQueue);
    handleError(error, "clReleaseCommandQueue");
  }
} JVM_END

JVM_ENTRY(jlong, GPUSupport_createDevice(JNIEnv *env, jclass vsclazz)) {
  cl_int error = 0;
  cl_platform_id clPlatform = NULL;
  cl_device_id clDevice = NULL;

  error = clGetPlatformIDs(1, &clPlatform, NULL);
  handleError(error, "clGetPlatformIDs");

  error = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 1, &clDevice, NULL);
  handleError(error, "clGetDeviceIDs");

  return (jlong)clDevice;
} JVM_END

JVM_ENTRY(void, GPUSupport_releaseDevice(JNIEnv *env, jclass vsclazz, jlong jDevice)) {
  cl_int error = 0;
  cl_device_id clDevice = (cl_device_id)jDevice;
  if (clDevice) {
    error = clReleaseDevice(clDevice);
    handleError(error, "clReleaseDevice");
  }
} JVM_END

JVM_ENTRY(jfloatArray, GPUSupport_add(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  int clLength = (int)length;

  cl_mem a_mem_obj = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    clLength * sizeof(float), cArray1, &error);
  handleError(error, "createBuffer");
  cl_mem b_mem_obj = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    clLength * sizeof(float), cArray2, &error);
  handleError(error, "createBuffer");
  cl_mem c_mem_obj = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * clLength, NULL, &error);
  handleError(error, "createBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "vector_add", &error);
  handleError(error, "createKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj);
  handleError(error, "kernel");

  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem_obj);
  handleError(error, "setArgs");

  error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem_obj);
  handleError(error, "setArgs");

  size_t global_item_size[] = {(size_t)clLength};

  cl_event event;

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, &event);
  handleError(error, "ndRange");

  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);

  jfloatArray result = env->NewFloatArray(clLength);
  float *ptrResult = (float *)clEnqueueMapBuffer(clCommandQueue, c_mem_obj, CL_TRUE, CL_MAP_READ, 0, sizeof(float) * clLength, 0, NULL, NULL, NULL);
  env->SetFloatArrayRegion(result, 0, clLength, ptrResult);
  clEnqueueUnmapMemObject(clCommandQueue, c_mem_obj, ptrResult, 0, NULL, NULL);

  clReleaseMemObject(a_mem_obj);
  clReleaseMemObject(b_mem_obj);
  clReleaseMemObject(c_mem_obj);
  clReleaseKernel(kernel);
  return result;
} JVM_END

JVM_ENTRY(void, GPUSupport_gpuAdditionHostPtr(JNIEnv *env, jclass vsclazz, jintArray a, jintArray b, jintArray c)) {
  cl_int error = 0;
  jsize length = env->GetArrayLength(a);
  const char * kernel_string[] = {"__kernel void vector_add(__global const int * A, __global const int * B, __global int * C, int N) {int i = get_global_id(0);C[i] = A[i] + B[i];}"};

  const size_t kernel_length = strlen(kernel_string[0]);

  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &error);
  handleError(error, "context");

  int* abuf = env->GetIntArrayElements(a, 0);
  int* bbuf = env->GetIntArrayElements(b, 0);
  int* cbuf = env->GetIntArrayElements(b, 0);

  cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
                                    length * sizeof(int), abuf, &error);
  handleError(error, "createBuffer");

  cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                    length * sizeof(int), bbuf, &error);
  handleError(error, "createBuffer");
  cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                    length * sizeof(int), cbuf, &error);
  handleError(error, "createBuffer");

  cl_command_queue_properties properties[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, properties, NULL);

  cl_program program = clCreateProgramWithSource(context, 1, 
                                                 kernel_string, &kernel_length, NULL);

  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj);
  handleError(error, "kernel");

  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem_obj);
  handleError(error, "setArgs");

  error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem_obj);
  handleError(error, "setArgs");

  error = clSetKernelArg(kernel, 3, sizeof(int), &length);
  handleError(error, "setArgs");


  size_t local_item_size[] = {4};
  size_t global_item_size[] = {(size_t)length}; // Process the entire lists

  cl_event event;

  error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                         global_item_size, local_item_size, 0, NULL, &event);
  handleError(error, "ndRange");
  clWaitForEvents(1, &event);
  cl_ulong start = 0, end = 0;
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  // END-START gives you hints on kind of “pure HW execution time”
  // the resolution of the events is 1e-09 sec
  cl_double g_NDRangePureExecTimeMs = (cl_double)(end - start)*(cl_double)(1e-06);

  // void* mappedBuffer = clEnqueueMapBuffer(command_queue, c_mem_obj, CL_TRUE, CL_MAP_READ, 0, sizeof(int) * length, 0, NULL, NULL, &error);
  // handleError(error, "MapBuffer");

  env->ReleaseIntArrayElements(c, cbuf, 0);
  // error = clEnqueueUnmapMemObject(command_queue, c_mem_obj, mappedBuffer, 0, NULL, NULL);
  // handleError(error, "UnmapBuffer");

  clReleaseMemObject(a_mem_obj);
  clReleaseMemObject(b_mem_obj);
  clReleaseMemObject(c_mem_obj);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  abuf = NULL;
  bbuf = NULL;
  cbuf = NULL;
} JVM_END

JVM_ENTRY(void, GPUSupport_subtract(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloatArray jArray3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  float * cArray3 = env->GetFloatArrayElements(jArray3, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer3 = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray3, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "subtract", NULL);

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
  env->ReleaseFloatArrayElements(jArray3, cArray3, 0);
} JVM_END

// JVM_ENTRY(void, GPUSupport_matrixFmaGPU(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloatArray jArray3, jint K, jint N, jint k, jint length)) {
//   cl_program clProgram = (cl_program)jProgram;
//   cl_context clContext = (cl_context)jContext;
//   cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
//   float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
//   float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
//   float * cArray3 = env->GetFloatArrayElements(jArray3, 0);
//   int clK = (int)K;
//   int clN = (int)N;
//   int clk = (int)k;
//   int clLength = (int)length;

//   cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
//   handleError(error, "clCreateBuffer");
//   cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
//   handleError(error, "clCreateBuffer");
//   cl_mem clBuffer3 = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray3, &error);
//   handleError(error, "clCreateBuffer");

//   cl_kernel kernel = clCreateKernel(clProgram, "matrix_fma", &error);
//   handleError(error, "clCreateKernel");

//   error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
//   handleError(error, "clSetKernelArg");
//   error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
//   handleError(error, "clSetKernelArg");
//   error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clBuffer3);
//   handleError(error, "clSetKernelArg");

//   error = clSetKernelArg(kernel, 3, sizeof(int), &clK);
//   handleError(error, "clSetKernelArg");

//   error = clSetKernelArg(kernel, 4, sizeof(int), &clN);
//   handleError(error, "clSetKernelArg");

//   error = clSetKernelArg(kernel, 5, sizeof(int), &clk);
//   handleError(error, "clSetKernelArg");

//   size_t global_item_size[] = {(size_t)clLength};

//   error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
//                          global_item_size, NULL, 0, NULL, NULL);
//   handleError(error, "clEnqueueNDRangeKernel");

//   clReleaseKernel(kernel);
// } JVM_END

// JVM_ENTRY(void, GPUSupport_fmaGPU(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloatArray jArray3, jfloatArray jArray4, jint length)) {
//   cl_program clProgram = (cl_program)jProgram;
//   cl_context clContext = (cl_context)jContext;
//   cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
//   float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
//   float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
//   float * cArray3 = env->GetFloatArrayElements(jArray3, 0);
//   float * cArray4 = env->GetFloatArrayElements(jArray4, 0);
//   int clK = (int)K;
//   int clN = (int)N;
//   int clk = (int)k;
//   int clLength = (int)length;

//   cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
//   handleError(error, "clCreateBuffer");
//   cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
//   handleError(error, "clCreateBuffer");
//   cl_mem clBuffer3 = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray3, &error);
//   handleError(error, "clCreateBuffer");
//   cl_mem clBuffer4 = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray4, &error);
//   handleError(error, "clCreateBuffer");

//   cl_kernel kernel = clCreateKernel(clProgram, "vector_fma", &error);
//   handleError(error, "clCreateKernel");

//   error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
//   handleError(error, "clSetKernelArg");
//   error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
//   handleError(error, "clSetKernelArg");
//   error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clBuffer3);
//   handleError(error, "clSetKernelArg");
//   error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &clBuffer4);
//   handleError(error, "clSetKernelArg");

//   size_t global_item_size[] = {(size_t)clLength};

//   error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
//                          global_item_size, NULL, 0, NULL, NULL);
//   handleError(error, "clEnqueueNDRangeKernel");

//   clReleaseKernel(kernel);
// } JVM_END

JVM_ENTRY(jfloat, GPUSupport_reduceAdd(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");

  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(float), NULL, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "vector_reduce", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");


  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "setArgs");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  float sum = 0;
  clEnqueueReadBuffer(clCommandQueue, clBuffer2, CL_TRUE, 0, sizeof(float), &sum, 0, NULL, NULL);

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  return sum;
} JVM_END


JVM_ENTRY(void, GPUSupport_subtractionMinuend(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloat jMinuend, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  float clMinuend = (float)jMinuend;
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "subtraction_minuend", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 2, sizeof(float), &clMinuend);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_multiply(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloat jFactor, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  float clFactor = (float)jFactor;
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "multiply", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 2, sizeof(float), &clFactor);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_multiplyBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloatArray jArray3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  float * cArray3 = env->GetFloatArrayElements(jArray3, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer3 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray3, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "vector_multiply", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArgSVMPointer(kernel, 2, clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
  env->ReleaseFloatArrayElements(jArray3, cArray3, 0);
} JVM_END


JVM_ENTRY(void, GPUSupport_sqrt(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "sqrt", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_division(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloat jDivisor, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  float clDivisor = (float)jDivisor;
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "division", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 2, sizeof(float), &clDivisor);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_divisionBuffer(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloatArray jArray3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  float * cArray3 = env->GetFloatArrayElements(jArray3, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer3 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray3, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "vector_division", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
  env->ReleaseFloatArrayElements(jArray3, cArray3, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_log(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "log", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_exp(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "exp", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_abs(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "abs", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_compareGT(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloat jComparee, jfloatArray jArray2, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  float clComparee = (float)jComparee;
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "compareGT", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 2, sizeof(float), &clComparee);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
} JVM_END

JVM_ENTRY(void, GPUSupport_blend(JNIEnv *env, jclass vsclazz, jlong jProgram, jlong jContext, jlong jCommandQueue, jfloatArray jArray1, jfloatArray jArray2, jfloatArray jMask, jfloatArray jArray3, jint length)) {
  cl_int error = 0;
  cl_program clProgram = (cl_program)jProgram;
  cl_context clContext = (cl_context)jContext;
  cl_command_queue clCommandQueue = (cl_command_queue)jCommandQueue;
  float * cArray1 = env->GetFloatArrayElements(jArray1, 0);
  float * cArray2 = env->GetFloatArrayElements(jArray2, 0);
  float * cMask = env->GetFloatArrayElements(jMask, 0);
  float * cArray3 = env->GetFloatArrayElements(jArray3, 0);
  int clLength = (int)length;

  cl_mem clBuffer1 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray1, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer2 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray2, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clMaskBuffer = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cMask, &error);
  handleError(error, "clCreateBuffer");
  cl_mem clBuffer3 = clCreateBuffer(clContext, CL_MEM_USE_HOST_PTR, sizeof(float) * clLength, cArray3, &error);
  handleError(error, "clCreateBuffer");

  cl_kernel kernel = clCreateKernel(clProgram, "blend", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer1);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clBuffer2);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clMaskBuffer);
  handleError(error, "clSetKernelArg");
  error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &clBuffer3);
  handleError(error, "clSetKernelArg");

  size_t global_item_size[] = {(size_t)clLength};

  error = clEnqueueNDRangeKernel(clCommandQueue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, NULL);
  handleError(error, "clEnqueueNDRangeKernel");

  error = clFinish(clCommandQueue);
  handleError(error, "clFinish");
  clReleaseKernel(kernel);
  env->ReleaseFloatArrayElements(jArray1, cArray1, 0);
  env->ReleaseFloatArrayElements(jArray2, cArray2, 0);
  env->ReleaseFloatArrayElements(jMask, cMask, 0);
  env->ReleaseFloatArrayElements(jArray3, cArray3, 0);
} JVM_END

#define CC (char*)  /*cast a literal from (const char*)*/
#define FN_PTR(f) CAST_FROM_FN_PTR(void*, &f)

static JNINativeMethod jdk_internal_vm_vector_GPUSupport_methods[] = {
    {CC "CreateContext",   CC "(J)J", FN_PTR(GPUSupport_createContext)},
    {CC "ReleaseContext",   CC "(J)V", FN_PTR(GPUSupport_releaseContext)},
    {CC "CreateProgram",   CC "(JLjava/lang/String;)J", FN_PTR(GPUSupport_createProgram)},
    {CC "ReleaseProgram",   CC "(J)V", FN_PTR(GPUSupport_releaseProgram)},
    {CC "CreateCommandQueue",   CC "(JJ)J", FN_PTR(GPUSupport_createCommandQueue)},
    {CC "ReleaseCommandQueue",   CC "(J)V", FN_PTR(GPUSupport_releaseCommandQueue)},
    {CC "CreateDevice",   CC "()J", FN_PTR(GPUSupport_createDevice)},
    {CC "ReleaseDevice",   CC "(J)V", FN_PTR(GPUSupport_releaseDevice)},
    {CC "Add",   CC "(JJJ[F[FI)[F", FN_PTR(GPUSupport_add)},
    // {CC "MatrixFma",   CC "(JJJ[F[F[FIIII)V", FN_PTR(GPUSupport_matrixFmaGPU)},
    // {CC "Fma",   CC "(JJJ[F[F[F[FI)V", FN_PTR(GPUSupport_fmaGPU)},
    {CC "ReduceAdd",   CC "(JJJ[FI)F", FN_PTR(GPUSupport_reduceAdd)},
    {CC "SubtractionMinuend",   CC "(JJJ[F[FFI)V", FN_PTR(GPUSupport_subtractionMinuend)},
    {CC "Subtract",   CC "(JJJ[F[F[FI)V", FN_PTR(GPUSupport_subtract)},
    {CC "Multiply",   CC "(JJJ[F[FFI)V", FN_PTR(GPUSupport_multiply)},
    {CC "Multiply",   CC "(JJJ[F[F[FI)V", FN_PTR(GPUSupport_multiplyBuffer)},
    {CC "Sqrt",   CC "(JJJ[F[FI)V", FN_PTR(GPUSupport_sqrt)},
    {CC "Division",   CC "(JJJ[F[FFI)V", FN_PTR(GPUSupport_division)},
    {CC "Division",   CC "(JJJ[F[F[FI)V", FN_PTR(GPUSupport_divisionBuffer)},
    {CC "Log",   CC "(JJJ[F[FI)V", FN_PTR(GPUSupport_log)},
    {CC "Exp",   CC "(JJJ[F[FI)V", FN_PTR(GPUSupport_exp)},
    {CC "Abs",   CC "(JJJ[F[FI)V", FN_PTR(GPUSupport_abs)},
    {CC "CompareGT",   CC "(JJJ[FF[FI)V", FN_PTR(GPUSupport_compareGT)},
    {CC "Blend",   CC "(JJJ[F[F[F[FI)V", FN_PTR(GPUSupport_blend)},
    {CC "gpuAddHostPtr",   CC "([I[I[I)V", FN_PTR(GPUSupport_gpuAdditionHostPtr)},

};

JVM_ENTRY(void, JVM_RegisterGPUSupportMethods(JNIEnv* env, jclass vsclass)) {
  ThreadToNativeFromVM ttnfv(thread);

  int ok = env->RegisterNatives(vsclass, jdk_internal_vm_vector_GPUSupport_methods, sizeof(jdk_internal_vm_vector_GPUSupport_methods)/sizeof(JNINativeMethod));
  guarantee(ok == 0, "register jdk.internal.vm.vector.GPUSupport natives");
} JVM_END
