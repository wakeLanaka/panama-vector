#include "prims/gpuSupport.hpp"
#include "runtime/interfaceSupport.inline.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"
#include <iostream>

cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_device_id device = NULL;
cl_kernel kernel = NULL;
cl_program program = NULL;

int mem_object_counter = 0;

cl_mem first_mem;
cl_mem second_mem;
cl_mem result_mem;

int * result;

const char *getErrorString(cl_int error) {
  switch(error){
      // run-time and JIT compiler errors
      case 0: return "CL_SUCCESS";
      case -1: return "CL_DEVICE_NOT_FOUND";
      case -2: return "CL_DEVICE_NOT_AVAILABLE";
      case -3: return "CL_COMPILER_NOT_AVAILABLE";
      case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case -5: return "CL_OUT_OF_RESOURCES";
      case -6: return "CL_OUT_OF_HOST_MEMORY";
      case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
      case -8: return "CL_MEM_COPY_OVERLAP";
      case -9: return "CL_IMAGE_FORMAT_MISMATCH";
      case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case -11: return "CL_BUILD_PROGRAM_FAILURE";
      case -12: return "CL_MAP_FAILURE";
      case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
      case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
      case -15: return "CL_COMPILE_PROGRAM_FAILURE";
      case -16: return "CL_LINKER_NOT_AVAILABLE";
      case -17: return "CL_LINK_PROGRAM_FAILURE";
      case -18: return "CL_DEVICE_PARTITION_FAILED";
      case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

      // compile-time errors
      case -30: return "CL_INVALID_VALUE";
      case -31: return "CL_INVALID_DEVICE_TYPE";
      case -32: return "CL_INVALID_PLATFORM";
      case -33: return "CL_INVALID_DEVICE";
      case -34: return "CL_INVALID_CONTEXT";
      case -35: return "CL_INVALID_QUEUE_PROPERTIES";
      case -36: return "CL_INVALID_COMMAND_QUEUE";
      case -37: return "CL_INVALID_HOST_PTR";
      case -38: return "CL_INVALID_MEM_OBJECT";
      case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case -40: return "CL_INVALID_IMAGE_SIZE";
      case -41: return "CL_INVALID_SAMPLER";
      case -42: return "CL_INVALID_BINARY";
      case -43: return "CL_INVALID_BUILD_OPTIONS";
      case -44: return "CL_INVALID_PROGRAM";
      case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
      case -46: return "CL_INVALID_KERNEL_NAME";
      case -47: return "CL_INVALID_KERNEL_DEFINITION";
      case -48: return "CL_INVALID_KERNEL";
      case -49: return "CL_INVALID_ARG_INDEX";
      case -50: return "CL_INVALID_ARG_VALUE";
      case -51: return "CL_INVALID_ARG_SIZE";
      case -52: return "CL_INVALID_KERNEL_ARGS";
      case -53: return "CL_INVALID_WORK_DIMENSION";
      case -54: return "CL_INVALID_WORK_GROUP_SIZE";
      case -55: return "CL_INVALID_WORK_ITEM_SIZE";
      case -56: return "CL_INVALID_GLOBAL_OFFSET";
      case -57: return "CL_INVALID_EVENT_WAIT_LIST";
      case -58: return "CL_INVALID_EVENT";
      case -59: return "CL_INVALID_OPERATION";
      case -60: return "CL_INVALID_GL_OBJECT";
      case -61: return "CL_INVALID_BUFFER_SIZE";
      case -62: return "CL_INVALID_MIP_LEVEL";
      case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
      case -64: return "CL_INVALID_PROPERTY";
      case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
      case -66: return "CL_INVALID_COMPILER_OPTIONS";
      case -67: return "CL_INVALID_LINKER_OPTIONS";
      case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

      // extension errors
      case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
      case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
      case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
      case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
      case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
      case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
      default: return "Unknown OpenCL error";
      }
}

void handleError(cl_int value, const char * errorString) {
  if (value != CL_SUCCESS) {
    std::cerr << "Error[" << errorString << "]: " << getErrorString(value) << std::endl;
  }
}

unsigned int verifyZeroCopyPtr(void *ptr, unsigned int sizeOfContentsOfPtr) {
  int status;
  if((uintptr_t)ptr % 4096 == 0) {
    if(sizeOfContentsOfPtr % 64 == 0) {
      status = 1;
    }
    else status = 0;
  }
  else status = 0;
  return status;
}

cl_int error = 0;

JVM_ENTRY(jintArray, GPUSupport_intoArray(JNIEnv *env, jclass vsclazz, jint first, jint length, jintArray output)) {
  void* mappedBuffer = clEnqueueMapBuffer(command_queue, result_mem, CL_TRUE, CL_MAP_READ, 0, sizeof(int) * length, 0, NULL, NULL, &error);
  handleError(error, "EnqueueMapBuffer");

  error = clEnqueueUnmapMemObject(command_queue, result_mem, mappedBuffer, 0, NULL, NULL);
  handleError(error, "UnmapMemObject");

  env->SetIntArrayRegion(output, 0, length, result);

  clReleaseMemObject(first_mem);
  clReleaseMemObject(second_mem);
  clReleaseMemObject(result_mem);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  mem_object_counter = 0;

  return output;
} JVM_END

JVM_ENTRY(jint, GPUSupport_add(JNIEnv *env, jclass vsclazz, jint first, jint second, jint lowerBound, jint upperBound, jint maxGroupSize)) {
  const char * kernel_string[] = {"__kernel void vector_add(__constant const int *A, __constant const int *B, __global int *C, int upperBound) {int i = get_global_id(0);if(i < upperBound){C[i] = A[i] + B[i];}}"};
  const size_t kernelLength = strlen(kernel_string[0]);


  program = clCreateProgramWithSource(context, 1, kernel_string, &kernelLength, &error);
  handleError(error, "program");

  error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  handleError(error, "buildProgram");

  kernel = clCreateKernel(program, "vector_add", &error);
  handleError(error, "createKernel");

  int mem_object_number = mem_object_counter;
  mem_object_counter++;
  result = (int*)ZUtils::alloc_aligned(4096, sizeof(int) * upperBound);

  result_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
          upperBound * sizeof(int), result, &error);
  handleError(error, "createBuffer");

  error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &first_mem);
  handleError(error, "setKernel 0");
  error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &second_mem);
  handleError(error, "setKernel 1");
  error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_mem);
  handleError(error, "setKernel 2");
  error = clSetKernelArg(kernel, 3, sizeof(int), &upperBound);
  handleError(error, "setKernel 3");

  size_t localWorkSize[] = { (size_t)maxGroupSize };
  size_t globalWorkSize[] = {(upperBound - lowerBound + localWorkSize[0] - 1) / localWorkSize[0] * localWorkSize[0]};
  error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
          globalWorkSize, localWorkSize, 0, NULL, NULL);
  handleError(error, "enqueueNDRangeKernel");
  return mem_object_number;
} JVM_END

JVM_ENTRY(jint, GPUSupport_allocateArray(JNIEnv *env, jclass vsclazz, jintArray j_array)) {
  int mem_object_number = mem_object_counter;
  mem_object_counter++;
  int * abuf = env->GetIntArrayElements(j_array, 0);
  jsize length = env->GetArrayLength(j_array);

  // int * abuf = (int*)ZUtils::alloc_aligned(4096, sizeof(int) * length);
  // memcpy(abuf, env->GetIntArrayElements(j_array, 0), sizeof(int) * length);
  // std::cout << "verify: " << verifyZeroCopyPtr(abuf, sizeof(int) * length) << std::endl;

  if (mem_object_number == 0) {
    first_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, length * sizeof(int), abuf, &error);
  } else {
    second_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, length * sizeof(int), abuf, &error);
  }
  handleError(error, "clCreateBuffer");
  return mem_object_number;
} JVM_END

JVM_ENTRY(jint, GPUSupport_initializeGPU(JNIEnv *env, jclass vsclazz)) {
  cl_platform_id platform;
  size_t maxWorkGroupSize;

  error = clGetPlatformIDs(1, &platform, NULL);
  handleError(error, "clGetPlatformIDs");

  error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  handleError(error, "clGetDeviceIDs");

  error = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                sizeof(size_t), &maxWorkGroupSize, NULL);
  handleError(error, "clGetDeviceInfo");

  context = clCreateContext(0, 1, &device, NULL, NULL, &error);
  handleError(error, "clCreateContext");

  command_queue = clCreateCommandQueueWithProperties(context, device, 0, &error);
  handleError(error, "clCreateCommandQueueWithProperties");

  return maxWorkGroupSize;
} JVM_END

JVM_ENTRY(void, GPUSupport_gpuAdditionHostPtr(JNIEnv *env, jclass vsclazz, jintArray a, jintArray b, jintArray c)) {
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
  std::cout << "kernel: " << g_NDRangePureExecTimeMs << std::endl;

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


JVM_ENTRY(void, GPUSupport_gpuAddition(JNIEnv *env, jclass vsclazz, jintArray a, jintArray b, jintArray c)) {
  jsize length = env->GetArrayLength(a);
  const char * kernel_string[] = {"__kernel void vector_add(__global const int * A, __global const int * B, __global int * C, int N) {int i = get_global_id(0);C[i] = A[i] + B[i];}"};

  const size_t kernel_length = strlen(kernel_string[0]);

  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &error);
  handleError(error, "clCreateContext");
  int* abuf = (int *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(int) * length, 0);
  int* bbuf = (int *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(int) * length, 0);
  int* cbuf = (int *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(int) * length, 0);

  cl_event map_start;
  cl_event map_end;

  // // Create a command queue
  cl_command_queue_properties properties[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, properties, NULL);

  clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_WRITE, abuf, sizeof(int) * length, 0, 0, &map_start);
  clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_WRITE, bbuf, sizeof(int) * length, 0, 0, 0);
  clEnqueueSVMMemcpy(command_queue, CL_TRUE, abuf, env->GetIntArrayElements(a,0), sizeof(int) * length, 0, 0, 0);
  clEnqueueSVMMemcpy(command_queue, CL_TRUE, bbuf, env->GetIntArrayElements(b,0), sizeof(int) * length, 0, 0, 0);
  // memcpy(abuf, env->GetIntArrayElements(a, 0), sizeof(int) * length);
  // memcpy(bbuf, env->GetIntArrayElements(b, 0), sizeof(int) * length);
  clEnqueueSVMUnmap(command_queue, abuf, 0, 0, 0);
  clEnqueueSVMUnmap(command_queue, bbuf, 0, 0, &map_end);
  clWaitForEvents(1, &map_start);
  clWaitForEvents(1, &map_end);
  cl_ulong start = 0, end = 0;
  clGetEventProfilingInfo(map_start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  clGetEventProfilingInfo(map_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  // END-START gives you hints on kind of “pure HW execution time”
  // the resolution of the events is 1e-09 sec
  cl_double memcopyDuration = (cl_double)(end - start)*(cl_double)(1e-06);

  // // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, 
                                                 kernel_string, &kernel_length, NULL);

  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

  error = clSetKernelArgSVMPointer(kernel, 0, abuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, bbuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, cbuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 3, sizeof(int), &length);
  handleError(error, "clSetKernelArg");


  size_t local_item_size[] = {4};
  size_t global_item_size[] = {(size_t)length}; // Process the entire lists

  cl_event event;
  error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, &event);
  handleError(error, "clEnqueueNDRangeKernel");

  clWaitForEvents(1, &event);
  // cl_ulong start = 0, end = 0;
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  // END-START gives you hints on kind of “pure HW execution time”
  // the resolution of the events is 1e-09 sec
  cl_double g_NDRangePureExecTimeMs = (cl_double)(end - start)*(cl_double)(1e-06);

  clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_READ, cbuf, sizeof(int) * length, 0, 0, &map_start);
  int * javaArray = env->GetIntArrayElements(c,0);
  clEnqueueSVMMemcpy(command_queue, CL_TRUE, javaArray, cbuf, sizeof(int) * length, 0, 0, 0);
  env->ReleaseIntArrayElements(c, javaArray, 0);
  // env->SetIntArrayRegion(c, 0, length, cbuf);
  clEnqueueSVMUnmap(command_queue, cbuf, 0, 0, &map_end);
  handleError(error, "clEnqueueUnmapMemObject");
  clWaitForEvents(1, &map_start);
  clWaitForEvents(1, &map_end);
  clGetEventProfilingInfo(map_start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  clGetEventProfilingInfo(map_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  // END-START gives you hints on kind of “pure HW execution time”
  // the resolution of the events is 1e-09 sec
  memcopyDuration = memcopyDuration + (cl_double)(end - start)*(cl_double)(1e-06);
  std::cout << "kernel: " << g_NDRangePureExecTimeMs << " ms" << std::endl;
  std::cout << "memcopy: " << memcopyDuration << " ms" << std::endl;


  clSVMFree(context, abuf);
  clSVMFree(context, bbuf);
  clSVMFree(context, cbuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
} JVM_END


JVM_ENTRY(void, GPUSupport_gpuMatrix(JNIEnv *env, jclass vsclazz, jfloatArray a, jfloatArray b, jfloatArray c, jint n)) {
  jsize length = env->GetArrayLength(a);
  const char * kernel_string[] = {"__kernel void vector_matrix(__constant const float * A, __constant const float * B, __global float * C, int n) {int i = get_global_id(0); int j = get_global_id(1);if(i<n && j<n){float sum = 0; for(int k = 0; k < n; k++){sum += A[i * n + k] * B[k * n + j];}C[i*n + j] = sum;}}"};

  const size_t kernel_length = strlen(kernel_string[0]);
  cl_event event;
  cl_ulong start = 0, end = 0;

  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &error);
  handleError(error, "clCreateContext");
  float* abuf = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float) * length, 0);
  float* bbuf = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, sizeof(float) * length, 0);
  float* cbuf = (float *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, sizeof(float) * length, 0);


  // // Create a command queue

  cl_command_queue_properties properties[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, properties, &error);
  handleError(error, "clCreateCommandQueueWithProperties");

  cl_event map_start;
  cl_event map_end;
  error = clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_WRITE, abuf, sizeof(float) * length, 0, 0, &map_start);
  handleError(error, "clEnqueueSVMMap");

  clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_WRITE, bbuf, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");

  memcpy(abuf, env->GetFloatArrayElements(a, 0), sizeof(float) * length);
  memcpy(bbuf, env->GetFloatArrayElements(b, 0), sizeof(float) * length);

  error = clEnqueueSVMUnmap(command_queue, abuf, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");
  error = clEnqueueSVMUnmap(command_queue, bbuf, 0, 0, &map_end);
  handleError(error, "clEnqueueSVMUnmap");
  // clWaitForEvents(1, &map_start);
  // clWaitForEvents(1, &map_end);
  // clGetEventProfilingInfo(map_start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  // clGetEventProfilingInfo(map_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  cl_double mapExec1 = (cl_double)(end - start)*(cl_double)(1e-06);

  // // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, kernel_string, &kernel_length, &error);
  handleError(error, "clCreateProgramWithSource");

  error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  handleError(error, "clBuildProgram");

  cl_kernel kernel = clCreateKernel(program, "vector_matrix", &error);
  handleError(error, "clCreateKernel");

  error = clSetKernelArgSVMPointer(kernel, 0, abuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, bbuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, cbuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 3, sizeof(int), &n);
  handleError(error, "clSetKernelArg");


  // size_t local_item_size[] = {4, 4};
  size_t global_item_size[] = {(size_t)n, (size_t)n}; // Process the entire lists

  error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
                         global_item_size, NULL, 0, NULL, &event);
  handleError(error, "clEnqueueNDRangeKernel");
  // clWaitForEvents(1, &event);
  // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  // clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  //END-START gives you hints on kind of “pure HW execution time”
  //the resolution of the events is 1e-09 sec
  // cl_double g_NDRangePureExecTimeMs = (cl_double)(end - start)*(cl_double)(1e-06);

  clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_READ, cbuf, sizeof(int) * length, 0, 0, &map_start);
  env->SetFloatArrayRegion(c, 0, length, cbuf);
  clEnqueueSVMUnmap(command_queue, abuf, 0, 0, &map_end);
  handleError(error, "clEnqueueUnmapMemObject");
  // clWaitForEvents(1, &map_start);
  // clWaitForEvents(1, &map_end);
  // clGetEventProfilingInfo(map_start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  // clGetEventProfilingInfo(map_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  // cl_double mapExec2 = (cl_double)(end - start)*(cl_double)(1e-06);


  // std::cout << "ndRangeKernel: " << g_NDRangePureExecTimeMs << " ms" << std::endl;
  // std::cout << "Map1: " << mapExec1 << " ms" << std::endl;
  // std::cout << "Map2: " << mapExec2 << " ms" << std::endl;
  clSVMFree(context, abuf);
  clSVMFree(context, bbuf);
  clSVMFree(context, cbuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
} JVM_END

//TODO
cl_context jmhContext = NULL;
cl_command_queue jmhCommandQueue = NULL;
int * svmABuf = NULL;
int * svmBBuf = NULL;
int * svmCBuf = NULL;

JVM_ENTRY(void, GPUSupport_initSVM(JNIEnv *env, jclass vsclazz, jint n)) {
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  jmhContext = clCreateContext(0, 1, &device, NULL, NULL, &error);
  handleError(error, "clCreateContext");

  svmABuf = (int *) clSVMAlloc(jmhContext, CL_MEM_READ_ONLY, sizeof(int) * n, 0);

  svmBBuf = (int *) clSVMAlloc(jmhContext, CL_MEM_READ_ONLY, sizeof(int) * n, 0);
  svmCBuf = (int *) clSVMAlloc(jmhContext, CL_MEM_WRITE_ONLY, sizeof(int) * n, 0);

  svmCommandQueue = clCreateCommandQueueWithProperties(context, device, properties, &error);
  handleError(error, "clCreateCommandQueueWithProperties");

  error = clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_WRITE, abuf, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");

  clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_WRITE, bbuf, sizeof(float) * length, 0, 0, NULL);
  handleError(error, "clEnqueueSVMMap");

  for(int i = 0; i < n; i++) {
    svmABuf[i] = rand();
    svmBBuf[i] = rand();

  }
  error = clEnqueueSVMUnmap(command_queue, abuf, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");
  error = clEnqueueSVMUnmap(command_queue, bbuf, 0, 0, NULL);
  handleError(error, "clEnqueueSVMUnmap");

} JVM_END
JVM_ENTRY(void, GPUSupport_svmAdd(JNIEnv *env, jclass vsclazz)) {
  const char * kernel_string[] = {"__kernel void vector_add(__global const int * A, __global const int * B, __global int * C, int N) {int i = get_global_id(0);C[i] = A[i] + B[i];}"};

  const size_t kernel_length = strlen(kernel_string[0]);
  cl_program program = clCreateProgramWithSource(context, 1, 
                                                 kernel_string, &kernel_length, NULL);

  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

  error = clSetKernelArgSVMPointer(kernel, 0, abuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 1, bbuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArgSVMPointer(kernel, 2, cbuf);
  handleError(error, "clSetKernelArg");

  error = clSetKernelArg(kernel, 3, sizeof(int), &length);
  handleError(error, "clSetKernelArg");


  size_t local_item_size[] = {4};
  size_t global_item_size[] = {(size_t)length}; // Process the entire lists

  cl_event event;
  error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                         global_item_size, NULL, 0, NULL, &event);
  handleError(error, "clEnqueueNDRangeKernel");

  clWaitForEvents(1, &event);
  // cl_ulong start = 0, end = 0;
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  // END-START gives you hints on kind of “pure HW execution time”
  // the resolution of the events is 1e-09 sec
  cl_double g_NDRangePureExecTimeMs = (cl_double)(end - start)*(cl_double)(1e-06);

  clEnqueueSVMMap(command_queue, CL_TRUE, CL_MAP_READ, cbuf, sizeof(int) * length, 0, 0, &map_start);
  int * javaArray = env->GetIntArrayElements(c,0);
  clEnqueueSVMMemcpy(command_queue, CL_TRUE, javaArray, cbuf, sizeof(int) * length, 0, 0, 0);
  env->ReleaseIntArrayElements(c, javaArray, 0);
  // env->SetIntArrayRegion(c, 0, length, cbuf);
  clEnqueueSVMUnmap(command_queue, cbuf, 0, 0, &map_end);
  handleError(error, "clEnqueueUnmapMemObject");
  clWaitForEvents(1, &map_start);
  clWaitForEvents(1, &map_end);
  clGetEventProfilingInfo(map_start, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  clGetEventProfilingInfo(map_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  // END-START gives you hints on kind of “pure HW execution time”
  // the resolution of the events is 1e-09 sec
  memcopyDuration = memcopyDuration + (cl_double)(end - start)*(cl_double)(1e-06);
  std::cout << "kernel: " << g_NDRangePureExecTimeMs << " ms" << std::endl;
  std::cout << "memcopy: " << memcopyDuration << " ms" << std::endl;


  clSVMFree(context, abuf);
  clSVMFree(context, bbuf);
  clSVMFree(context, cbuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
} JVM_END


#define CC (char*)  /*cast a literal from (const char*)*/
#define FN_PTR(f) CAST_FROM_FN_PTR(void*, &f)

static JNINativeMethod jdk_internal_vm_vector_GPUSupport_methods[] = {
    {CC "initializeGPU",   CC "()I", FN_PTR(GPUSupport_initializeGPU)},
    {CC "allocateArray",   CC "([I)I", FN_PTR(GPUSupport_allocateArray)},
    {CC "add",   CC "(IIIII)I", FN_PTR(GPUSupport_add)},
    {CC "intoArray",   CC "(II[I)[I", FN_PTR(GPUSupport_intoArray)},
    {CC "gpuAddition",   CC "([I[I[I)V", FN_PTR(GPUSupport_gpuAddition)},
    {CC "gpuMatrix",   CC "([F[F[FI)V", FN_PTR(GPUSupport_gpuMatrix)},
    {CC "gpuAdditionHostPtr",   CC "([I[I[I)V", FN_PTR(GPUSupport_gpuAdditionHostPtr)}

};

JVM_ENTRY(void, JVM_RegisterGPUSupportMethods(JNIEnv* env, jclass vsclass)) {
  ThreadToNativeFromVM ttnfv(thread);

  int ok = env->RegisterNatives(vsclass, jdk_internal_vm_vector_GPUSupport_methods, sizeof(jdk_internal_vm_vector_GPUSupport_methods)/sizeof(JNINativeMethod));
  guarantee(ok == 0, "register jdk.internal.vm.vector.GPUSupport natives");
} JVM_END
