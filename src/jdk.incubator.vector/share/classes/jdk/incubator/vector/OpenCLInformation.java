package jdk.incubator.vector;

import jdk.incubator.vector.GPUInformation;
import jdk.internal.vm.vector.SVMBufferSupport;


/**
 *  OpenCL informations
 */
public class OpenCLInformation implements GPUInformation {
    private long context;
    private long device;
    private long commandQueue;
    private long program;
    private int maxWorkGroupSize;

    /**
     *  Creates an OpenCLInformation
     */
    public OpenCLInformation() {
        device = SVMBufferSupport.CreateDevice();
        context = SVMBufferSupport.CreateContext(device);
        program = SVMBufferSupport.CreateProgram(context, getKernels());
        commandQueue = SVMBufferSupport.CreateCommandQueue(context, device);
        maxWorkGroupSize = SVMBufferSupport.GetMaxWorkGroupSize(device);
    }

    /**
     *  Deallocates the context
     */
    public void ReleaseContext() {
        SVMBufferSupport.ReleaseContext(this.context);
    }

    /**
     *  Deallocates the command queue
     */
    public void ReleaseCommandQueue() {
        SVMBufferSupport.ReleaseCommandQueue(this.commandQueue);
    }

    /**
     *  Deallocates the device
     */
    public void ReleaseDevice() {
        SVMBufferSupport.ReleaseDevice(this.device);
    }

    /**
     *  Get the current opencl context
     *  @return the current opencl context
     */
    public long GetContext() {
        return context;
    }

    /**
     *  Get the current opencl command queue
     *  @return the current opencl command queue
     */
    public long GetCommandQueue() {
        return commandQueue;
    }

    /**
     *  Get the current opencl program
     *  @return the current opencl program
     */
    public long GetProgram() {
        return program;
    }

    /**
     *  Get the maximal work group size
     *  @return the maximal work group size for this device
     */
    public int GetMaxWorkGroupSize() {
        return maxWorkGroupSize;
    }

    private String getFMAKernels() {
        return "__kernel void matrix_fma(__global const float * A, __global const float * B, __global float * C, int m, int n, int k) { int i = get_global_id(0); int a = (i / n) * m + k; int b = i % n + k * n; C[i] = fma(A[a], B[b], C[i]);} __kernel void vector_fma(__global const float * A, __global const float * B, __global float * C, __global float * D){int i = get_global_id(0); D[i] = fma(A[i], B[i], C[i]);}";
    }

    private String getAddKernels() {
        return "__kernel void addFLOATFLOATFLOAT(__global const float * A, const float value, __global float * C){int i = get_global_id(0); C[i] = A[i] + value;}__kernel void addINTFLOATFLOAT(__global const int * A, const float value, __global float * C){int i = get_global_id(0); C[i] = A[i] + value;}__kernel void addFLOATINTFLOAT(__global const float * A, const int value, __global float * C){int i = get_global_id(0); C[i] = A[i] + value;}__kernel void addINTINTINT(__global const int * A, const int value, __global int * C){int i = get_global_id(0); C[i] = A[i] + value;}__kernel void addBufferFLOATFLOATFLOAT(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] + B[i];}__kernel void addBufferINTFLOATFLOAT(__global const int * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] + B[i];}__kernel void addBufferFLOATINTFLOAT(__global const float * A, __global const int * B, __global float * C){int i = get_global_id(0); C[i] = A[i] + B[i];}__kernel void addBufferINTINTINT(__global const int * A, __global const int * B, __global int * C){int i = get_global_id(0); C[i] = A[i] + B[i];} __kernel void sumreduceFLOAT(__global const float * A, __global float * B, __local float * localData){const int gid = get_global_id(0); const int lid = get_local_id(0); const int localSize = get_local_size(0); localData[lid] = work_group_reduce_add(A[gid]); if(lid == 0){B[get_group_id(0)] = localData[0];}} __kernel void sumreduceINT(__global const int * A, __global int * B, __local int * localData){const int gid = get_global_id(0); const int lid = get_local_id(0); const int localSize = get_local_size(0); localData[lid] = work_group_reduce_add(A[gid]); if(lid == 0){B[get_group_id(0)] = localData[0];}}";
    }

    private String getMulKernels(){
        return "__kernel void mulFLOATFLOATFLOAT(__global const float * A, const float value, __global float * C){int i = get_global_id(0); C[i] = A[i] * value;} __kernel void mulINTFLOATFLOAT(__global const int * A, const float value, __global float * C){int i = get_global_id(0); C[i] = A[i] * value;} __kernel void mulFLOATINTFLOAT(__global const float * A, const int value, __global float * C){int i = get_global_id(0); C[i] = A[i] * value;} __kernel void mulINTINTINT(__global const int * A, const int value, __global int * C){int i = get_global_id(0); C[i] = A[i] * value;} __kernel void mulBufferFLOATFLOATFLOAT(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] * B[i];} __kernel void mulBufferINTFLOATFLOAT(__global const int * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] * B[i];} __kernel void mulBufferFLOATINTFLOAT(__global const float * A, __global const int * B, __global float * C){int i = get_global_id(0); C[i] = A[i] * B[i];} __kernel void mulBufferINTINTINT(__global const int * A, __global const int * B, __global int * C){int i = get_global_id(0); C[i] = A[i] * B[i];} __kernel void multiplyrange(__global const float * A, __global const float * B, __global float * C, int index1, int index2){int i = get_global_id(0); C[i] = A[i + index1] * B[i + index2];} __kernel void multiplyRepeatBufferFLOATFLOATFLOAT(__global const float * A, __global const float * B, __global float * C, int size) {int i = get_global_id(0); C[i] = A[i] * B[i % size];} __kernel void multiplyRepeatBufferFLOATINTFLOAT(__global const float * A, __global const int * B, __global float * C, int size) {int i = get_global_id(0); C[i] = A[i] * B[i % size];}__kernel void multiplyRepeatBufferINTFLOATFLOAT(__global const int * A, __global const float * B, __global float * C, int size) {int i = get_global_id(0); C[i] = A[i] * B[i % size];}__kernel void multiplyRepeatBufferINTINTINT(__global const int * A, __global const int * B, __global int * C, int size) {int i = get_global_id(0); C[i] = A[i] * B[i % size];}";
    }

    private String getSubKernels(){
        return "__kernel void subFLOATFLOATFLOAT(__global const float * A, const float value, __global float * C){int i = get_global_id(0); C[i] = A[i] - value;}__kernel void subINTFLOATFLOAT(__global const int * A, const float value, __global float * C){int i = get_global_id(0); C[i] = A[i] - value;}__kernel void subFLOATINTFLOAT(__global const float * A, const int value, __global float * C){int i = get_global_id(0); C[i] = A[i] - value;}__kernel void subINTINTINT(__global const int * A, const int value, __global int * C){int i = get_global_id(0); C[i] = A[i] - value;}__kernel void subBufferFLOATFLOATFLOAT(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] - B[i];}__kernel void subBufferINTFLOATFLOAT(__global const int * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] - B[i];}__kernel void subBufferFLOATINTFLOAT(__global const float * A, __global const int * B, __global float * C){int i = get_global_id(0); C[i] = A[i] - B[i];}__kernel void subBufferINTINTINT(__global const int * A, __global const int * B, __global int * C){int i = get_global_id(0); C[i] = A[i] - B[i];}";
    }


    private String getSpecialKernels(){
        return "__kernel void compareGTFLOATFLOATFLOAT(__global const float * A, const float B, __global float * C){int i = get_global_id(0); if(A[i] > B){C[i] = 1.0f;} else {C[i] = 0.0f;}}__kernel void compareGTFLOATINTFLOAT(__global const float * A, const int B, __global float * C){int i = get_global_id(0); if(A[i] > B){C[i] = 1.0f;} else {C[i] = 0.0f;}}__kernel void compareGTINTFLOATFLOAT(__global const int * A, const float B, __global float * C){int i = get_global_id(0); if(A[i] > B){C[i] = 1.0f;} else {C[i] = 0.0f;}}__kernel void compareGTINTINTFLOAT(__global const int * A, const int B, __global float * C){int i = get_global_id(0); if(A[i] > B){C[i] = 1.0f;} else {C[i] = 0.0f;}} __kernel void absBufferFLOATFLOAT(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = fabs(A[i]);} __kernel void absBufferINTINT(__global const int * A, __global int * C){int i = get_global_id(0); C[i] = abs(A[i]);} __kernel void expBufferFLOATFLOAT(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = exp(A[i]);} __kernel void expBufferINTFLOAT(__global const int * A, __global float * C){int i = get_global_id(0); C[i] = exp((float)A[i]);} __kernel void logBufferFLOATFLOAT(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = log(A[i]);} __kernel void logBufferINTFLOAT(__global const int * A, __global float * C){int i = get_global_id(0); C[i] = log((float)A[i]);}__kernel void cosBufferFLOATFLOAT(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = cos(A[i]);} __kernel void cosBufferINTFLOAT(__global const int * A, __global float * C){int i = get_global_id(0); C[i] = cos((float)A[i]);}__kernel void sinBufferFLOATFLOAT(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = sin(A[i]);} __kernel void sinBufferINTFLOAT(__global const int * A, __global float * C){int i = get_global_id(0); C[i] = sin((float)A[i]);} __kernel void sqrtBufferFLOATFLOAT(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = sqrt(A[i]);} __kernel void sqrtBufferINTFLOAT(__global const int * A, __global float * C){int i = get_global_id(0); C[i] = sqrt((float)A[i]);} __kernel void blend(__global const float * A, __global const float * B, __global const float * C, __global float * D){ int i = get_global_id(0); D[i] = mix(A[i], B[i], C[i]);} __kernel void repeatFullFLOATINTFLOAT(__global const float * A, int repetition, __global float * B) { int i = get_global_id(0); int size = get_global_size(0); for(int x = 0; x < repetition; x++){B[i + x * size] = A[i];}} __kernel void repeatFullINTINTINT(__global const int * A, int repetition, __global int * B) { int i = get_global_id(0); int size = get_global_size(0); for(int x = 0; x < repetition; x++){B[i + x * size] = A[i];}} __kernel void repeatEachNumberFLOATINTFLOAT(__global const float * A, int repetition, __global float * B) { int i = get_global_id(0); for(int x = 0; x < repetition; x++){B[i * repetition + x] = A[i];}}__kernel void repeatEachNumberINTINTINT(__global const int * A, int repetition, __global int * B) { int i = get_global_id(0); for(int x = 0; x < repetition; x++){B[i * repetition + x] = A[i];}} __kernel void lshlINTINTINT(__global const int * A, int amount, __global int * B){int i = get_global_id(0); B[i] = A[i] << amount;} __kernel void ashrINTINTINT(__global const int * A, const int amount, __global int * B){int i = get_global_id(0); B[i] = A[i] >> amount;} __kernel void orINTINTINT(__global const int * A, const int B, __global int * C){int i = get_global_id(0); C[i] = A[i] | B;}  __kernel void orBufferINTINTINT(__global const int * A, __global const int * B, __global int * C){int i = get_global_id(0); C[i] = A[i] | B[i];} __kernel void andINTINTINT(__global const int * A, int amount, __global int * B){int i = get_global_id(0); B[i] = A[i] & amount;} __kernel void andBufferINTINTINT(__global const int * A, __global const int * B, __global int * C){int i = get_global_id(0); C[i] = A[i] & B[i];} __kernel void maxFLOATFLOATFLOAT(__global const float * A, const float value, __global float * C){int i = get_global_id(0); C[i] = max(A[i], value);} __kernel void maxINTINTINT(__global const int * A, const int value, __global int * C){int i = get_global_id(0); C[i] = max(A[i], value);} __kernel void minFLOATFLOATFLOAT(__global const float * A, const float value, __global float * C){int i = get_global_id(0); C[i] = min(A[i], value);} __kernel void minINTINTINT(__global const int * A, const int value, __global int * C){int i = get_global_id(0); C[i] = min(A[i], value);} __kernel void toFloat(__global const int * A, __global float * C){int i = get_global_id(0); C[i] = (float)A[i];}    __kernel void toInt(__global const float * A, __global int * C){int i = get_global_id(0); C[i] = (int)A[i];}__kernel void rolFLOATINTFLOAT(__global const float * A, int amount, __global float * B){int i = get_global_id(0); int size = get_global_size(0); B[i] = A[(i + amount) % size];} __kernel void rolINTINTINT(__global const int * A, int amount, __global int * B){int i = get_global_id(0); int size = get_global_size(0); B[i] = A[(i + amount) % size];}    __kernel void rorFLOATINTFLOAT(__global const float * A, int amount, __global float * B){int i = get_global_id(0); int size = get_global_size(0); B[(i + amount) % size] = A[i];} __kernel void rorINTINTINT(__global const int * A, int amount, __global int * B){int i = get_global_id(0); int size = get_global_size(0); B[(i + amount) % size] = A[i];} __kernel void eachAreaFMA(__global const float * A,  __global const float * B, __global float * C, int width, int kernelwidth){int i = get_global_id(0);  float sum = 0.0f; for(int x = 0; x < kernelwidth * kernelwidth; x++){int row = x / kernelwidth * width;  sum = fma(A[i + i/(width - kernelwidth + 1) * (kernelwidth - 1) + row + x % kernelwidth], B[x], sum);} C[i] = sum;} __kernel void mulVectorBufferFLOATFLOATFLOAT(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); int size = get_global_size(0); for(int x = 0; x < size; x++){C[i] = fma(A[i * size + x], B[x], C[i]);}} __kernel void mulVectorBufferFLOATINTFLOAT(__global const float * A, __global const int * B, __global float * C){int i = get_global_id(0); int size = get_global_size(0); for(int x = 0; x < size; x++){C[i] = fma(A[i * size + x], (float)B[x], C[i]);}} __kernel void mulVectorBufferINTFLOATFLOAT(__global const int * A, __global const float * B, __global float * C){int i = get_global_id(0); int size = get_global_size(0); for(int x = 0; x < size; x++){C[i] = fma((float)A[i * size + x], B[x], C[i]);}} __kernel void mulVectorBufferINTINTINT(__global const int * A, __global const int * B, __global int * C){int i = get_global_id(0); int size = get_global_size(0); for(int x = 0; x < size; x++){C[i] = fma((float)A[i * size + x], B[x], C[i]);}}";
    }


    private String getDivKernels(){
        return "__kernel void divFLOATFLOATFLOAT(__global const float * A, const float value, __global float * C){int i = get_global_id(0); C[i] = A[i] / value;} __kernel void divINTFLOATFLOAT(__global const int * A, const float value, __global float * C){int i = get_global_id(0); C[i] = A[i] / value;} __kernel void divFLOATINTFLOAT(__global const float * A, const int value, __global float * C){int i = get_global_id(0); C[i] = A[i] / value;} __kernel void divBufferFLOATFLOATFLOAT(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] / B[i];} __kernel void divBufferINTFLOATFLOAT(__global const int * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] / B[i];} __kernel void divBufferFLOATINTFLOAT(__global const float * A, __global const int * B, __global float * C){int i = get_global_id(0); C[i] = A[i] / B[i];}";
    }

    private String getKernels() {
        return getFMAKernels() + getAddKernels() + getMulKernels() + getSubKernels() + getSpecialKernels() + getDivKernels();
    }
}
