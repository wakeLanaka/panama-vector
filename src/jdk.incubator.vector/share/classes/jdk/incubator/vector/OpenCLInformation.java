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

    // private Class[] types = new Class[]{FLoat.TYPE, Long.TYPE, Integer.TYPE};

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

    // private String[] createKernel(String name, String rhs){
    //     String[] kernel = new String[];
    //     for(int i = 0; i < types.length; i++) {
    //         var firstType = types[i];
    //         String kernelSignature = "__kernel void " + name;
    //         kernelSignature += firstType;
    //         for(int k = 0; k < types.length; k++){
    //             var secondType = types[k];
    //             kernelSignature += "(__global " + firstType.toString() + "* b1, ";
    //             kernelSignature += "__global " + secondType.toString() + "* b2, ";
    //             Class resultType;
    //             if(i < k){
    //                 resultType = firstType;
    //             } else {
    //                 resultType = secondType;
    //             }
    //             kernelSignature += "__global " + resultType.toString() + "* b3 ";
    //         }

    //     }
    // }

    private String getFMAKernels() {
        return "__kernel void matrix_fma(__global const float * A, __global const float * B, __global float * C, int m, int n, int k) { int i = get_global_id(0); int a = (i / n) * m + k; int b = i % n + k * n; C[i] = fma(A[a], B[b], C[i]);} __kernel void vector_fma(__global const float * A, __global const float * B, __global float * C, __global float * D){int i = get_global_id(0); D[i] = fma(A[i], B[i], C[i]);}";
    }

    private String getAddKernels() {
        return "__kernel void vector_add(__global const float * A, __global const float * B, __global float * C) { int i = get_global_id(0); C[i] = A[i] + B[i];}__kernel void sumreduce(__global const float * A, __global float * B, __local float * localData){const int gid = get_global_id(0); const int lid = get_local_id(0); const int localSize = get_local_size(0); localData[lid] = work_group_reduce_add(A[gid]); if(lid == 0){B[get_group_id(0)] = localData[0];}}";
    }

    private String getMulKernels(){
        return "__kernel void multiplyFFI(__global const float * A, __global float * C, const int factor){int i = get_global_id(0); C[i] = A[i] * factor;} __kernel void multiplyIII(__global const int * A, __global int * C, const int factor){int i = get_global_id(0); C[i] = A[i] * factor;} __kernel void multiplyIFF(__global const int * A, __global float * C, const float factor){int i = get_global_id(0); C[i] = A[i] * factor;} __kernel void multiplyFFF(__global const float * A, __global float * C, const float factor){int i = get_global_id(0); C[i] = A[i] * factor;} __kernel void vector_multiply(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] * B[i];} __kernel void vector_multiplyInt(__global const int * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] * B[i];}__kernel void multiplyDivide(__global const float * A, __global float * B, float v1, float v2, float v3){int i = get_global_id(0); B[i] = A[i] * v1 * v2 / v3;} __kernel void multiplyrange(__global const float * A, __global const float * B, __global float * C, int index1, int index2){int i = get_global_id(0); C[i] = A[i + index1] * B[i + index2];}__kernel void multiplyRepeat(__global float * A, __global float const * B, int size) {int i = get_global_id(0); A[i] = A[i] * B[i % size];}__kernel void multArea(__global const float * A,  __global const float * B, __global float * C, int offset, int widthA, int widthB){int i = get_global_id(0); C[i] = A[i % widthB + offset + widthA * (i / widthB)] * B[i];}";
    }

    private String getSubKernels(){
        return "__kernel void subtract(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] - B[i];}__kernel void subtraction_subtrahend(__global const float * A, __global float * C, const float subtrahend){int i = get_global_id(0); C[i] = A[i] - subtrahend;}";
    }

    private String getSpecialKernels(){
        return "__kernel void sqrt(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = sqrt(A[i]);}    __kernel void log(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = log(A[i]);}    __kernel void exp(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = exp(A[i]);}    __kernel void abs(__global const const float * A, __global float * C){int i = get_global_id(0); C[i] = fabs(A[i]);}    __kernel void blend(__global const float * A, __global const float * B, __global const float * C, __global float * D){ int i = get_global_id(0); D[i] = mix(A[i], B[i], C[i]);}    __kernel void compareGT(__global const float * A, __global float * C, const float B){int i = get_global_id(0); if(A[i] > B){C[i] = 1.0f;} else {C[i] = 0.0f;}} __kernel void cos(__global const float * A, __global float * B){int i = get_global_id(0); B[i] = cos(A[i]);} __kernel void sinx(__global const float * A, __global float * B){int i = get_global_id(0); B[i] = sin(A[i]);}__kernel void repeat1(__global const float * A, __global float * B, int repetition, int length) { int i = get_global_id(0); for(int x = 0; x < repetition; x++){B[i + x * length] = A[i];}} __kernel void repeat2(__global const float * A, __global float * B, int repetition) { int i = get_global_id(0); for(int x = 0; x < repetition; x++){B[i * repetition + x] = A[i];}}__kernel void lshl(__global const int * A, __global int * B, int amount){int i = get_global_id(0); B[i] = A[i] << amount;}    __kernel void ashr(__global const int * A, __global int * B, int amount){int i = get_global_id(0); B[i] = A[i] >> amount;}    __kernel void vector_bitwiseor(__global const int * A, __global const int * B, __global int * C){int i = get_global_id(0); C[i] = A[i] | B[i];}    __kernel void bitwiseand(__global const int * A, __global int * B, int andValue){int i = get_global_id(0); B[i] = A[i] & andValue;}__kernel void vector_max(__global const float * A, __global float * C, float value){int i = get_global_id(0); C[i] = max(A[i], value);}    __kernel void vector_min(__global const float * A, __global float * C, float value){int i = get_global_id(0); C[i] = min(A[i], value);}    __kernel void toFloat(__global const int * A, __global float * C){int i = get_global_id(0); C[i] = (float)A[i];}    __kernel void toInt(__global const float * A, __global int * C){int i = get_global_id(0); C[i] = (int)A[i];}__kernel void rol(__global const float * A, __global float * B, int amount, int size){int i = get_global_id(0); B[i] = A[(i + amount) % size];}    __kernel void ror(__global const float * A, __global float * B, int amount, int size){int i = get_global_id(0); B[(i + amount) % size] = A[i];} __kernel void eachAreaFMA(__global const float * A,  __global const float * B, __global float * C, int width, int kernelwidth){int i = get_global_id(0);  float sum = 0.0f; for(int x = 0; x < kernelwidth * kernelwidth; x++){int row = x / kernelwidth * width;  sum = fma(A[i + i/(width - kernelwidth + 1) * (kernelwidth - 1) + row + x % kernelwidth], B[x], sum);} C[i] = sum;}";
    }

    private String getDivKernels(){
        return "__kernel void vector_division(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] / B[i];} __kernel void division(__global const float * A, __global float * C, const float divisor){int i = get_global_id(0); C[i] = A[i] / divisor;}";
    }

    private String getFunctionKernels(){
        return     "float cdf(float inp) {   float Y = 0.2316419f;  float A1 = 0.31938153f;  float A2 = -0.356563782f;  float A3 = 1.781477937f;  float A4 = -1.821255978f;  float A5 = 1.330274429f;  float PI = M_PI_F;    float x = fabs(inp);  float vterm = 1 / (1 + x * Y);  float vterm_pow2 = vterm * vterm;  float vterm_pow3 = vterm_pow2 * vterm;  float vterm_pow4 = vterm_pow2 * vterm_pow2;  float vterm_pow5 = vterm_pow2 * vterm_pow3;  float part1 = 1 / sqrt(2 * PI) * exp(x * -x * 0.5f);  float part2 = vterm * A1 + vterm_pow2 * A2 + vterm_pow3 * A3 + vterm_pow4 * A4 + vterm_pow5 * A5;  if (inp >= 0.0f){ return 1.0f - part1 * part2;} else{      return part1 * part2;}}    __kernel void blackscholes(float sig, float r, __global float * x, __global float * call, __global float * put, __global float * t, __global float * s0) {  int i = get_global_id(0);  float sig_sq_by2 = 0.5f * sig * sig;  float log_s0byx = log(s0[i] / x[i]);  float sig_sqrt_t = sig * sqrt(t[i]);  float exp_neg_rt = exp(-r * t[i]);  float d1 = (log_s0byx + (r + sig_sq_by2) * t[i])/(sig_sqrt_t);  float d2 = d1 - sig_sqrt_t;  call[i] = s0[i] * cdf(d1) - exp_neg_rt * x[i] * cdf(d2);  put[i]  = call[i] + exp_neg_rt - s0[i];} __kernel void dft(__global const float * inReal, __global float * outReal, __global const float * inImag, __global float * outImag, int size) { int i = get_global_id(0); float sumReal = 0; float sumImag = 0; for(int t = 0; t < size; t++){ float angle = (i * 2 * M_PI_F * t)/size; sumReal += (inReal[t] * cos(angle)) + (inImag[t] * sin(angle)); sumImag += -(inReal[t] * sin(angle)) + (inImag[t] * cos(angle)); } outReal[i] = sumReal; outImag[i] = sumImag;}";
    }

    private String getKernels() {
        return getFMAKernels() + getAddKernels() + getMulKernels() + getSubKernels() + getSpecialKernels() + getDivKernels() + getFunctionKernels();
    }
}
