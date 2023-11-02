package jdk.incubator.vector;

import jdk.internal.vm.vector.SVMBufferSupport;
import jdk.incubator.vector.GPUInformation;
import java.util.function.*;

/**
    TODO Call release... when instance gets destroyed
    TODO Use generics for different types
    TODO Add Typesafety
    abcd
*/
public class SVMBuffer {

    private GPUInformation info;

    /**
     *  Pointer to the SVMBuffer
     */
    public final long svmBuffer;

    /**
     *  Amount of elements in the SVMBuffer
     */
    public final int length;

    private SVMBuffer(GPUInformation info, float[] array) {
        this.info = info;
        this.length = array.length;
        this.svmBuffer = SVMBufferSupport.CopyFromFloatArray(info.GetContext(), info.GetCommandQueue(), array);
    }

    private SVMBuffer(GPUInformation info, int[] array) {
        this.info = info;
        this.length = array.length;
        this.svmBuffer = SVMBufferSupport.CopyFromArray(info.GetContext(), info.GetCommandQueue(), array);
    }

    private SVMBuffer(GPUInformation info, int length) {
        this.info = info;
        this.length = length;
        this.svmBuffer = SVMBufferSupport.CreateWriteSVMBuffer(info.GetContext(), length);
    }

    /**
     *  Loads a SVMBuffer from an array of type {@code int[]}
     *  @param info informations for the gpu
     *  @param array the array
     *  @return the SVMBuffer loaded from the array
     */
    public static SVMBuffer fromArray(GPUInformation info, int[] array) {
        return new SVMBuffer(info, array);
    }

    /**
     *  Loads a SVMBuffer from an array of type {@code float[]}
     *  @param info informations for the gpu
     *  @param array the array
     *  @return the SVMBuffer loaded from the array
     */
    public static SVMBuffer fromArray(GPUInformation info, float[] array) {
        return new SVMBuffer(info, array);
    }

    /**
     *  Fused multiply-add (FMA) of SVMBuffers. This method can be used for
     *  array multiplication.
     *  @param factor the factor used in FMA
     *  @param summand the summand and result used in FMA
     *  @param m amount of columns of this SVMBuffers
     *  @param n amount of columns of the factor SVMBuffer
     *  @param k the kth column of this SVMBuffer is gets calculated with the
     *  kth row of the factor SVMBuffer
     */
    public void matrixFma(SVMBuffer factor, SVMBuffer summand, int m, int n, int k) {
        SVMBufferSupport.MatrixFmaSVMBuffer(this.info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, summand.svmBuffer, m, n, k, summand.length);
    }

    /**
     *  Fused multiply-add (FMA) of SVMBuffers.
     *  @param factor the factor used in FMA
     *  @param summand the summand and result used in FMA
     */
    public void Fma(SVMBuffer factor, SVMBuffer summand){
        SVMBufferSupport.FmaSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, summand.svmBuffer, summand.length);
    }

    /**
     *  Reduces this SVMBuffer to a single value using addition
     *  @return the result of the reduction
     */
    public float reduceAdd() {
        return SVMBufferSupport.ReduceAdd(info.GetContext(), info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.length);
    }

    /**
     *  Elementwise addition of this SVMBuffer and the summand SVMBuffer
     *  @param summand The SVMBuffer which gets added the this SVMBuffer
     *  @return a new SVMBuffer containing the added elements
     */
    public SVMBuffer Add(SVMBuffer summand) {
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.AddSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, summand.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Writes a value at index of this SVMBuffer
     *  @param index of the element
     *  @param value to set
     */
    public void writeSVMBuffer(int index, int value) {
        SVMBufferSupport.WriteSVMBuffer(info.GetCommandQueue(), this.svmBuffer, index, this.length, value);
    }

    /**
     *  Returns the value of a single element
     *  @param index of the element
     *  @return the value at the index
     */
    public int readSVMBuffer(int index) {
        return SVMBufferSupport.ReadSVMBuffer(info.GetCommandQueue(), this.svmBuffer, index, this.length);
    }

    /**
     *  Stores this SVMBuffer into an Array of type {@code int[]}
     *  @param array the array of type {@code int[]}
     */
    public void intoArray(int[] array) {
        SVMBufferSupport.CopyToArray(info.GetContext(), info.GetCommandQueue(), this.svmBuffer, array);
    }

    /**
     *  Stores this SVMBuffer into an Array of type {@code float[]}
     *  @param array the array of type {@code float[]}
     */
    public void intoArray(float[] array) {
        SVMBufferSupport.CopyToFloatArray(info.GetContext(), info.GetCommandQueue(), this.svmBuffer, array);
    }

    /**
     *  Deallocates the Memory of this SVMBuffer
     */
    public void releaseSVMBuffer() {
        SVMBufferSupport.ReleaseSVMBuffer(info.GetContext(), this.svmBuffer);
    }

    static final class OpenCLInformation implements GPUInformation {
        private long context;
        private long device;
        private long commandQueue;
        private long program;

        private OpenCLInformation() {
            device = SVMBufferSupport.CreateDevice();
            context = SVMBufferSupport.CreateContext(device);
            program = SVMBufferSupport.CreateProgram(context, getKernels());
            commandQueue = SVMBufferSupport.CreateCommandQueue(context, device);
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
            TODO Solve this with a Strategy Pattern!
            TODO Remove atomicadd (#pragma OPENCL EXTENSION cl_ext_float_atomics : enable)
        */
        private String getKernels() {
            return  "__kernel void matrix_fma(__global const float * A, __global const float * B, __global float * C, int m, int n, int k) { int i = get_global_id(0); int a = floor((float)i / n) * m + k; int b = i % n + k * n; C[i] = fma(A[a], B[b], C[i]);} __kernel void vector_fma(__global const float * A, __global const float * B, __global float * C){int i = get_global_id(0); C[i] = fma(A[i], B[i], C[i]);} __kernel void vector_add(__global const int * A, __global const int * B, __global int * C, int N) { int i = get_global_id(0); if(i < N){ C[i] = A[i] + B[i]; } } inline float atomicadd(volatile __global float* address, const float value){    float old = value;    while ((old = atomic_xchg(address, atomic_xchg(address, 0.0f)+old))!=0.0f); return old;} __kernel void vector_reduce(__global const float * A, __global float * result) { result[0] = 0; int i = get_global_id(0); int local_id = get_local_id(0); float res = work_group_reduce_add(A[i]); if(local_id == 0){atomicadd(result, res);}}";
        }

    }

    /**
     * Creates the information to use the gpu with opencl
     */
    public static final GPUInformation SPECIES_PREFERRED
        = new OpenCLInformation();
}
