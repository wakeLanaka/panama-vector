package jdk.incubator.vector;

import java.util.Arrays;
import jdk.internal.vm.vector.SVMBufferSupport;
import jdk.incubator.vector.GPUInformation;
import java.util.function.*;

/**
    TODO Use generics for different types
    TODO Add Typesafety
*/
public class SVMBuffer {

    private GPUInformation info;

    /**
     *  Address to the SVMBuffer
     */
    public long svmBuffer;

    /**
     *  Amount of elements in the SVMBuffer
     */
    public final int length;

    private SVMBuffer(GPUInformation info, float[] array) {
        this.info = info;
        this.length = array.length;
        this.svmBuffer = SVMBufferSupport.CopyFromArray(info.GetContext(), info.GetCommandQueue(), array);
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
     *  Loads a SVMBuffer from an array of type {@code float[]}
     *  @param info for the gpu
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
     *  @return SVMBuffer of FMA this * factor + summand
     */
    public SVMBuffer Fma(SVMBuffer factor, SVMBuffer summand){
        SVMBuffer result = new SVMBuffer(info, length);

        SVMBufferSupport.FmaSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, summand.svmBuffer, result.svmBuffer, summand.length);
        return result;
    }

    /**
     *  Fused multiply-add (FMA) of SVMBuffers.
     *  @param factor the factor used in FMA
     *  @param summand the summand and result used in FMA
     *  @param result the result of the FMA
     */
    public void Fma(SVMBuffer factor, SVMBuffer summand, SVMBuffer result){
        SVMBufferSupport.FmaSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, summand.svmBuffer, result.svmBuffer, summand.length);
    }

    /**
     *  Reduces this SVMBuffer to a single value using addition
     *  @return the result of the sum reduction
     */
    public float SumReduce(){
        return SVMBufferSupport.SumReduce(info.GetContext(), info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.length);
    }

    /**
     *  Elementwise addition of this SVMBuffer and the summand SVMBuffer
     *  @param summand The SVMBuffer which gets added the this SVMBuffer
     *  @return a new SVMBuffer containing the added elements
     */
    public SVMBuffer Add(SVMBuffer summand) {
        SVMBuffer results = new SVMBuffer(info, this.length);
        SVMBufferSupport.AddSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, summand.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Elementwise addition of this SVMBuffer and the summand SVMBuffer
     *  @param summand The SVMBuffer which gets added the this SVMBuffer
     *  @return a new SVMBuffer containing the added elements
     */
    public SVMBuffer AddInPlace(SVMBuffer summand) {
        SVMBufferSupport.AddSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, summand.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Elementwise addition of this SVMBuffer and the summand SVMBuffer
     *  @param info of the gpu
     *  @param result a new SVMBuffer containing the added elements
     *  @param summand The SVMBuffer which gets added the this SVMBuffer
     */
    public static void Add(GPUInformation info, SVMBuffer result, SVMBuffer summand) {
        SVMBufferSupport.AddSVMBuffer(info.GetProgram(), info.GetCommandQueue(), result.svmBuffer, summand.svmBuffer, result.svmBuffer, result.length);
    }

    /**
     *  Subtracts the subtrahend from this SVMBuffer
     *  @param subtrahend of the subtraction
     *  @return the result of the subtraction
     */
    public SVMBuffer Subtract(SVMBuffer subtrahend){
        SVMBuffer result = new SVMBuffer(info, length);

        SVMBufferSupport.Subtract(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, subtrahend.svmBuffer, result.svmBuffer, this.length);
        return result;
    }

    /**
     *  Subtracts the subtrahend from this SVMBuffer
     *  @param subtrahend of the subtraction
     *  @return the result of the subtraction
     */
    public SVMBuffer SubtractInPlace(SVMBuffer subtrahend){
        SVMBufferSupport.Subtract(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, subtrahend.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Subtracts the subtrahend from this SVMBuffer
     *  @param info of the gpu
     *  @param subtrahend of the subtraction
     *  @param result the result of the subtraction
     */
    public static void Subtract(GPUInformation info, SVMBuffer result, SVMBuffer subtrahend){
        SVMBufferSupport.Subtract(info.GetProgram(), info.GetCommandQueue(), result.svmBuffer, subtrahend.svmBuffer, result.svmBuffer, result.length);
    }

    /**
     *  Subtracts this SVMBuffer from the minuend
     *  @param minuend of the subtraction
     *  @return the subtracted SVMBuffer
     */
    public SVMBuffer SubtractionMinuend(float minuend){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.SubtractionMinuend(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, minuend, this.length);
        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer Multiply(float factor){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, factor, this.length);

        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer MultiplyInPlace(float factor){
        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, factor, this.length);

        return this;
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer Multiply(SVMBuffer factors){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factors.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer MultiplyInPlace(SVMBuffer factors){
        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factors.svmBuffer, this.svmBuffer, this.length);

        return this;
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer MultiplyInPlaceRepeat(SVMBuffer factors){
        SVMBufferSupport.MultiplyInPlaceRepeat(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factors.svmBuffer, factors.length, this.length);
        return this;
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param info of the gpu
     *  @param result the multiplied SVMBuffer
     *  @param factors of the multiplication
     */
    public static void Multiply(GPUInformation info, SVMBuffer result, SVMBuffer factors){
        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), result.svmBuffer, factors.svmBuffer, result.svmBuffer, result.length);
    }

    /**
     *  Find the square root of this SVMBuffer
     *  @return the new SVMBuffer containing the square roots
     */
    public SVMBuffer Sqrt(){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Sqrt(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Find the square root of this SVMBuffer
     *  @return the new SVMBuffer containing the square roots
     */
    public SVMBuffer SqrtInPlace(){
        SVMBufferSupport.Sqrt(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);

        return this;
    }

    /**
     *  Creates a new SVMBuffer initialized with @param value
     *  @param info for the gpu
     *  @param value of the elements
     *  @param length of the SVMBuffer
     *  @return initialized SVMBuffer
     */
    public static SVMBuffer Broadcast(GPUInformation info, float value, int length) {
        float[] array = new float[length];
        Arrays.fill(array, value);
        return new SVMBuffer(info, array);
    }

    /**
     *  Divides this SVMBuffer with the divisor
     *  @param divisor of the division
     *  @return the divided SVMBuffer
     */
    public SVMBuffer Division(float divisor){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, divisor, this.length);

        return results;
    }

    /**
     *  Divides this SVMBuffer with the divisor
     *  @param divisor of the division
     *  @return the divided SVMBuffer
     */
    public SVMBuffer DivisionInPlace(float divisor){
        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, divisor, this.length);

        return this;
    }

    /**
     *  Divides this SVMBuffer with the divisor SVMBuffer
     *  @param divisors of the division
     *  @return the divided SVMBuffer
     */
    public SVMBuffer Division(SVMBuffer divisors){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, divisors.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Divides this SVMBuffer with the divisor SVMBuffer
     *  @param divisors of the division
     *  @return the divided SVMBuffer
     */
    public SVMBuffer DivisionInPlace(SVMBuffer divisors){
        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, divisors.svmBuffer, this.svmBuffer, this.length);

        return this;
    }

    /**
     *  Divides this SVMBuffer with the divisor SVMBuffer
     *  @param info of the gpu
     *  @param result the divided SVMBuffer
     *  @param divisors of the division
     */
    public static void Division(GPUInformation info, SVMBuffer result, SVMBuffer divisors){

        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), result.svmBuffer, divisors.svmBuffer, result.svmBuffer, result.length);
    }

    /**
     *  Calculates the natural logarithm of this SVMBuffer
     *  @return this SVMBuffer containing the natural logarithms
     */
    public SVMBuffer LogInPlace(){
        SVMBufferSupport.Log(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);

        return this;
    }

    /**
     *  Calculates the natural logarithm of this SVMBuffer
     *  @return the new SVMBuffer containing the natural logarithms
     */
    public SVMBuffer Log(){
        SVMBuffer result = new SVMBuffer(info, length);

        SVMBufferSupport.Log(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, result.svmBuffer, this.length);

        return result;
    }

    /**
     *  Calculates the cos
     *  @return this SVMBuffer after applying cos
     */
    public SVMBuffer CosInPlace(){
        SVMBufferSupport.Cos(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Calculates the cos
     *  @return the new SVMBuffer after applying cos
     */
    public SVMBuffer Cos(){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Cos(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Calculates the sin
     *  @return this SVMBuffer containing the sin
     */
    public SVMBuffer SinInPlace(){
        SVMBufferSupport.Sin(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Calculates the sin
     *  @return the new SVMBuffer containing the sin
     */
    public SVMBuffer Sin(){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Sin(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Calculates the base-e exponential of this SVMBuffer
     *  @return the new SVMBuffer containing the base-e exponentials
     */
    public SVMBuffer ExpInPlace(){
        SVMBufferSupport.Exp(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);

        return this;
    }

    /**
     *  Calculates the base-e exponential of this SVMBuffer
     *  @return the new SVMBuffer containing the base-e exponentials
     */
    public SVMBuffer Exp(){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Exp(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Calculates the absolute value of this SVMBuffer
     *  @return the new SVMBuffer containing the absolute values
     */
    public SVMBuffer AbsInPlace(){
        SVMBufferSupport.Abs(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);

        return this;
    }

    /**
     *  Calculates the absolute value of this SVMBuffer
     *  @return the new SVMBuffer containing the absolute values
     */
    public SVMBuffer Abs(){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Abs(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);

        return results;
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
        SVMBufferSupport.ReleaseSVMBuffer(info.GetContext(), info.GetCommandQueue(), this.svmBuffer);
        this.svmBuffer = 0;
    }

    /**
     *  Compares each value of this svmBuffer if it is greater than the comparee
     *  @param comparee of the greater than comparison
     *  @return the mask of the greater than comparison
     */
    public SVMBuffer CompareGT(float comparee){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.CompareGT(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, comparee, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Calculates the BlackScholes on the GPU using a specified OpenCL kernel
     *  @param info of the gpu
     *  @param sig sig
     *  @param r r
     *  @param xBuffer x
     *  @param callBuffer call
     *  @param putBuffer put
     *  @param tBuffer t
     *  @param s0Buffer s0
     */
    public static void BlackScholes(GPUInformation info, float sig, float r, SVMBuffer xBuffer, SVMBuffer callBuffer, SVMBuffer putBuffer, SVMBuffer tBuffer, SVMBuffer s0Buffer){

        SVMBufferSupport.BlackScholes(info.GetProgram(), info.GetCommandQueue(), sig, r, xBuffer.svmBuffer, callBuffer.svmBuffer, putBuffer.svmBuffer, tBuffer.svmBuffer, s0Buffer.svmBuffer, xBuffer.length);
    }

    /**
     *  Calculates the linear blend of x (this) and y implemented as: x + (y - x) * a
     *  @param comparee y parameter
     *  @param mask a parameter
     *  @return the new SVMBuffer
     */
    public SVMBuffer Blend(SVMBuffer comparee, SVMBuffer mask){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Blend(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, comparee.svmBuffer, mask.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Calculates the linear blend of x (this) and y implemented as: x + (y - x) * a
     *  @param comparee y parameter
     *  @param mask a parameter
     *  @return this SVMBuffer
     */
    public SVMBuffer BlendInPlace(SVMBuffer comparee, SVMBuffer mask){
        SVMBufferSupport.Blend(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, comparee.svmBuffer, mask.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Creates a new SVMBuffer with all values from 0 to length
     *  @param info for the gpu
     *  @param length of the SVMBuffer
     *  @return new SVMBuffer
     */
    public static SVMBuffer Iota(GPUInformation info, int length){
        float[] results = new float[length];
        for(int i = 0; i < length; i++){
            results[i] = i;
        }
        return new SVMBuffer(info, results);
    }

    /**
     *  Calculates the DFT on the GPU using a specified OpenCL kernel
     *  @param info for the gpu
     *  @param b1 input real SVMBuffer
     *  @param b2 output real SVMBuffer
     *  @param b3 input imag SVMBuffer
     *  @param b4 output imag SVMBuffer
     */
    public static void DFT(GPUInformation info, SVMBuffer b1, SVMBuffer b2, SVMBuffer b3, SVMBuffer b4){
        SVMBufferSupport.DFT(info.GetProgram(), info.GetCommandQueue(), b1.svmBuffer, b2.svmBuffer, b3.svmBuffer, b4.svmBuffer, b1.length);
    }

    static final class OpenCLInformation implements GPUInformation {
        private long context;
        private long device;
        private long commandQueue;
        private long program;

        public OpenCLInformation() {
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
            return  "__kernel void matrix_fma(__global const float * A, __global const float * B, __global float * C, int m, int n, int k) { int i = get_global_id(0); int a = floor((float)i / n) * m + k; int b = i % n + k * n; C[i] = fma(A[a], B[b], C[i]);} __kernel void vector_fma(__global const float * A, __global const float * B, __global float * C, __global float * D){int i = get_global_id(0); D[i] = fma(A[i], B[i], C[i]);} __kernel void vector_add(__global const float * A, __global const float * B, __global float * C) { int i = get_global_id(0); C[i] = A[i] + B[i];} __kernel void vector_multiply(__global float * A, __global float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] * B[i];} __kernel void multiply(__global float * A, __global float * C, const float factor){int i = get_global_id(0); C[i] = A[i] * factor;} __kernel void subtract(__global const float * A, __global float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] - B[i];} __kernel void subtraction_minuend(__global const float * A, __global float * C, const float minuend){int i = get_global_id(0); C[i] = minuend - A[i];} __kernel void subtraction_subtrahend(__global float * A, __global float * C, const float subtrahend){int i = get_global_id(0); C[i] = A[i] - subtrahend;} __kernel void sqrt(__global float * A, __global float * C){int i = get_global_id(0); C[i] = sqrt(A[i]);} __kernel void vector_division(__global float * A, __global float * B, __global float * C){int i = get_global_id(0); C[i] = A[i] / B[i];} __kernel void division(__global float * A, __global float * C, const float divisor){int i = get_global_id(0); C[i] = A[i] / divisor;} __kernel void log(__global float * A, __global float * C){int i = get_global_id(0); C[i] = log(A[i]);} __kernel void exp(__global float * A, __global float * C){int i = get_global_id(0); C[i] = exp(A[i]);} __kernel void abs(__global const float * A, __global float * C){int i = get_global_id(0); C[i] = fabs(A[i]);}  __kernel void blend(__global float * A, __global float * B, __global float * C, __global float * D){ int i = get_global_id(0); D[i] = mix(A[i], B[i], C[i]);} __kernel void compareGT(__global float * A, __global float * C, const float B){int i = get_global_id(0); if(A[i] > B){C[i] = 1.0f;} else {C[i] = 0.0f;}} float cdf(float inp) {   float Y = 0.2316419f;  float A1 = 0.31938153f;  float A2 = -0.356563782f;  float A3 = 1.781477937f;  float A4 = -1.821255978f;  float A5 = 1.330274429f;  float PI = M_PI_F;    float x = fabs(inp);  float vterm = 1 / (1 + x * Y);  float vterm_pow2 = vterm * vterm;  float vterm_pow3 = vterm_pow2 * vterm;  float vterm_pow4 = vterm_pow2 * vterm_pow2;  float vterm_pow5 = vterm_pow2 * vterm_pow3;  float part1 = 1 / sqrt(2 * PI) * exp(x * -x * 0.5f);  float part2 = vterm * A1 + vterm_pow2 * A2 + vterm_pow3 * A3 + vterm_pow4 * A4 + vterm_pow5 * A5;  if (inp >= 0.0f){ return 1.0f - part1 * part2;} else{      return part1 * part2;}} __kernel void blackscholes(float sig, float r, __global float * x, __global float * call, __global float * put, __global float * t, __global float * s0) {  int i = get_global_id(0);  float sig_sq_by2 = 0.5f * sig * sig;  float log_s0byx = log(s0[i] / x[i]);  float sig_sqrt_t = sig * sqrt(t[i]);  float exp_neg_rt = exp(-r * t[i]);  float d1 = (log_s0byx + (r + sig_sq_by2) * t[i])/(sig_sqrt_t);  float d2 = d1 - sig_sqrt_t;  call[i] = s0[i] * cdf(d1) - exp_neg_rt * x[i] * cdf(d2);  put[i]  = call[i] + exp_neg_rt - s0[i];} __kernel void cos(__global const float * A, __global float * B){int i = get_global_id(0); B[i] = cos(A[i]);} __kernel void sinx(__global const float * A, __global float * B){int i = get_global_id(0); B[i] = sin(A[i]);}  __kernel void dft(__global const float * inReal, __global float * outReal, __global const float * inImag, __global float * outImag, int size) { int i = get_global_id(0); float sumReal = 0; float sumImag = 0; for(int t = 0; t < size; t++){ float angle = (i * 2 * M_PI_F * t)/size; sumReal += (inReal[t] * cos(angle)) + (inImag[t] * sin(angle)); sumImag += -(inReal[t] * sin(angle)) + (inImag[t] * cos(angle)); } outReal[i] = sumReal; outImag[i] = sumImag;}   __kernel void multiplyDivide(__global const float * A, __global float * B, float v1, float v2, float v3){int i = get_global_id(0); B[i] = A[i] * v1 * v2 / v3;} __kernel void sumreduce(__global float * A, __global float * B, __global float * result, int size, __local float * localData){const int gid = get_global_id(0); const int lid = get_local_id(0); const int localSize = get_local_size(0); localData[lid] = work_group_reduce_add(A[gid]); if(lid == 0){B[get_group_id(0)] = localData[0];} if(gid == 0){float sum = 0.0f; for(int i = 0; i < get_num_groups(0); i++){sum += B[i];} result[0] = sum;}}";
        }
    }


    /**
     * Creates the information to use the gpu with opencl
     */
    public static final GPUInformation SPECIES_PREFERRED
        = new OpenCLInformation();
}
