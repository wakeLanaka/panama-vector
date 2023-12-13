package jdk.incubator.vector;

import java.util.Arrays;
import jdk.internal.vm.vector.SVMBufferSupport;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.OpenCLInformation;
import java.util.function.*;

/**
 *  An OpenCL SVMBuffer with similar API to vector api
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
        this.svmBuffer = SVMBufferSupport.CreateReadWriteFloatSVMBuffer(info.GetContext(), length);
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
    public SVMBuffer fma(SVMBuffer factor, SVMBuffer summand){
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
    public void fma(SVMBuffer factor, SVMBuffer summand, SVMBuffer result){
        SVMBufferSupport.FmaSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, summand.svmBuffer, result.svmBuffer, summand.length);
    }

    /**
     *  Reduces this SVMBuffer to a single value using addition
     *  @return the result of the sum reduction
     */
    public float SumReduce(){
        return SVMBufferSupport.SumReduce(info.GetContext(), info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, info.GetMaxWorkGroupSize(), this.length);
    }

    /**
     *  Elementwise addition of this SVMBuffer and the summand SVMBuffer
     *  @param summand The SVMBuffer which gets added the this SVMBuffer
     *  @return a new SVMBuffer containing the added elements
     */
    public SVMBuffer add(SVMBuffer summand) {
        SVMBuffer results = new SVMBuffer(info, this.length);
        SVMBufferSupport.AddSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, summand.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Elementwise addition of this SVMBuffer and the summand SVMBuffer
     *  @param summand The SVMBuffer which gets added the this SVMBuffer
     *  @return a new SVMBuffer containing the added elements
     */
    public SVMBuffer addInPlace(SVMBuffer summand) {
        SVMBufferSupport.AddSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, summand.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Subtracts the subtrahend from this SVMBuffer
     *  @param subtrahend of the subtraction
     *  @return the result of the subtraction
     */
    public SVMBuffer sub(SVMBuffer subtrahend){
        SVMBuffer result = new SVMBuffer(info, length);

        SVMBufferSupport.Subtract(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, subtrahend.svmBuffer, result.svmBuffer, this.length);
        return result;
    }

    /**
     *  Subtracts the subtrahend from this SVMBuffer
     *  @param subtrahend of the subtraction
     *  @return the result of the subtraction
     */
    public SVMBuffer subInPlace(SVMBuffer subtrahend){
        SVMBufferSupport.Subtract(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, subtrahend.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Subtracts the subtrahend from this SVMBuffer
     *  @param subtrahend of the subtraction
     *  @return the subtracted SVMBuffer
     */
    public SVMBuffer sub(float subtrahend){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Subtract(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, subtrahend, this.length);
        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mul(float factor){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, factor, this.length);

        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mulInPlace(float factor){
        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, factor, this.length);

        return this;
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mul(SVMBuffer factors){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factors.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mulInPlace(SVMBuffer factors){
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
     *  Find the square root of this SVMBuffer
     *  @return the new SVMBuffer containing the square roots
     */
    public SVMBuffer sqrt(){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Sqrt(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Find the square root of this SVMBuffer
     *  @return the new SVMBuffer containing the square roots
     */
    public SVMBuffer sqrtInPlace(){
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
    public static SVMBuffer broadcast(GPUInformation info, float value, int length) {
        float[] array = new float[length];
        Arrays.fill(array, value);
        return new SVMBuffer(info, array);
    }

    /**
     *  Divides this SVMBuffer with the divisor
     *  @param divisor of the division
     *  @return the divided SVMBuffer
     */
    public SVMBuffer div(float divisor){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, divisor, this.length);

        return results;
    }

    /**
     *  Divides this SVMBuffer with the divisor
     *  @param divisor of the division
     *  @return the divided SVMBuffer
     */
    public SVMBuffer divInPlace(float divisor){
        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, divisor, this.length);

        return this;
    }

    /**
     *  Divides this SVMBuffer with the divisor SVMBuffer
     *  @param divisors of the division
     *  @return the divided SVMBuffer
     */
    public SVMBuffer div(SVMBuffer divisors){
        SVMBuffer results = new SVMBuffer(info, length);

        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, divisors.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  Divides this SVMBuffer with the divisor SVMBuffer
     *  @param divisors of the division
     *  @return the divided SVMBuffer
     */
    public SVMBuffer divInPlace(SVMBuffer divisors){
        SVMBufferSupport.Division(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, divisors.svmBuffer, this.svmBuffer, this.length);

        return this;
    }

    /**
     *  Calculates the natural logarithm of this SVMBuffer
     *  @return the new SVMBuffer containing the natural logarithms
     */
    public SVMBuffer log(){
        SVMBuffer result = new SVMBuffer(info, length);
        SVMBufferSupport.Log(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, result.svmBuffer, this.length);
        return result;
    }

    /**
     *  Calculates the natural logarithm of this SVMBuffer
     *  @return this SVMBuffer containing the natural logarithms
     */
    public SVMBuffer logInPlace(){
        SVMBufferSupport.Log(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Calculates the cos
     *  @return the new SVMBuffer after applying cos
     */
    public SVMBuffer cos(){
        SVMBuffer results = new SVMBuffer(info, length);
        SVMBufferSupport.Cos(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Calculates the cos
     *  @return this SVMBuffer after applying cos
     */
    public SVMBuffer cosInPlace(){
        SVMBufferSupport.Cos(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Calculates the sin
     *  @return the new SVMBuffer containing the sin
     */
    public SVMBuffer sin(){
        SVMBuffer results = new SVMBuffer(info, length);
        SVMBufferSupport.Sin(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Calculates the sin
     *  @return this SVMBuffer containing the sin
     */
    public SVMBuffer sinInPlace(){
        SVMBufferSupport.Sin(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Calculates the base-e exponential of this SVMBuffer
     *  @return the new SVMBuffer containing the base-e exponentials
     */
    public SVMBuffer exp(){
        SVMBuffer results = new SVMBuffer(info, length);
        SVMBufferSupport.Exp(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Calculates the base-e exponential of this SVMBuffer
     *  @return the new SVMBuffer containing the base-e exponentials
     */
    public SVMBuffer expInPlace(){
        SVMBufferSupport.Exp(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Calculates the absolute value of this SVMBuffer
     *  @return the new SVMBuffer containing the absolute values
     */
    public SVMBuffer abs(){
        SVMBuffer results = new SVMBuffer(info, length);
        SVMBufferSupport.Abs(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Calculates the absolute value of this SVMBuffer
     *  @return the new SVMBuffer containing the absolute values
     */
    public SVMBuffer absInPlace(){
        SVMBufferSupport.Abs(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, this.length);
        return this;
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
        SVMBufferSupport.CopyToArray(info.GetContext(), info.GetCommandQueue(), this.svmBuffer, array);
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
    public SVMBuffer compareGT(float comparee){
        SVMBuffer results = new SVMBuffer(info, length);
        SVMBufferSupport.CompareGT(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, comparee, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Calculates the linear blend of x (this) and y implemented as: x + (y - x) * a
     *  @param comparee y parameter
     *  @param mask a parameter
     *  @return the new SVMBuffer
     */
    public SVMBuffer blend(SVMBuffer comparee, SVMBuffer mask){
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
    public SVMBuffer blendInPlace(SVMBuffer comparee, SVMBuffer mask){
        SVMBufferSupport.Blend(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, comparee.svmBuffer, mask.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Creates a new SVMBuffer with all values from 0 to length
     *  @param info for the gpu
     *  @param length of the SVMBuffer
     *  @return new SVMBuffer
     */
    public static SVMBuffer iota(GPUInformation info, int length){
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
     * Creates the information to use the gpu with opencl
     */
    public static final GPUInformation SPECIES_PREFERRED
        = new OpenCLInformation();
}
