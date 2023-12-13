package jdk.incubator.vector;

import java.util.Arrays;
import jdk.internal.vm.vector.GPUSupport;
import jdk.incubator.vector.GPUSpecies;
import java.util.function.*;

import static jdk.internal.vm.vector.VectorSupport.*;

import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.OpenCLInformation;

/**
    abcd
*/
public class GPUVector {

    private GPUInformation info;

    /**
     *  Abcd
     */
    public float[] array;

    /**
     *  Return array length
     */
    public int length;

    /**
        abc
        @param info abc
        @param array abc
    */
    private GPUVector(GPUInformation info, float[] array) {
        this.info = info;
        this.array = array;
        this.length = array.length;
    }

    /**
        abc
        @param info abc
        @param length abc
    */
    private GPUVector(GPUInformation info, int length) {
        var array = new float[length];
        this.info = info;
        this.array = array;
        this.length = array.length;
    }

    /**
        abc
        @param info asdf
        @param array asdf
        @return asdf
    */
    public static GPUVector fromArray(GPUInformation info, float[] array) {
        return new GPUVector(info, array);
    }

    /**
     *  Fused multiply-add (FMA) of SVMBuffers. This method can be used for
     *  array multiplication.
     *  @param factor the factor used in FMA
     *  @param summand the summand and result used in FMA
     *  @param m amount of columns of this SVMBuffers
     *  @param n amount of columns of the factor GPUVector
     *  @param k the kth column of this GPUVector is gets calculated with the
     *  kth row of the factor GPUVector
     */
    public void matrixFma(GPUVector factor, GPUVector summand, int m, int n, int k){
        GPUSupport.MatrixFma(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, factor.array, summand.array, m, n, k, this.length);
    }

    /**
     *  Fused multiply-add (FMA) of SVMBuffers.
     *  @param factor the factor used in FMA
     *  @param summand the summand and result used in FMA
     *  @return GPUVector of FMA this * factor + summand
     */
    public GPUVector Fma(GPUVector factor, GPUVector summand){
        var result = new GPUVector(info, length);

        GPUSupport.Fma(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, factor.array, summand.array, result.array, summand.length);
        return result;
    }

    /**
     *  Fused multiply-add (FMA) of GPUVectors.
     *  @param factor the factor used in FMA
     *  @param summand the summand and result used in FMA
     *  @param result the result of the FMA
     */
    public void Fma(GPUVector factor, GPUVector summand, GPUVector result){
        GPUSupport.Fma(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, factor.array, summand.array, result.array, summand.length);
    }

    /**
     *  Reduces this GPUVector to a single value using addition
     *  @return the result of the reduction
     */
    public float reduceAdd() {
        return GPUSupport.ReduceAdd(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, this.length);
    }

    /**
     *  Elementwise addition of this GPUVector and the summand GPUVector
     *  @param summand The GPUVector which gets added the this GPUVector
     *  @return a new GPUVector containing the added elements
     */
    public GPUVector Add(GPUVector summand) {
        var results = GPUSupport.Add(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, summand.array, this.length);
        return new GPUVector(info, results);
    }

    /**
     *  abc
     *  @param a abc
     *  @param b abc
     *  @param c abc
     */
    public static void AddHost(int[] a, int [] b, int[] c){
        GPUSupport.gpuAddHostPtr(a,b,c);
    }

    // /**
    //  *  Elementwise addition of this GPUVector and the summand GPUVector
    //  *  @param info of the gpu
    //  *  @param result a new GPUVector containing the added elements
    //  *  @param summand The GPUVector which gets added the this GPUVector
    //  */
    // public static void Add(GPUInformation info, GPUVector result, GPUVector summand) {
    //     GPUSupport.AddSVMBuffer(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), result.array, summand.array, result.array, result.length);
    // }

    /**
     *  Subtracts the subtrahend from this GPUVector
     *  @param subtrahend of the subtraction
     *  @return the result of the subtraction
     */
    public GPUVector Subtract(GPUVector subtrahend){
        var result = new GPUVector(info, length);

        GPUSupport.Subtract(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, subtrahend.array, result.array, this.length);
        return result;
    }

    /**
     *  Subtracts the subtrahend from this GPUVector
     *  @param info of the gpu
     *  @param subtrahend of the subtraction
     *  @param result the result of the subtraction
     */
    public static void Subtract(GPUInformation info, GPUVector result, GPUVector subtrahend){
        GPUSupport.Subtract(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), result.array, subtrahend.array, result.array, result.length);
    }

    /**
     *  Subtracts this GPUVector from the minuend
     *  @param minuend of the subtraction
     *  @return the subtracted GPUVector
     */
    public GPUVector SubtractionMinuend(float minuend){
        var result = new GPUVector(info, length);

        GPUSupport.SubtractionMinuend(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, result.array, minuend, this.length);
        return result;
    }

    /**
     *  Multiplies this GPUVector with the factor
     *  @param factor of the multiplication
     *  @return the multiplied GPUVector
     */
    public GPUVector Multiply(float factor){
        var result = new GPUVector(info, length);

        GPUSupport.Multiply(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, result.array, factor, this.length);

        return result;
    }

    /**
     *  Multiplies this GPUVector with the factors GPUVector
     *  @param factors of the multiplication
     *  @return the multiplied GPUVector
     */
    public GPUVector Multiply(GPUVector factors){
        var result = new GPUVector(info, length);

        GPUSupport.Multiply(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, factors.array, result.array, this.length);

        return result;
    }

    /**
     *  Multiplies this GPUVector with the factors GPUVector
     *  @param info of the gpu
     *  @param result the multiplied GPUVector
     *  @param factors of the multiplication
     */
    public static void Multiply(GPUInformation info, GPUVector result, GPUVector factors){
        GPUSupport.Multiply(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), result.array, factors.array, result.array, result.length);
    }

    /**
     *  Find the square root of this GPUVector
     *  @return the new GPUVector containing the square roots
     */
    public GPUVector Sqrt(){
        var result = new GPUVector(info, length);

        GPUSupport.Sqrt(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, result.array, this.length);

        return result;
    }

    /**
     *  Creates a new GPUVector initialized with @param value
     *  @param info informations for the gpu
     *  @param value of the elements
     *  @param length of the GPUVector
     *  @return initialized GPUVector
     */
    public static GPUVector Broadcast(GPUInformation info, float value, int length) {
        float[] array = new float[length];
        Arrays.fill(array, value);
        return new GPUVector(info, array);
    }

    /**
     *  Divides this GPUVector with the divisor
     *  @param divisor of the division
     *  @return the divided GPUVector
     */
    public GPUVector Division(float divisor){
        var result = new GPUVector(info, length);

        GPUSupport.Division(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, result.array, divisor, this.length);

        return result;
    }

    /**
     *  Divides this GPUVector with the divisor GPUVector
     *  @param divisors of the division
     *  @return the divided GPUVector
     */
    public GPUVector Division(GPUVector divisors){
        var result = new GPUVector(info, length);

        GPUSupport.Division(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, divisors.array, result.array, this.length);

        return result;
    }

    /**
     *  Divides this GPUVector with the divisor GPUVector
     *  @param info of the gpu
     *  @param result the divided GPUVector
     *  @param divisors of the division
     */
    public static void Division(GPUInformation info, GPUVector result, GPUVector divisors){

        GPUSupport.Division(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), result.array, divisors.array, result.array, result.length);
    }

    /**
     *  Calculates the natural logarithm of this GPUVector
     *  @return the new GPUVector containing the natural logarithms
     */
    public GPUVector Log(){
        var result = new GPUVector(info, length);

        GPUSupport.Log(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, result.array, this.length);

        return result;
    }

    /**
     *  Calculates the natural logarithm of this GPUVector
     *  @param info of the gpu
     *  @param result the new GPUVector containing the natural logarithms
     */
    public static void Log(GPUInformation info, GPUVector result){
        GPUSupport.Log(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), result.array, result.array, result.length);
    }

    /**
     *  Calculates the base-e exponential of this GPUVector
     *  @return the new GPUVector containing the base-e exponentials
     */
    public GPUVector Exp(){
        var result = new GPUVector(info, length);

        GPUSupport.Exp(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, result.array, this.length);

        return result;
    }

    /**
     *  Calculates the base-e exponential of this GPUVector
     *  @param info of the gpu
     *  @param result the new GPUVector containing the base-e exponentials
     */
    public static void Exp(GPUInformation info, GPUVector result){
        GPUSupport.Exp(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), result.array, result.array, result.length);
    }

    /**
     *  Calculates the absolute value of this GPUVector
     *  @return the new GPUVector containing the absolute values
     */
    public GPUVector Abs(){
        var result = new GPUVector(info, length);

        GPUSupport.Abs(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, result.array, this.length);

        return result;
    }

    /**
     * abcd
     *  @param comparee abc
     *  @return abcd
     */
    public GPUVector CompareGT(float comparee){
        var result = new GPUVector(info, length);

        GPUSupport.CompareGT(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, comparee, result.array, this.length);

        return result;
    }

    /**
     * abcd
     *  @param comparee abc
     *  @param mask abc
     *  @return abcd
     */
    public GPUVector Blend(GPUVector comparee, GPUVector mask){
        var results = new GPUVector(info, length);

        GPUSupport.Blend(info.GetProgram(), info.GetContext(), info.GetCommandQueue(), this.array, comparee.array, mask.array, results.array, this.length);

        return results;
    }

    // /**
    //     abc
    // */
    // static final class IntGPUSpecies implements GPUSpecies  {

    //     static private int arrayLength = 0;
    //     static private int maxGroupSize = 0;

    //     private IntGPUSpecies() {
    //         IntGPUSpecies.maxGroupSize = GPUSupport.initializeGPU();
    //     }

    //     /**
    //         abc
    //     */
    //     public int loopBound(int length) {
    //         IntGPUSpecies.arrayLength = length;
    //         return length;
    //     }

    //     /**
    //         abc
    //     */
    //     public int length(int i) {
    //         return IntGPUSpecies.arrayLength - i;
    //     }


    //     /**
    //         abc
    //         @return abc
    //     */
    //     public static int GetMaxGroupSize() {
    //         return IntGPUSpecies.maxGroupSize;
    //     }
    // }


    // /*
    //  *  Return a GPUInformation
    //  **/
    // public static final GPUInformation SPECIES_PREFERRED
    //     = new OpenCLInformation();
}
