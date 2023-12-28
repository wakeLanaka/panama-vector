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
     *  abc
     */
    public final Class<?> type;


    private Class<?> getType(Class<?> a, Class<?> b){
        if (a == Float.TYPE || b == Float.TYPE) {
            return Float.TYPE;
        }
        return Integer.TYPE;
    }

    /**
     *  Address to the SVMBuffer
     */
    public long svmBuffer;

    /**
     *  Amount of elements in the SVMBuffer
     */
    public int length;

    private <T> SVMBuffer(GPUInformation info, float[] array) {
        this.info = info;
        this.length = array.length;
        this.type = Float.TYPE;
        this.svmBuffer = SVMBufferSupport.CopyFromArray(info.GetContext(), info.GetCommandQueue(), array);
    }

    private SVMBuffer(GPUInformation info, int[] array) {
        this.info = info;
        this.length = array.length;
        this.type = Integer.TYPE;
        this.svmBuffer = SVMBufferSupport.CopyFromArray(info.GetContext(), info.GetCommandQueue(), array);
    }

    private SVMBuffer(GPUInformation info, int length, Class<?> type) {
        this.info = info;
        this.length = length;
        this.type = type;
        if(type == Float.TYPE){
            this.svmBuffer = SVMBufferSupport.CreateReadWriteFloatSVMBuffer(info.GetContext(), length);
        }else if(type == Integer.TYPE){
            this.svmBuffer = SVMBufferSupport.CreateReadWriteIntSVMBuffer(info.GetContext(), length);
        }

    }

    /**
     *  Fills this buffer with the the array elements
     *  @param array of elements to be copied to SVMBuffer
     *  @return this Filled SVMBuffer
     */
    public SVMBuffer fill(float[] array){
        SVMBufferSupport.Fill(info.GetContext(), info.GetCommandQueue(), this.svmBuffer, array);
        return this;
    }

    /**
     *  Fills this buffer with the the array elements
     *  @param array of elements to be copied to SVMBuffer
     *  @return this Filled SVMBuffer
     */
    public SVMBuffer fill(int[] array){
        SVMBufferSupport.Fill(info.GetContext(), info.GetCommandQueue(), this.svmBuffer, array);
        return this;
    }

    /**
     *  Loads a SVMBuffer from an array of type
     *  @param info for the gpu
     *  @param length of the array
     *  @param type of the array
     *  @return the SVMBuffer loaded from the array
     */
    public static SVMBuffer zero(GPUInformation info, int length, Class<?> type) {
        return new SVMBuffer(info, length, type);
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
     *  Loads a SVMBuffer from an array of type {@code float[]}
     *  @param info for the gpu
     *  @param array the array
     *  @return the SVMBuffer loaded from the array
     */
    public static SVMBuffer fromArray(GPUInformation info, int[] array) {
        return new SVMBuffer(info, array);
    }

    /**
     *  Loads a SVMBuffer with elements from an array of type {@code float[]}
     *  @param info for the gpu
     *  @param array to be copied to the SVMBuffer
     *  @param index of first element to be copied to SVMBuffer
     *  @param amount of elements to be copied to SVMBuffer
     *  @return the SVMBuffer loaded from the array
     */
    public static SVMBuffer fromArray(GPUInformation info, float[] array, int index, int amount) {
        return new SVMBuffer(info, Arrays.copyOfRange(array, index, index + amount));
    }

    /**
     *  Fused multiply-add (FMA) of SVMBuffers. This method can be used for
     *  array multiplication.
     *  @param factor used in FMA
     *  @param summand used in FMA
     *  @param m amount of columns of this SVMBuffers
     *  @param n amount of columns of the factor SVMBuffer
     *  @param k the kth column of this is getting fma with the kth
     *  row of the factor
     *  @return sumand of fma
     */
    public SVMBuffer matrixFma(SVMBuffer factor, SVMBuffer summand, int m, int n, int k) {
        SVMBufferSupport.MatrixFmaSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, summand.svmBuffer, m, n, k, summand.length);
        return summand;
    }

    /**
     *  TODO
     *  @param factor TODO
     *  @param width TODO
     *  @param kernelWidth TODO
     *  @param resultLength TODO
     *  @return sumand of fma
     */
    public SVMBuffer eachAreaFMA(SVMBuffer factor, int width, int kernelWidth, int resultLength) {
        Class<?> resultType = getType(type,factor.type);
        SVMBuffer result = new SVMBuffer(info, resultLength, resultType);
        SVMBufferSupport.EachAreaFMA(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, result.svmBuffer, width, kernelWidth, result.length);
        return result;
    }


    /**
     *  Fused multiply-add (FMA) of SVMBuffers.
     *  @param factor the factor used in FMA
     *  @param summand the summand and result used in FMA
     *  @return SVMBuffer of FMA this * factor + summand
     */
    public SVMBuffer fma(SVMBuffer factor, SVMBuffer summand){
        Class<?> resultType = getType(type,factor.type);
        SVMBuffer result = new SVMBuffer(info, length, resultType);
        SVMBufferSupport.FmaSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, summand.svmBuffer, result.svmBuffer, summand.length);
        return result;
    }

    /**
     *  Reduces this SVMBuffer to a single value using addition
     *  @return the result of the sum reduction
     */
    public float sumReduce(){
        return SVMBufferSupport.SumReduce(info.GetContext(), info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, info.GetMaxWorkGroupSize(), this.length);
    }

    /**
     *  Elementwise addition of this SVMBuffer and the summand SVMBuffer
     *  @param summand The SVMBuffer which gets added the this SVMBuffer
     *  @return a new SVMBuffer containing the added elements
     */
    public SVMBuffer add(SVMBuffer summand) {
        Class<?> resultType = getType(type,summand.type);
        SVMBuffer results = new SVMBuffer(info, this.length, resultType);
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
        Class<?> resultType = getType(type,subtrahend.type);
        SVMBuffer result = new SVMBuffer(info, length, resultType);

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
        SVMBuffer results = new SVMBuffer(info, length, this.type);

        SVMBufferSupport.Subtract(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, subtrahend, this.length);
        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mul(float factor){
        SVMBuffer results = new SVMBuffer(info, length, Float.TYPE);
        switch(this.type.toString()) {
            case "float":
                SVMBufferSupport.MultiplyFFF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, factor, this.length);
                break;
            case "int":
                SVMBufferSupport.MultiplyIFF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, factor, this.length);
                break;
            default: throw new AssertionError("Multiplication of " + this.type + "with float is not supported!");
        }
        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mul(int factor){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        switch(this.type.toString()) {
            case "float":
                SVMBufferSupport.MultiplyFFI(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, factor, this.length);
                break;
            case "int":
                SVMBufferSupport.MultiplyIII(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, factor, this.length);
                break;
            default: throw new AssertionError("Multiplication of " + this.type + "with int is not supported!");
        }
        return results;
    }

    /**
     *  Multiplies an rectangular area of this SVMBuffer with the facotr SVMBuffer
     *  @param factor to be multiplied with this SVMBuffer
     *  @param offset is the top left element of this SVMBuffer of the rectangular area
     *  @param thisWidth width of the matrix of this SVMBuffer
     *  @param factorWidth width of the matrix of the factor SVMBuffer
     *  @return new SVMBuffer with the results. (Length == factor.length)
     */
    public SVMBuffer mulArea(SVMBuffer factor, int offset, int thisWidth, int factorWidth) {
        Class<?> resultType = getType(type,factor.type);
        SVMBuffer results = new SVMBuffer(info, factor.length, resultType);
        SVMBufferSupport.MultiplyArea(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, results.svmBuffer, offset, thisWidth, factorWidth, factor.length);
        return results;
    }

    /**
     *  Multiplies range of this SVMBuffer with range of factor
     *  @param index1 start index of this SVMBuffer
     *  @param factor of the multiplication
     *  @param index2 start index of factor SVMBuffer
     *  @param amount of elements to be multiplied
     *  @return the new SVMBuffer of size amount containing the multiplied elements
     */
    public SVMBuffer mulRange(int index1, SVMBuffer factor, int index2, int amount){
        Class<?> resultType = getType(type,factor.type);
        SVMBuffer results = new SVMBuffer(info, amount, resultType);
        SVMBufferSupport.MultiplyRange(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, index1, factor.svmBuffer, index2, results.svmBuffer, amount);
        return results;
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mulInPlace(float factor){
        if(this.type != Float.TYPE){
            throw new AssertionError("Cannot multiply in place as this type is " + this.type);
        }
        SVMBufferSupport.MultiplyFFF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, factor, this.length);
        return this;
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mul(SVMBuffer factors){
        Class<?> resultType = getType(type,factors.type);
        SVMBuffer results = new SVMBuffer(info, length, resultType);

        SVMBufferSupport.Multiply(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factors.svmBuffer, results.svmBuffer, this.length);

        return results;
    }

    /**
     *  TODO
     *  @param amount TODO
     *  @return TODO
     */
    public SVMBuffer ror(int amount){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.Ror(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, amount, this.length);
        return results;
    }

    /**
     *  TODO
     *  @param amount TODO
     *  @return TODO
     */
    public SVMBuffer rol(int amount){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.Rol(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, amount, this.length);
        return results;
    }

    /**
     *  TODO
     *  @param amount TODO
     *  @return TODO
     */
    public SVMBuffer rolInPlace(int amount){
        SVMBufferSupport.Rol(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, amount, this.length);
        return this;
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mulInt(SVMBuffer factors){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.MultiplyInt(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factors.svmBuffer, results.svmBuffer, this.length);
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
        SVMBuffer results = new SVMBuffer(info, length, Float.TYPE);

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
        Class<?> resultType = getType(type, Float.TYPE);
        SVMBuffer results = new SVMBuffer(info, length, resultType);

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
        Class<?> resultType = getType(type,divisors.type);
        SVMBuffer results = new SVMBuffer(info, length, resultType);

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
        SVMBuffer result = new SVMBuffer(info, length, Float.TYPE);
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
        SVMBuffer results = new SVMBuffer(info, length, Float.TYPE);
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
        SVMBuffer results = new SVMBuffer(info, length, Float.TYPE);
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
        SVMBuffer results = new SVMBuffer(info, length, Float.TYPE);
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
        SVMBuffer results = new SVMBuffer(info, length, this.type);
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
     *  TODO
     *  @param repetition TODO
     *  @return TODO
     */
    public SVMBuffer repeat1(int repetition){
        SVMBuffer results = new SVMBuffer(info, length * repetition, this.type);
        SVMBufferSupport.Repeat1(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, repetition, length);
        return results;
    }

    /**
     *  TODO
     *  @param repetition TODO
     *  @return TODO
     */
    public SVMBuffer repeat2(int repetition){
        SVMBuffer results = new SVMBuffer(info, length * repetition, this.type);
        SVMBufferSupport.Repeat2(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, repetition, length);
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
        SVMBufferSupport.CopyToArray(info.GetContext(), info.GetCommandQueue(), this.svmBuffer, array);
    }

    /**
     *  Deallocates the Memory of this SVMBuffer
     */
    public void releaseSVMBuffer() {
        SVMBufferSupport.ReleaseSVMBuffer(info.GetContext(), info.GetCommandQueue(), this.svmBuffer);
        this.svmBuffer = 0;
        this.length = 0;
    }

    /**
     *  Deallocates the Memory of this SVMBuffer
     */
    public void releaseSVMBufferInt() {
        SVMBufferSupport.ReleaseSVMBufferInt(info.GetContext(), info.GetCommandQueue(), this.svmBuffer);
        this.svmBuffer = 0;
        this.length = 0;
    }


    /**
     *  Compares each value of this svmBuffer if it is greater than the comparee
     *  @param comparee of the greater than comparison
     *  @return the mask of the greater than comparison
     */
    public SVMBuffer compareGT(float comparee){
        SVMBuffer results = new SVMBuffer(info, length, Float.TYPE);
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
        SVMBuffer results = new SVMBuffer(info, length, Float.TYPE);
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
     *  Logical rightshift of this SVMBuffer
     *  @param amount to be shifted
     *  @return this SVMBuffer
     */
    public SVMBuffer ashrInPlace(int amount) {
        SVMBufferSupport.Ashr(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, amount, this.length);
        return this;
    }

    /**
     *  Logical rightshift of this SVMBuffer
     *  @param amount to be shifted
     *  @return this SVMBuffer
     */
    public SVMBuffer ashr(int amount) {
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.Ashr(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, amount, this.length);
        return results;
    }

    /**
     *  Logical leftshift of this SVMBuffer
     *  @param amount to be shifted
     *  @return this SVMBuffer
     */
    public SVMBuffer lshl(int amount) {
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.Lshl(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, amount, this.length);
        return results;
    }

    /**
     *  Logical leftshift of this SVMBuffer
     *  @param amount to be shifted
     *  @return this SVMBuffer
     */
    public SVMBuffer lshlInPlace(int amount) {
        SVMBufferSupport.Lshl(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, amount, this.length);
        return this;
    }

    /**
     *  Bitwise and of this SVMBuffer
     *  @param value for the bitwise and
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer and(int value){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.And(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, value, this.length);
        return results;
    }

    /**
     *  Bitwise or of this SVMBuffer
     *  @param other for the bitwise or
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer orInPlace(SVMBuffer other){
        SVMBufferSupport.Or(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, other.svmBuffer, this.svmBuffer, this.length);
        return this;
    }

    /**
     *  Bitwise or of this SVMBuffer
     *  @param other for the bitwise or
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer or(SVMBuffer other){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.Or(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, other.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Get max values of this SVMBuffer and value
     *  @param value max value
     *  @return the max values
     */
    public SVMBuffer max(float value){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.Max(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, value, this.length);
        return results;
    }

    /**
     *  Get max values of this SVMBuffer and value
     *  @param value max value
     *  @return the max values
     */
    public SVMBuffer maxInPlace(float value){
        SVMBufferSupport.Max(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, value, this.length);
        return this;
    }

    /**
     *  Get max values of this SVMBuffer and value
     *  @param value max value
     *  @return the max values
     */
    public SVMBuffer min(float value){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        SVMBufferSupport.Min(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, value, this.length);
        return results;
    }

    /**
     *  Get max values of this SVMBuffer and value
     *  @param value max value
     *  @return the max values
     */
    public SVMBuffer minInPlace(float value){
        SVMBufferSupport.Min(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, this.svmBuffer, value, this.length);
        return this;
    }

    /**
     *  Casts this SVMBuffer to a Int SVMBuffer
     *  @return the Int SVMBuffer
     */
    public SVMBuffer toInt(){
        SVMBuffer results = new SVMBuffer(info, length, Integer.TYPE);
        SVMBufferSupport.ToInt(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Casts this SVMBuffer to a Float SVMBuffer
     *  @return the Int SVMBuffer
     */
    public SVMBuffer toFloat(){
        SVMBuffer results = new SVMBuffer(info, length, Float.TYPE);
        SVMBufferSupport.ToFloat(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);
        return results;
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
