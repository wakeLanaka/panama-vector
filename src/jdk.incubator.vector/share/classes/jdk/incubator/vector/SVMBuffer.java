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
     *  Type of the SVMbuffer
     */
    public final Type type;

    /**
     *  Address to the SVMBuffer
     */
    public long svmBuffer;

    /**
     *  Amount of elements in the SVMBuffer
     */
    public int length;

    private SVMBuffer(GPUInformation info, float[] array) {
        this.info = info;
        this.length = array.length;
        this.type = Type.FLOAT;
        this.svmBuffer = SVMBufferSupport.CopyFromArray(info.GetContext(), info.GetCommandQueue(), array);
    }

    private SVMBuffer(GPUInformation info, int[] array) {
        this.info = info;
        this.length = array.length;
        this.type = Type.INT;
        this.svmBuffer = SVMBufferSupport.CopyFromArray(info.GetContext(), info.GetCommandQueue(), array);
    }

    private SVMBuffer(GPUInformation info, int length, Type type) {
        this.info = info;
        this.length = length;
        this.type = type;
        if(type == Type.FLOAT){
            this.svmBuffer = SVMBufferSupport.CreateReadWriteFloatSVMBuffer(info.GetContext(), length);
        } else if(type == Type.INT){
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
    public static SVMBuffer zero(GPUInformation info, int length, Type type) {
        return new SVMBuffer(info, length, type);
    }

    /**
     *  Loads a SVMBuffer from an array 
     *  @param info for the gpu
     *  @param array the array
     *  @return the SVMBuffer loaded from the array
     */
    public static SVMBuffer fromArray(GPUInformation info, float[] array) {
        return new SVMBuffer(info, array);
    }

    /**
     *  Loads a SVMBuffer from an array of type {@code int[]}
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
     *  Calculates the FMA for each element of this SVMBuffer with the factor
     *  @param factor for the fma
     *  @param width of the matrix of this SVMBuffer
     *  @param kernelWidth of the factor
     *  @param resultLength of the resulting fma
     *  @return the new SVMBuffer
     */
    public SVMBuffer eachAreaFMA(SVMBuffer factor, int width, int kernelWidth, int resultLength) {
        Type resultType = type.resultOf(factor.type);
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
        Type resultType = type.resultOf(factor.type);
        SVMBuffer result = new SVMBuffer(info, length, resultType);
        SVMBufferSupport.FmaSVMBuffer(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, factor.svmBuffer, summand.svmBuffer, result.svmBuffer, summand.length);
        return result;
    }

    /**
     *  Reduces this SVMBuffer to a single value using addition
     *  @return the result of the sum reduction
     */
    public float sumReduceFloat() {
        return SVMBufferSupport.SumReduceFLOAT(info.GetContext(), info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, info.GetMaxWorkGroupSize(), this.length);
    }

    /**
     *  Reduces this SVMBuffer to a single value using addition
     *  @return the result of the sum reduction
     */
    public int sumReduceInt() {
        return SVMBufferSupport.SumReduceINT(info.GetContext(), info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, info.GetMaxWorkGroupSize(), this.length);
    }


    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mul(int factor){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(factor, results, "mul");
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mulInPlace(int factor){
        return executeOperationWithArguments(factor, this, "mul");
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mul(float factor){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(factor, results, "mul");
    }

    /**
     *  Multiplies this SVMBuffer with the factor
     *  @param factor of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mulInPlace(float factor){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Cannot change the type of this SVMBuffer from " + this.type + " to " + Type.FLOAT);
        }
        return executeOperationWithArguments(factor, this, "mul");
    }

    /**
     *  Adds value to this SVMBuffer
     *  @param value of the addition
     *  @return the added SVMBuffer
     */
    public SVMBuffer add(int value){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "add");
    }

    /**
     *  Adds value to this SVMBuffer
     *  @param value of the addition
     *  @return the added SVMBuffer
     */
    public SVMBuffer addInPlace(int value){
        return executeOperationWithArguments(value, this, "add");
    }

    /**
     *  Adds value to this SVMBuffer
     *  @param value of the addition
     *  @return the added SVMBuffer
     */
    public SVMBuffer add(float value){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(value, results, "add");
    }

    /**
     *  Adds value to this SVMBuffer
     *  @param value of the addition
     *  @return the added SVMBuffer
     */
    public SVMBuffer addInPlace(float value){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Cannot change the type of this SVMBuffer from " + this.type + " to " + Type.FLOAT);
        }
        return executeOperationWithArguments(value, this, "add");
    }

    /**
     *  subtracts the value from this SVMBuffer
     *  @param value subtrahend of the subtraction
     *  @return the subtracted SVMBuffer
     */
    public SVMBuffer sub(int value){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "sub");
    }

    /**
     *  subtracts the value from this SVMBuffer
     *  @param value subtrahend of the subtraction
     *  @return the subtracted SVMBuffer
     */
    public SVMBuffer subInPlace(int value){
        return executeOperationWithArguments(value, this, "sub");
    }

    /**
     *  subtracts the value from this SVMBuffer
     *  @param value subtrahend of the subtraction
     *  @return the subtracted SVMBuffer
     */
    public SVMBuffer sub(float value){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(value, results, "sub");
    }

    /**
     *  subtracts the value from this SVMBuffer
     *  @param value subtrahend of the subtraction
     *  @return the subtracted SVMBuffer
     */
    public SVMBuffer subInPlace(float value){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Cannot change the type of this SVMBuffer from " + this.type + " to " + Type.FLOAT);
        }
        return executeOperationWithArguments(value, this, "sub");
    }

    /**
     *  Divides the value from this SVMBuffer
     *  @param value divisor of the division
     *  @return the division SVMBuffer
     */
    public SVMBuffer div(int value){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "div");
    }

    /**
     *  Divides the value from this SVMBuffer
     *  @param value divisor of the division
     *  @return the division SVMBuffer
     */
    public SVMBuffer divInPlace(int value){
        return executeOperationWithArguments(value, this, "div");
    }

    /**
     *  Divides the value from this SVMBuffer
     *  @param value divisor of the division
     *  @return the division SVMBuffer
     */
    public SVMBuffer div(float value){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(value, results, "div");
    }

    /**
     *  Divides the value from this SVMBuffer
     *  @param value divisor of the division
     *  @return the division SVMBuffer
     */
    public SVMBuffer divInPlace(float value){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Cannot change the type of this SVMBuffer from " + this.type + " to " + Type.FLOAT);
        }
        return executeOperationWithArguments(value, this, "div");
    }



    /**
     *  Multiplies the values by this SVMBuffer
     *  @param values of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mul(SVMBuffer values){
        Type resultType = type.resultOf(values.type);
        SVMBuffer results = new SVMBuffer(info, length, resultType);
        return executeOperationWithArguments(values, results, "mul");
    }

    /**
     *  Multiplies the values by this SVMBuffer
     *  @param values of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer mulInPlace(SVMBuffer values){
        Type resultType = type.resultOf(values.type);
        if(this.type != resultType){
            throw new AssertionError("Cannot change the type of this SVMBuffer from " + this.type + " to " + resultType);
        }
        return executeOperationWithArguments(values, this, "mul");
    }

    /**
     *  Multiplies this SVMBuffer with the vector. Used if this matrix represents a matrix
     *  @param vector of the multiplication with the matrix
     *  @return the new matrix resulting from the multiplication
     */
    public SVMBuffer mulVector(SVMBuffer vector){
        Type resultType = type.resultOf(vector.type);
        SVMBuffer results = new SVMBuffer(info, vector.length, resultType);
        return executeOperationWithArguments(vector, results, "mulVector");
    }

    /**
     *  Adds the values to this SVMBuffer
     *  @param values of the addition
     *  @return the added SVMBuffer
     */
    public SVMBuffer add(SVMBuffer values){
        Type resultType = type.resultOf(values.type);
        SVMBuffer results = new SVMBuffer(info, length, resultType);
        return executeOperationWithArguments(values, results, "add");
    }

    /**
     *  Adds the values to this SVMBuffer
     *  @param values of the addition
     *  @return the added SVMBuffer
     */
    public SVMBuffer addInPlace(SVMBuffer values){
        Type resultType = type.resultOf(values.type);
        if(this.type != resultType){
            throw new AssertionError("Cannot change the type of this SVMBuffer from " + this.type + " to " + resultType);
        }
        return executeOperationWithArguments(values, this, "add");
    }

    /**
     *  subtracts the values SVMBuffer from this SVMBuffer
     *  @param values of the subtraction
     *  @return the subtracted SVMBuffer
     */
    public SVMBuffer sub(SVMBuffer values){
        Type resultType = type.resultOf(values.type);
        SVMBuffer results = new SVMBuffer(info, length, resultType);
        return executeOperationWithArguments(values, results, "sub");
    }

    /**
     *  subtracts the values SVMBuffer from this SVMBuffer
     *  @param values of the subtraction
     *  @return the subtracted SVMBuffer
     */
    public SVMBuffer subInPlace(SVMBuffer values){
        Type resultType = type.resultOf(values.type);
        if(this.type != resultType){
            throw new AssertionError("Cannot change the type of this SVMBuffer from " + this.type + " to " + resultType);
        }
        return executeOperationWithArguments(values, this, "sub");
    }

    /**
     *  divides the values SVMBuffer from this SVMBuffer
     *  @param values of the divisor
     *  @return the divided SVMBuffer
     */
    public SVMBuffer div(SVMBuffer values){
        Type resultType = type.resultOf(values.type);
        SVMBuffer results = new SVMBuffer(info, length, resultType);
        return executeOperationWithArguments(values, results, "div");
    }

    /**
     *  divides the values SVMBuffer from this SVMBuffer
     *  @param values of the divisor
     *  @return the divided SVMBuffer
     */
    public SVMBuffer divInPlace(SVMBuffer values){
        Type resultType = type.resultOf(values.type);
        if(this.type != resultType){
            throw new AssertionError("Cannot change the type of this SVMBuffer from " + this.type + " to " + resultType);
        }
        return executeOperationWithArguments(values, this, "div");
    }

    // TODO Improve this for Matrix multiplication
    /**
     *  Multiplies range of this SVMBuffer with range of factor
     *  @param index1 start index of this SVMBuffer
     *  @param factor of the multiplication
     *  @param index2 start index of factor SVMBuffer
     *  @param amount of elements to be multiplied
     *  @return the new SVMBuffer of size amount containing the multiplied elements
     */
    public SVMBuffer mulRange(int index1, SVMBuffer factor, int index2, int amount){
        Type resultType = type.resultOf(factor.type);
        SVMBuffer results = new SVMBuffer(info, amount, resultType);
        SVMBufferSupport.MultiplyRange(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, index1, factor.svmBuffer, index2, results.svmBuffer, amount);
        return results;
    }

    /**
     *  Rotates this SVMBuffer to the right
     *  @param amount to be rotated
     *  @return the rotated SVMBuffer
     */
    public SVMBuffer ror(int amount){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(amount, results, "ror");
    }

    /**
     *  Rotates this SVMBuffer to the right
     *  @param amount to be rotated
     *  @return the rotated SVMBuffer
     */
    public SVMBuffer rorInPlace(int amount){
        return executeOperationWithArguments(amount, this, "ror");
    }

    /**
     *  Rotates this SVMBuffer to the left
     *  @param amount to be rotated
     *  @return the rotated SVMBuffer
     */
    public SVMBuffer rol(int amount){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(amount, results, "rol");
    }

    /**
     *  Rotates this SVMBuffer to the left
     *  @param amount to be rotated
     *  @return the rotated SVMBuffer
     */
    public SVMBuffer rolInPlace(int amount){
        return executeOperationWithArguments(amount, this, "rol");
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer MultiplyRepeat(SVMBuffer factors){
        Type resultType = type.resultOf(factors.type);
        SVMBuffer results = new SVMBuffer(info, length, resultType);
        return executeOperationWithArguments(factors, results, factors.length, "multiplyRepeat");
    }

    /**
     *  Multiplies this SVMBuffer with the factors SVMBuffer
     *  @param factors of the multiplication
     *  @return the multiplied SVMBuffer
     */
    public SVMBuffer MultiplyRepeatInPlace(SVMBuffer factors){
        return executeOperationWithArguments(factors, this, factors.length, "multiplyRepeat");
    }

    /**
     *  Calculates the square root of each element of SVMBuffer
     *  @return the new SVMBuffer containing the square roots
     */
    public SVMBuffer sqrt(){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(results, "sqrt");
    }

    /**
     *  Find the square root of this SVMBuffer
     *  @return the new SVMBuffer containing the square roots
     */
    public SVMBuffer sqrtInPlace(){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        return executeOperationWithArguments(this, "sqrt");
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
     *  Creates a new SVMBuffer initialized with @param value
     *  @param info for the gpu
     *  @param value of the elements
     *  @param length of the SVMBuffer
     *  @return initialized SVMBuffer
     */
    public static SVMBuffer broadcast(GPUInformation info, int value, int length) {
        int[] array = new int[length];
        Arrays.fill(array, value);
        return new SVMBuffer(info, array);
    }

    /**
     *  Calculates the natural logarithm of this SVMBuffer
     *  @return the new SVMBuffer containing the natural logarithms
     */
    public SVMBuffer log(){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(results, "log");
    }

    /**
     *  Calculates the natural logarithm of this SVMBuffer
     *  @return this SVMBuffer containing the natural logarithms
     */
    public SVMBuffer logInPlace(){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        return executeOperationWithArguments(this, "log");
    }

    /**
     *  Calculates the cos
     *  @return the new SVMBuffer after applying cos
     */
    public SVMBuffer cos(){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(results, "cos");
    }

    /**
     *  Calculates the cos
     *  @return this SVMBuffer after applying cos
     */
    public SVMBuffer cosInPlace(){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        return executeOperationWithArguments(this, "cos");
    }

    /**
     *  Calculates the sin
     *  @return the new SVMBuffer containing the sin
     */
    public SVMBuffer sin(){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(results, "sin");
    }

    /**
     *  Calculates the sin
     *  @return this SVMBuffer containing the sin
     */
    public SVMBuffer sinInPlace(){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        return executeOperationWithArguments(this, "sin");
    }

    /**
     *  Calculates the base-e exponential of this SVMBuffer
     *  @return the new SVMBuffer containing the base-e exponentials
     */
    public SVMBuffer exp(){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(results, "exp");
    }

    /**
     *  Calculates the base-e exponential of this SVMBuffer
     *  @return the new SVMBuffer containing the base-e exponentials
     */
    public SVMBuffer expInPlace(){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        return executeOperationWithArguments(this, "exp");
    }

    /**
     *  Calculates the absolute value of this SVMBuffer
     *  @return the new SVMBuffer containing the absolute values
     */
    public SVMBuffer abs(){
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(results, "abs");
    }

    /**
     *  Calculates the absolute value of this SVMBuffer
     *  @return the new SVMBuffer containing the absolute values
     */
    public SVMBuffer absInPlace(){
        return executeOperationWithArguments(this, "abs");
    }

    /**
     *  Creates a new SVMBuffer which repeats this full buffer repetition times
     *  @param repetition the amount of repetitions
     *  @return the new SVMBuffer
     */
    public SVMBuffer repeatFullBuffer(int repetition){
        SVMBuffer results = new SVMBuffer(info, length * repetition, this.type);
        return executeOperationWithArguments(repetition, results, "repeatFull", this.length);
    }

    /**
     *  Creates a new SVMBuffer which repeats each number of this buffer repetition times
     *  @param repetition the amount of repetitions
     *  @return the new SVMBuffer
     */
    public SVMBuffer repeatEachNumber(int repetition){
        SVMBuffer results = new SVMBuffer(info, length * repetition, this.type);
        return executeOperationWithArguments(repetition, results, "repeatEachNumber", this.length);
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
     *  Stores this SVMBuffer into an Array of type {@code float[]}
     *  @param array the array of type {@code float[]}
     *  @param length to be moved to the array
     *  @param offset of the first element
     */
    public void intoArray(float[] array,  int offset, int length) {
        SVMBufferSupport.CopyToArray(info.GetContext(), info.GetCommandQueue(), this.svmBuffer, array, this.length, length, offset);
    }

    /**
     *  Stores this SVMBuffer into an Array of type {@code int[]}
     *  @param array the array of type {@code int[]}
     *  @param length to be moved to the array
     *  @param offset of the first element
     */
    public void intoArray(int[] array,  int offset, int length) {
        SVMBufferSupport.CopyToArray(info.GetContext(), info.GetCommandQueue(), this.svmBuffer, array, this.length, length, offset);
    }

    /**
     *  Deallocates the Memory of this SVMBuffer
     */
    public void releaseSVMBuffer() {
        if (type == Type.FLOAT){
            SVMBufferSupport.ReleaseSVMBuffer(info.GetContext(), info.GetCommandQueue(), this.svmBuffer);
        } else {
            SVMBufferSupport.ReleaseSVMBufferInt(info.GetContext(), info.GetCommandQueue(), this.svmBuffer);
        }
        this.svmBuffer = 0;
        this.length = 0;
    }

    /**
     *  Compares each value of this svmBuffer if it is greater than the comparee
     *  @param comparee of the greater than comparison
     *  @return the mask of the greater than comparison
     */
    public SVMBuffer compareGT(float comparee){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(comparee, results, "compareGT");
    }

    /**
     *  Compares each value of this svmBuffer if it is greater than the comparee
     *  @param comparee of the greater than comparison
     *  @return the mask of the greater than comparison
     */
    public SVMBuffer compareGT(int comparee){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
        return executeOperationWithArguments(comparee, results, "compareGT");
    }

    /**
     *  Calculates the linear blend of x (this) and y implemented as: x + (y - x) * a
     *  @param comparee y parameter
     *  @param mask a parameter
     *  @return the new SVMBuffer
     */
    public SVMBuffer blend(SVMBuffer comparee, SVMBuffer mask){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
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
    public SVMBuffer ashr(int amount) {
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(amount, results, "ashr");
    }

    /**
     *  Logical rightshift of this SVMBuffer
     *  @param amount to be shifted
     *  @return this SVMBuffer
     */
    public SVMBuffer ashrInPlace(int amount) {
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        return executeOperationWithArguments(amount, this, "ashr");
    }

    /**
     *  Logical leftshift of this SVMBuffer
     *  @param amount to be shifted
     *  @return this SVMBuffer
     */
    public SVMBuffer lshl(int amount) {
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(amount, results, "lshl");
    }

    /**
     *  Logical leftshift of this SVMBuffer
     *  @param amount to be shifted
     *  @return this SVMBuffer
     */
    public SVMBuffer lshlInPlace(int amount) {
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        return executeOperationWithArguments(amount, this, "lshl");
    }

    /**
     *  Bitwise and of this SVMBuffer
     *  @param value for the bitwise and
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer and(int value){
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "and");
    }

    /**
     *  Bitwise and of this SVMBuffer
     *  @param value for the bitwise and
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer andInPlace(int value){
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        return executeOperationWithArguments(value, this, "and");
    }

    /**
     *  Bitwise and of this SVMBuffer
     *  @param value for the bitwise and
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer and(SVMBuffer value){
        if(this.type != Type.INT || value.type != Type.INT ){
            throw new AssertionError("Type of this SVMBuffer or the argument is not " + Type.INT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "and");
    }

    /**
     *  Bitwise and of this SVMBuffer
     *  @param value for the bitwise and
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer andInPlace(SVMBuffer value){
        if(this.type != Type.INT || value.type != Type.INT ){
            throw new AssertionError("Type of this SVMBuffer or the argument is not " + Type.INT);
        }
        return executeOperationWithArguments(value, this, "and");
    }

    /**
     *  Bitwise and of this SVMBuffer
     *  @param value for the bitwise and
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer or(int value){
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "or");
    }

    /**
     *  Bitwise and of this SVMBuffer
     *  @param value for the bitwise and
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer orInPlace(int value){
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        return executeOperationWithArguments(value, this, "or");
    }

    /**
     *  Bitwise or of this SVMBuffer
     *  @param value for the bitwise or
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer or(SVMBuffer value){
        if(this.type != Type.INT || value.type != Type.INT ){
            throw new AssertionError("Type of this SVMBuffer or the argument is not " + Type.INT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "or");
    }

    /**
     *  Bitwise or of this SVMBuffer
     *  @param value for the bitwise or
     *  @return the resulting SVMBuffer
     */
    public SVMBuffer orInPlace(SVMBuffer value){
        if(this.type != Type.INT || value.type != Type.INT ){
            throw new AssertionError("Type of this SVMBuffer or the argument is not " + Type.INT);
        }
        return executeOperationWithArguments(value, this, "or");
    }

    /**
     *  Get max values of this SVMBuffer and value
     *  @param value max value
     *  @return the max values
     */
    public SVMBuffer max(float value){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "max");
    }

    /**
     *  Get max values of this SVMBuffer and value
     *  @param value max value
     *  @return the max values
     */
    public SVMBuffer maxInPlace(float value){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        return executeOperationWithArguments(value, this, "max");
    }

    /**
     *  Get max values of this SVMBuffer and value
     *  @param value max value
     *  @return the max values
     */
    public SVMBuffer max(int value){
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "max");
    }

    /**
     *  Get max values of this SVMBuffer and value
     *  @param value max value
     *  @return the max values
     */
    public SVMBuffer maxInPlace(int value){
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        return executeOperationWithArguments(value, this, "max");
    }

    /**
     *  Get min values of this SVMBuffer and value
     *  @param value min value
     *  @return the min values
     */
    public SVMBuffer min(float value){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "min");
    }

    /**
     *  Get min values of this SVMBuffer and value
     *  @param value min value
     *  @return the min values
     */
    public SVMBuffer minInPlace(float value){
        if(this.type != Type.FLOAT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.FLOAT);
        }
        return executeOperationWithArguments(value, this, "min");
    }

    /**
     *  Get min values of this SVMBuffer and value
     *  @param value min value
     *  @return the min values
     */
    public SVMBuffer min(int value){
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        SVMBuffer results = new SVMBuffer(info, length, this.type);
        return executeOperationWithArguments(value, results, "min");
    }

    /**
     *  Get min values of this SVMBuffer and value
     *  @param value min value
     *  @return the min values
     */
    public SVMBuffer minInPlace(int value){
        if(this.type != Type.INT){
            throw new AssertionError("Type of this SVMBuffer is not " + Type.INT);
        }
        return executeOperationWithArguments(value, this, "min");
    }

    /**
     *  Casts this SVMBuffer to a Int SVMBuffer
     *  @return the Int SVMBuffer
     */
    public SVMBuffer toInt(){
        SVMBuffer results = new SVMBuffer(info, length, Type.INT);
        SVMBufferSupport.ToInt(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, this.length);
        return results;
    }

    /**
     *  Casts this SVMBuffer to a Float SVMBuffer
     *  @return the Int SVMBuffer
     */
    public SVMBuffer toFloat(){
        SVMBuffer results = new SVMBuffer(info, length, Type.FLOAT);
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

    private <T extends Number> SVMBuffer executeOperationWithArguments(T value, SVMBuffer results, String kernelName){
        return executeOperationWithArguments(value, results, kernelName, results.length);
    }

    private <T extends Number> SVMBuffer executeOperationWithArguments(T value, SVMBuffer results, String kernelName, int globalSize){
        var primitiveType = Type.fromNumber(value);
        kernelName += this.type.toString() + primitiveType.toString() + results.type.toString();
        // System.out.println(kernelName);
        switch(this.type) {
            case Type.FLOAT:
                if (primitiveType == Type.FLOAT) {
                    SVMBufferSupport.executeKernelWithTypesFFF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, (float)value, results.svmBuffer, globalSize, kernelName);
                } else {
                    SVMBufferSupport.executeKernelWithTypesFIF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, (int)value, results.svmBuffer, globalSize, kernelName);
                }
                break;
            case Type.INT:
                if (primitiveType == Type.FLOAT) {
                    SVMBufferSupport.executeKernelWithTypesIFF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, (float)value, results.svmBuffer, globalSize, kernelName);
                } else {
                    SVMBufferSupport.executeKernelWithTypesIII(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, (int)value, results.svmBuffer, globalSize, kernelName);
                }
                break;
            default: throw new AssertionError("Invalid operation type");
        }
        return results;
    }

    private SVMBuffer executeOperationWithArguments(SVMBuffer results, String kernelName){
        return executeOperationWithArguments(results, kernelName, results.length);
    }

    private SVMBuffer executeOperationWithArguments(SVMBuffer results, String kernelName, int globalSize){
        kernelName += "Buffer" + this.type.toString() + results.type.toString();
        // System.out.println(kernelName);
        if(this.type == Type.FLOAT && results.type == Type.FLOAT){
            SVMBufferSupport.executeKernelWithTypesBufferFF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, globalSize, kernelName);
        } else if(this.type == Type.FLOAT && results.type == Type.INT){
            SVMBufferSupport.executeKernelWithTypesBufferFI(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, globalSize, kernelName);
        } else if (this.type == Type.INT && results.type == Type.FLOAT){
            SVMBufferSupport.executeKernelWithTypesBufferIF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, globalSize, kernelName);
        } else if(this.type == Type.INT && results.type == Type.INT){
            SVMBufferSupport.executeKernelWithTypesBufferII(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, results.svmBuffer, globalSize, kernelName);
        } else {
            throw new IllegalArgumentException("Invalid operation types");
        }
        return results;
    }

    private SVMBuffer executeOperationWithArguments(SVMBuffer values, SVMBuffer results, int number, String kernelName){
        return executeOperationWithArguments(values, results, number, kernelName, results.length);
    }

    private SVMBuffer executeOperationWithArguments(SVMBuffer values, SVMBuffer results, int number, String kernelName, int globalSize){
        kernelName += "Buffer" + this.type.toString() + values.type.toString() + results.type.toString();
        // System.out.println(kernelName);
        if(this.type == Type.FLOAT && values.type == Type.FLOAT){
            SVMBufferSupport.executeKernelWithTypesBufferFFFI(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, values.svmBuffer, results.svmBuffer, number, globalSize, kernelName);
        } else if(this.type == Type.FLOAT && values.type == Type.INT){
            SVMBufferSupport.executeKernelWithTypesBufferFIFI(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, values.svmBuffer, results.svmBuffer, number, globalSize, kernelName);
        } else if (this.type == Type.INT && values.type == Type.FLOAT){
            SVMBufferSupport.executeKernelWithTypesBufferIFFI(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, values.svmBuffer, results.svmBuffer, number, globalSize, kernelName);
        } else if(this.type == Type.INT && values.type == Type.INT){
            SVMBufferSupport.executeKernelWithTypesBufferIIII(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, values.svmBuffer, results.svmBuffer, number, globalSize, kernelName);
        } else {
            throw new IllegalArgumentException("Invalid operation types");
        }
        return results;
    }

    private SVMBuffer executeOperationWithArguments(SVMBuffer values, SVMBuffer results, String kernelName){
        return executeOperationWithArguments(values, results, kernelName, results.length);
    }

    private SVMBuffer executeOperationWithArguments(SVMBuffer values, SVMBuffer results, String kernelName, int globalSize){
        kernelName += "Buffer" + this.type.toString() + values.type.toString() + results.type.toString();
        // System.out.println(kernelName);
        if(this.type == Type.FLOAT && values.type == Type.FLOAT){
            SVMBufferSupport.executeKernelWithTypesBufferFFF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, values.svmBuffer, results.svmBuffer, globalSize, kernelName);
        } else if(this.type == Type.FLOAT && values.type == Type.INT){
            SVMBufferSupport.executeKernelWithTypesBufferFIF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, values.svmBuffer, results.svmBuffer, globalSize, kernelName);
        } else if (this.type == Type.INT && values.type == Type.FLOAT){
            SVMBufferSupport.executeKernelWithTypesBufferIFF(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, values.svmBuffer, results.svmBuffer, globalSize, kernelName);
        } else if(this.type == Type.INT && values.type == Type.INT){
            SVMBufferSupport.executeKernelWithTypesBufferIII(info.GetProgram(), info.GetCommandQueue(), this.svmBuffer, values.svmBuffer, results.svmBuffer, globalSize, kernelName);
        } else {
            throw new IllegalArgumentException("Invalid operation types");
        }
        return results;
    }


    /**
     *  Types a SVMBuffer can take
     */
    public enum Type {
        /**
         *  Float type
         */
        FLOAT(Float.TYPE),
        /**
         *  Integer type
         */
        INT(Integer.TYPE);

        private Class<?> type;

        private Type(Class<?> type){
            this.type = type;
        }

        private Class<?> getType(){
            return this.type;
        }

        private Type resultOf(Type other){
            if(this.type == Float.TYPE || other.getType() == Float.TYPE){
                return Type.FLOAT;
            }
            return Type.INT;
        }

        private static <T extends Number> Type fromNumber(T value){
            if(Float.class.isInstance(value)){
                return Type.FLOAT;
            }
            return Type.INT;
        }
    }

    /**
     * Creates the information to use the gpu with opencl
     */
    public static final GPUInformation SPECIES_PREFERRED
        = new OpenCLInformation();
}
