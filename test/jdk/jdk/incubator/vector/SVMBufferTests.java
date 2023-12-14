/*
 * Copyright (c) 2018, 2022, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

/*
 * @test
 * @modules jdk.incubator.vector
 * @summary Tests the SVMBuffer
 * @run main SVMBufferTests
 */

import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;

import java.lang.Integer;
import java.util.List;
import java.util.Arrays;
import java.util.function.BiFunction;
import java.util.function.IntFunction;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.vector
 * @summary Tests the SVMBuffer
 * @run main SVMBufferTests
 */
public class SVMBufferTests {
    private static final GPUInformation SPECIES = SVMBuffer.SPECIES_PREFERRED;

    private static final float delta = 0.0001f;

    private static int size = 15;

    public static void main(String[] args) throws Exception {
        var a = iotaFloatArray(SVMBufferTests.size);
        var buffer1 = SVMBuffer.fromArray(SPECIES, a);
        var buffer2 = SVMBuffer.fromArray(SPECIES, a);

        SVMBufferTests.IntoArrayFloat(buffer1, a);
        SVMBufferTests.FromArrayFloat(a);
        SVMBufferTests.SumReduceFloat(buffer1, SVMBufferTests.size);
        SVMBufferTests.AddBufferFloat(buffer1, buffer2, a, a);
        SVMBufferTests.SubtractBufferFloat(buffer1, buffer2, a, a);
        SVMBufferTests.SubtractFloat(buffer1, a, 2.0f);
        SVMBufferTests.MultiplyFloat(buffer1, a, 2.0f);
        SVMBufferTests.MultiplyBufferFloat(buffer1, buffer2, a, a);
        SVMBufferTests.DivisionFloat(buffer1, a, 2.0f);
        SVMBufferTests.DivisionBufferFloat(buffer1, buffer2, a, a);
        SVMBufferTests.SqrtFloat(buffer1, a);
        SVMBufferTests.LogFloat(buffer1, a);
        SVMBufferTests.CosFloat(buffer1, a);
        SVMBufferTests.SinFloat(buffer1, a);
        SVMBufferTests.ExpFloat(buffer1, a);
        SVMBufferTests.AbsFloat(buffer1, a);
        SVMBufferTests.BroadcastFloat(2.0f);

        SVMBufferTests.InPlaceMethods();
    }

    private static void InPlaceMethods() throws Exception {
        var b = -55.321f;
        var a = iotaFloatArray(SVMBufferTests.size);
        var buffer1 = SVMBuffer.fromArray(SPECIES, a);
        var buffer2 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.AddBufferInPlaceFloat(buffer1, buffer2, a, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.SubtractInPlaceFloat(buffer1, buffer2, a, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.MultiplyInPlaceFloat(buffer1, a, b);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.MultiplyInPlaceBufferFloat(buffer1, buffer2, a, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.DivisionInPlaceFloat(buffer1, a, b);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.DivisionInPlaceBufferFloat(buffer1, buffer2, a, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.LogInPlaceFloat(buffer1, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.SinInPlaceFloat(buffer1, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.CosInPlaceFloat(buffer1, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.ExpInPlaceFloat(buffer1, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.AbsInPlaceFloat(buffer1, a);
        buffer1.releaseSVMBuffer();
        buffer1 = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.SqrtInPlaceFloat(buffer1, a);
    }

    public static void FromArrayFloat(float[] array) throws Exception {
        var buffer = SVMBuffer.fromArray(SPECIES, array);
        var bufferArray = new float[buffer.length];
        buffer.intoArray(bufferArray);
        AssertFloatArray(bufferArray, array, SVMBufferTests.delta, "FromArrayFloat");
    }

    public static void IntoArrayFloat(SVMBuffer buffer, float[] array) throws Exception {
        var bufferArray = new float[buffer.length];
        buffer.intoArray(bufferArray);
        AssertFloatArray(bufferArray, array, SVMBufferTests.delta, "IntoArrayFloat");
    }

    private static void MultiplyInPlaceBufferFloat(SVMBuffer buffer, SVMBuffer factors, float[] a, float[] b) throws Exception {
        var result = buffer.mulInPlace(factors);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyInPlaceBufferFloat";
        var expectedArray = ArrayMultiplicationFloat(a, b, info);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void MultiplyBufferFloat(SVMBuffer buffer, SVMBuffer factors, float[] a, float[] b) throws Exception {
        var result = buffer.mul(factors);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyBufferFloat";
        var expectedArray = ArrayMultiplicationFloat(a, b, info);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void MultiplyFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.mul(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyFloat";
        var expectedArray = ArrayMultiplicationFloat(a, b);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
    }

    private static void MultiplyInPlaceFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.mulInPlace(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyInPlaceFloat";
        var expectedArray = ArrayMultiplicationFloat(a, b);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void AddBufferFloat(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b) throws Exception {
        var result = buffer.add(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = ArrayAdditionFloat(a, b, info);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void AddBufferInPlaceFloat(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b) throws Exception {
        var result = buffer.addInPlace(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddInPlaceFloat";
        var expectedArray = ArrayAdditionFloat(a, b, info);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void SubtractBufferFloat(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b) throws Exception {
        var result = buffer.sub(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubtractBufferFloat";
        var expectedArray = ArraySubtractionFloat(a, b, info);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void SubtractFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.sub(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubtractFloat";
        var expectedArray = ArraySubtractionFloat(a, b);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
    }

    private static void SubtractInPlaceFloat(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b) throws Exception {
        var result = buffer.subInPlace(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "SubtractInPlaceFloat";
        var expectedArray = ArraySubtractionFloat(a, b, info);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void BroadcastFloat(float value) throws Exception {
        float[] resultArray = new float[SVMBufferTests.size];
        var buffer = SVMBuffer.broadcast(SPECIES, value, SVMBufferTests.size);
        buffer.intoArray(resultArray);
        checkArrayValue(resultArray, value, "BroadcastFloat");
    }

    private static void DivisionInPlaceBufferFloat(SVMBuffer buffer, SVMBuffer factors, float[] a, float[] b) throws Exception {
        var result = buffer.divInPlace(factors);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "DivisionInPlaceBufferFloat";
        var expectedArray = ArrayDivisionFloat(a, b, info);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void DivisionBufferFloat(SVMBuffer buffer, SVMBuffer factors, float[] a, float[] b) throws Exception {
        var result = buffer.div(factors);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "DivisionBufferFloat";
        var expectedArray = ArrayDivisionFloat(a, b, info);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void DivisionFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.div(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "DivisionFloat";
        var expectedArray = ArrayDivisionFloat(a, b);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
    }

    private static void DivisionInPlaceFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.divInPlace(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "DivisionInPlaceFloat";
        var expectedArray = ArrayDivisionFloat(a, b);
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void SqrtFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.sqrt();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArraySqrt(a);
        String info = "SqrtFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void SqrtInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.sqrtInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArraySqrt(a);
        String info = "SqrtInPlaceFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void LogFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.log();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayLog(a);
        String info = "LogFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void LogInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.logInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayLog(a);
        String info = "LogInPlaceFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void CosFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.cos();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayCos(a);
        String info = "CosFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void CosInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.cosInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayCos(a);
        String info = "CosInPlaceFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void ExpFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.exp();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayExp(a);
        System.out.println(resultArray[9]);
        System.out.println(expectedArray[9]);
        String info = "ExpFloat";
        float delta = 0.1f;
        AssertFloatArray(resultArray, expectedArray, delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void ExpInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.expInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayExp(a);
        String info = "ExpInPlaceFloat";
        float delta = 0.1f;
        AssertFloatArray(resultArray, expectedArray, delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void AbsFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.abs();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayAbs(a);
        String info = "AbsFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void AbsInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.absInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayAbs(a);
        String info = "AbsInPlaceFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void SinFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.sin();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArraySin(a);
        String info = "SinFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void SinInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.sinInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArraySin(a);
        String info = "SinInPlaceFloat";
        AssertFloatArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void checkArrayValue(float[] array, float value, String info) throws Exception {
        for (int i = 0; i < array.length; i++){
            if (Math.abs(array[i] - value) > SVMBufferTests.delta) {
                throw new RuntimeException(info + ": Element " + i + " is not < delta!");
            }
        }
    }

    private static void checkBufferAddressEqual(SVMBuffer result, SVMBuffer expected, String info) throws Exception {
        if(result.svmBuffer != expected.svmBuffer){
            throw new RuntimeException(info + ": svmBuffers are not the same!");
        }
    }

    private static void checkBufferAddressUnequal(SVMBuffer result, SVMBuffer expected, String info) throws Exception {
        if(result.svmBuffer == expected.svmBuffer){
            throw new RuntimeException(info + ": svmBuffers are the same!");
        }
    }

    private static float[] ArraySqrt(float[] a) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)Math.sqrt(a[i]);
        }
        return c;
    }

    private static float[] ArrayLog(float[] a) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)Math.log(a[i]);
        }
        return c;
    }

    private static float[] ArrayExp(float[] a) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)Math.exp(a[i]);
        }
        return c;
    }

    private static float[] ArrayAbs(float[] a) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)Math.abs(a[i]);
        }
        return c;
    }

    private static float[] ArraySin(float[] a) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)Math.sin(a[i]);
        }
        return c;
    }

    private static float[] ArrayCos(float[] a) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)Math.cos(a[i]);
        }
        return c;
    }

    private static float[] ArrayAdditionFloat(float[] a, float[] b, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] + b[i];
        }
        return c;
    }

    private static float[] ArraySubtractionFloat(float[] a, float[] b, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] - b[i];
        }
        return c;
    }

    private static float[] ArrayDivisionFloat(float[] a, float b) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] / b;
        }
        return c;
    }

    private static float[] ArrayDivisionFloat(float[] a, float[] b, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] / b[i];
        }
        return c;
    }

    private static float[] ArrayMultiplicationFloat(float[] a, float b) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] * b;
        }
        return c;
    }

    private static float[] ArrayMultiplicationFloat(float[] a, float[] b, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] * b[i];
        }
        return c;
    }

    private static void checkArrayLength(float[] a, float[] b, String info) throws Exception {
        if (a.length != b.length) {
            throw new RuntimeException(info + ": not same array length!");
        }
    }

    private static float[] ArraySubtractionFloat(float[] a, float b) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] - b;
        }
        return c;
    }

    private static void SumReduceFloat(SVMBuffer buffer, int size) throws Exception {
        var result = buffer.sumReduce();
        var resultExpected = (size + 1) * (size/2.0f) - size;
        if (Math.abs(result - resultExpected) > SVMBufferTests.delta) {
            throw new RuntimeException("SumReduce: is " + result + ", expected: " + resultExpected);
        }
    }

    private static float[] iotaFloatArray(int size){
        var array = new float[size]; 
        for (int i = 0; i < size; i++){
            array[i] = (float)i;
        }
        return array;
    }

    private static void AssertFloatArray(float[] f1, float[] f2, float delta, String info) throws Exception {
        if (f1.length != f2.length) {
            throw new RuntimeException(info = "Float Array's do not have the same length!");
        }
        int n = f1.length;

        for (int i = 0; i < n; i++) {
            if (Math.abs(f1[i] - f2[i]) > delta) {
                throw new RuntimeException(info + ": Element " + i + " is not < delta!");
            }
        }
    }
}
