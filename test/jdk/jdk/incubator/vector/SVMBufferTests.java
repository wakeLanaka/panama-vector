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

    private static int size = 8022;

    public static void main(String[] args) throws Exception {
        var floatIota = iotaFloatArray(size);
        var intIota = iotaIntArray(size);
        var one = ones(size);
        var bufferOne = SVMBuffer.fromArray(SPECIES, one);
        var bufferAInt = SVMBuffer.fromArray(SPECIES, intIota);
        var bufferBInt = SVMBuffer.fromArray(SPECIES, intIota);
        var bufferA = SVMBuffer.fromArray(SPECIES, floatIota);
        var bufferB = SVMBuffer.fromArray(SPECIES, floatIota);

        SVMBufferTests.IntoArrayFloat(bufferA, floatIota);
        SVMBufferTests.FromArrayFloat(floatIota);
        SVMBufferTests.SumReduceFloat(bufferOne);
        SVMBufferTests.AddBufferFloat(bufferA, bufferB, floatIota, floatIota);
        SVMBufferTests.SubtractBufferFloat(bufferA, bufferB, floatIota, floatIota);
        SVMBufferTests.SubtractFloat(bufferA, floatIota, 2.0f);
        SVMBufferTests.MultiplyFFF(bufferA, floatIota, 2.0f);
        SVMBufferTests.MultiplyFIF(bufferA, floatIota, 1);
        SVMBufferTests.MultiplyIII(bufferAInt, intIota, 1);
        SVMBufferTests.MultiplyBufferFloat(bufferA, bufferB, floatIota, floatIota);
        SVMBufferTests.DivisionFloat(bufferA, floatIota, 2.0f);
        SVMBufferTests.DivisionBufferFloat(bufferA, bufferB, floatIota, floatIota);
        SVMBufferTests.SqrtFloat(bufferA, floatIota);
        SVMBufferTests.LogFloat(bufferA, floatIota);
        SVMBufferTests.CosFloat(bufferA, floatIota);
        SVMBufferTests.SinFloat(bufferA, floatIota);
        SVMBufferTests.ExpFloat(bufferOne, one);
        SVMBufferTests.AbsFloat(bufferA, floatIota);
        SVMBufferTests.BroadcastFloat(2.0f);

        SVMBufferTests.InPlaceMethods(bufferA, bufferB, floatIota, bufferOne, one);
    }

    private static void InPlaceMethods(SVMBuffer bufferA, SVMBuffer bufferB, float[] a, SVMBuffer bufferOne, float[] one) throws Exception {
        var b = -55.321f;
        SVMBufferTests.AddBufferInPlaceFloat(bufferA, bufferB, a, a);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.SubtractInPlaceFloat(bufferA, bufferB, a, a);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.MultiplyInPlaceFloat(bufferA, a, b);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.MultiplyInPlaceBufferFloat(bufferA, bufferB, a, a);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.DivisionInPlaceFloat(bufferA, a, b);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.DivisionInPlaceBufferFloat(bufferA, bufferB, a, a);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.LogInPlaceFloat(bufferA, a);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.SinInPlaceFloat(bufferA, a);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.CosInPlaceFloat(bufferA, a);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.ExpInPlaceFloat(bufferOne, one);
        bufferOne.releaseSVMBuffer();
        bufferOne = SVMBuffer.fromArray(SPECIES, one);
        SVMBufferTests.AbsInPlaceFloat(bufferA, a);
        bufferA.releaseSVMBuffer();
        bufferA = SVMBuffer.fromArray(SPECIES, a);
        SVMBufferTests.SqrtInPlaceFloat(bufferA, a);
    }

    public static void FromArrayFloat(float[] array) throws Exception {
        var buffer = SVMBuffer.fromArray(SPECIES, array);
        var bufferArray = new float[buffer.length];
        buffer.intoArray(bufferArray);
        AssertArray(bufferArray, array, SVMBufferTests.delta, "FromArrayFloat");
    }

    public static void IntoArrayFloat(SVMBuffer buffer, float[] array) throws Exception {
        var bufferArray = new float[buffer.length];
        buffer.intoArray(bufferArray);
        AssertArray(bufferArray, array, SVMBufferTests.delta, "IntoArrayFloat");
    }

    private static void MultiplyInPlaceBufferFloat(SVMBuffer buffer, SVMBuffer factors, float[] a, float[] b) throws Exception {
        var result = buffer.mulInPlace(factors);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyInPlaceBufferFloat";
        var expectedArray = ArrayMultiplication(a, b, info);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void MultiplyBufferFloat(SVMBuffer buffer, SVMBuffer factors, float[] a, float[] b) throws Exception {
        var result = buffer.mul(factors);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyBufferFloat";
        var expectedArray = ArrayMultiplication(a, b, info);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void MultiplyFFF(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.mul(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyFFF";
        var expectedArray = ArrayMultiplication(a, b);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
    }

    private static void MultiplyFIF(SVMBuffer buffer, float[] a, int b) throws Exception {
        var result = buffer.mul(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyFIF";
        var expectedArray = ArrayMultiplication(a, b);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
    }

    private static void MultiplyIII(SVMBuffer buffer, int[] a, int b) throws Exception {
        var result = buffer.mul(b);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyIII";
        var expectedArray = ArrayMultiplication(a, b);
        AssertArray(resultArray, expectedArray, info);
    }

    private static void MultiplyInPlaceFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.mulInPlace(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "MultiplyInPlaceFloat";
        var expectedArray = ArrayMultiplication(a, b);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void AddBufferFloat(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b) throws Exception {
        var result = buffer.add(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = ArrayAdditionFloat(a, b, info);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void AddBufferInPlaceFloat(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b) throws Exception {
        var result = buffer.addInPlace(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddInPlaceFloat";
        var expectedArray = ArrayAdditionFloat(a, b, info);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void SubtractBufferFloat(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b) throws Exception {
        var result = buffer.sub(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubtractBufferFloat";
        var expectedArray = ArraySubtractionFloat(a, b, info);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void SubtractFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.sub(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubtractFloat";
        var expectedArray = ArraySubtractionFloat(a, b);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
    }

    private static void SubtractInPlaceFloat(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b) throws Exception {
        var result = buffer.subInPlace(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "SubtractInPlaceFloat";
        var expectedArray = ArraySubtractionFloat(a, b, info);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
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
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void DivisionBufferFloat(SVMBuffer buffer, SVMBuffer factors, float[] a, float[] b) throws Exception {
        var result = buffer.div(factors);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "DivisionBufferFloat";
        var expectedArray = ArrayDivisionFloat(a, b, info);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void DivisionFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.div(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "DivisionFloat";
        var expectedArray = ArrayDivisionFloat(a, b);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
    }

    private static void DivisionInPlaceFloat(SVMBuffer buffer, float[] a, float b) throws Exception {
        var result = buffer.divInPlace(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info =  "DivisionInPlaceFloat";
        var expectedArray = ArrayDivisionFloat(a, b);
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void SqrtFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.sqrt();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArraySqrt(a);
        String info = "SqrtFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void SqrtInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.sqrtInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArraySqrt(a);
        String info = "SqrtInPlaceFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void LogFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.log();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayLog(a);
        String info = "LogFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void LogInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.logInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayLog(a);
        String info = "LogInPlaceFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void CosFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.cos();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayCos(a);
        String info = "CosFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void CosInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.cosInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayCos(a);
        String info = "CosInPlaceFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void ExpFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.exp();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        float[] expectedArray = ArrayExp(a);
        String info = "ExpFloat";
        float delta = 1.1f;
        AssertArray(resultArray, expectedArray, delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void ExpInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.expInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayExp(a);
        String info = "ExpInPlaceFloat";
        float delta = 0.1f;
        AssertArray(resultArray, expectedArray, delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void AbsFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.abs();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayAbs(a);
        String info = "AbsFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void AbsInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.absInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArrayAbs(a);
        String info = "AbsInPlaceFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressEqual(result, buffer, info);
    }

    private static void SinFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.sin();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArraySin(a);
        String info = "SinFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
        checkBufferAddressUnequal(result, buffer, info);
    }

    private static void SinInPlaceFloat(SVMBuffer buffer, float[] a) throws Exception {
        var result = buffer.sinInPlace();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = ArraySin(a);
        String info = "SinInPlaceFloat";
        AssertArray(resultArray, expectedArray, SVMBufferTests.delta, info);
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
            c[i] = Math.abs(a[i]);
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

    private static float[] ArrayMultiplication(float[] a, float b) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] * b;
        }
        return c;
    }

    private static float[] ArrayMultiplication(float[] a, int b) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] * b;
        }
        return c;
    }

    private static int[] ArrayMultiplication(int[] a, int b) throws Exception {
        var c = new int[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] * b;
        }
        return c;
    }

    private static float[] ArrayMultiplication(int[] a, float b) throws Exception {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)a[i] * b;
        }
        return c;
    }

    private static float[] ArrayMultiplication(float[] a, float[] b, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] * b[i];
        }
        return c;
    }

    private static float[] ArrayMultiplication(float[] a, int[] b, String info) throws Exception {
        checkArrayLength(b, a, info);
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = a[i] * b[i];
        }
        return c;
    }

    private static int[] ArrayMultiplication(int[] a, int[] b, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new int[a.length];
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

    private static void checkArrayLength(int[] a, float[] b, String info) throws Exception {
        if (a.length != b.length) {
            throw new RuntimeException(info + ": not same array length!");
        }
    }

    private static void checkArrayLength(int[] a, int[] b, String info) throws Exception {
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

    private static void SumReduceFloat(SVMBuffer buffer) throws Exception {
        var result = (int)buffer.sumReduceFloat();

        if (Math.abs(result - SVMBufferTests.size) > SVMBufferTests.delta) {
            throw new RuntimeException("SumReduce is: " + result + ", expected: " + SVMBufferTests.size);
        }
    }

    private static float[] iotaFloatArray(int size){
        var array = new float[size];
        for (int i = 0; i < size; i++){
            array[i] = (float)i;
        }
        return array;
    }

    private static int[] iotaIntArray(int size){
        var array = new int[size];
        for (int i = 0; i < size; i++){
            array[i] = i;
        }
        return array;
    }

    private static float[] ones(int size){
        var array = new float[size];
        for (int i = 0; i < size; i++){
            array[i] = 1.0f;
        }
        return array;
    }

    private static void AssertArray(float[] f1, float[] f2, float delta, String info) throws Exception {
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

    private static void AssertArray(int[] f1, int[] f2, String info) throws Exception {
        if (f1.length != f2.length) {
            throw new RuntimeException(info = "Float Array's do not have the same length!");
        }
        int n = f1.length;

        for (int i = 0; i < n; i++) {
            if (f1[i] != f2[i]) {
                throw new RuntimeException(info + ": Element " + i + " is not < delta!");
            }
        }
    }
}
