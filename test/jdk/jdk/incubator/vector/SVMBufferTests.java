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
import jdk.incubator.vector.SVMBuffer.Type;
import jdk.incubator.vector.GPUInformation;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/*
 * @test
 * @modules jdk.incubator.vector
 * @summary Tests the SVMBuffer
 * @run main SVMBufferTests
 */
public class SVMBufferTests {
    private static final GPUInformation SPECIES = SVMBuffer.SPECIES_PREFERRED;

    private static int size = 8022;

    public static void main(String[] args) throws Exception {
        // specialFunctions();
        // add();
        // sub();
        // mul();
        // div();
        // sumReduce();
        // mathFunctions();
        // rotations();
        binary();
        compare();
    }

    private static void specialFunctions() throws Exception {
        float valueFloat = 5.0f;
        int valueInt = 5;
        var iotaFloat = TestHelper.iotaArrayFloat(0, size);
        var iotaInt = TestHelper.iotaArrayInt(0, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);

        testIntoArray(bufferFloat, iotaFloat);
        testIntoArray(bufferInt, iotaInt);
        testFromArray(iotaFloat);
        testFromArray(iotaInt);
        testBroadcast(valueFloat);
        testBroadcast(valueInt);
        testFill(iotaFloat);
        testFill(iotaInt);
        testRepeatFullBuffer(bufferFloat, iotaFloat, valueInt);
        testRepeatEachNumber(bufferFloat, iotaFloat, valueInt);
    }

    private static void add() throws Exception {
        float valueFloat = 5.0f;
        int valueInt = 5;
        var iotaFloat = TestHelper.iotaArrayFloat(0, size);
        var iotaInt = TestHelper.iotaArrayInt(0, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);
        BinaryOperator<Float> funcFloat = (a,b) -> a+b;
        BinaryOperator<Integer> funcInt = (a,b) -> a+b;

        testAdd(bufferFloat, bufferFloat, iotaFloat, iotaFloat, funcFloat);
        testAdd(bufferInt, bufferInt, iotaInt, iotaInt, funcInt);
        testAdd(bufferFloat, bufferInt, iotaFloat, iotaInt, funcFloat);
        testAdd(bufferInt, bufferFloat, iotaFloat, iotaInt, funcFloat);
        testAdd(bufferInt, valueInt, iotaInt, valueInt, funcInt);
        testAdd(bufferInt, valueFloat, iotaInt, valueFloat, funcFloat);
        testAdd(bufferFloat, valueFloat, iotaFloat, valueFloat, funcFloat);
        testAdd(bufferFloat, valueInt, iotaFloat, valueInt, funcFloat);
    }

    private static void sub() throws Exception {
        float valueFloat = 5.0f;
        int valueInt = 5;
        var iotaFloat = TestHelper.iotaArrayFloat(0, size);
        var iotaInt = TestHelper.iotaArrayInt(0, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);
        BinaryOperator<Float> funcFloat = (a,b) -> a-b;
        BinaryOperator<Integer> funcInt = (a,b) -> a-b;

        testSub(bufferFloat, bufferFloat, iotaFloat, iotaFloat, funcFloat);
        testSub(bufferInt, bufferInt, iotaInt, iotaInt, funcInt);
        testSub(bufferFloat, bufferInt, iotaFloat, iotaInt, funcFloat);
        testSub(bufferInt, bufferFloat, iotaFloat, iotaInt, funcFloat);
        testSub(bufferInt, valueInt, iotaInt, valueInt, funcInt);
        testSub(bufferInt, valueFloat, iotaInt, valueFloat, funcFloat);
        testSub(bufferFloat, valueFloat, iotaFloat, valueFloat, funcFloat);
        testSub(bufferFloat, valueInt, iotaFloat, valueInt, funcFloat);
    }

    private static void mul() throws Exception {
        float valueFloat = 5.0f;
        int valueInt = 5;
        var iotaFloat = TestHelper.iotaArrayFloat(0, size);
        var iotaInt = TestHelper.iotaArrayInt(0, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);
        BinaryOperator<Float> funcFloat = (a,b) -> a*b;
        BinaryOperator<Integer> funcInt = (a,b) -> a*b;

        testMul(bufferFloat, bufferFloat, iotaFloat, iotaFloat, funcFloat);
        testMul(bufferInt, bufferInt, iotaInt, iotaInt, funcInt);
        testMul(bufferFloat, bufferInt, iotaFloat, iotaInt, funcFloat);
        testMul(bufferInt, bufferFloat, iotaFloat, iotaInt, funcFloat);
        testMul(bufferInt, valueInt, iotaInt, valueInt, funcInt);
        testMul(bufferInt, valueFloat, iotaInt, valueFloat, funcFloat);
        testMul(bufferFloat, valueFloat, iotaFloat, valueFloat, funcFloat);
        testMul(bufferFloat, valueInt, iotaFloat, valueInt, funcFloat);
    }

    private static void div() throws Exception {
        float valueFloat = 5.0f;
        int valueInt = 5;
        var iotaFloat = TestHelper.iotaArrayFloat(1, size);
        var iotaInt = TestHelper.iotaArrayInt(1, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);
        BinaryOperator<Float> funcFloat = (a,b) -> a/b;
        BinaryOperator<Integer> funcInt = (a,b) -> a/b;

        testDiv(bufferFloat, bufferFloat, iotaFloat, iotaFloat, funcFloat);
        testDiv(bufferInt, bufferInt, iotaInt, iotaInt, funcInt);
        testDiv(bufferFloat, bufferInt, iotaFloat, iotaInt, funcFloat);
        testDiv(bufferInt, bufferFloat, iotaFloat, iotaInt, funcFloat);
        testDiv(bufferInt, valueInt, iotaInt, valueInt, funcInt);
        testDiv(bufferInt, valueFloat, iotaInt, valueFloat, funcFloat);
        testDiv(bufferFloat, valueFloat, iotaFloat, valueFloat, funcFloat);
        testDiv(bufferFloat, valueInt, iotaFloat, valueInt, funcFloat);
    }

    private static void sumReduce() throws Exception {
        var iotaFloat = TestHelper.iotaArrayFloat(1, size);
        var iotaInt = TestHelper.iotaArrayInt(1, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);

        sumReduceFloat(bufferFloat);
        sumReduceInt(bufferInt);
    }

    private static void mathFunctions() throws Exception {
        float valueFloat = 5.0f;
        int valueInt = 5;
        var iotaFloat = TestHelper.iotaArrayFloat(0, size);
        var iotaInt = TestHelper.iotaArrayInt(0, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);
        UnaryOperator<Float> absFloat = (a) -> (float)Math.abs(a);
        UnaryOperator<Integer> absInt = (a) -> (int)Math.abs(a);
        UnaryOperator<Float> sqrtFloat = (a) -> (float)Math.sqrt(a);
        UnaryOperator<Float> cosFloat = (a) -> (float)Math.cos(a);
        UnaryOperator<Float> sinFloat = (a) -> (float)Math.sin(a);
        UnaryOperator<Float> logFloat = (a) -> (float)Math.log(a);
        BinaryOperator<Float> maxFloat = (a,b) -> (float)Math.max(a,b);
        BinaryOperator<Integer> maxInt = (a,b) -> (int)Math.max(a,b);
        BinaryOperator<Float> minFloat = (a,b) -> (float)Math.min(a,b);
        BinaryOperator<Integer> minInt = (a,b) -> (int)Math.min(a,b);

        testSqrt(bufferFloat, iotaFloat, sqrtFloat);
        testSqrt(bufferInt, iotaInt, sqrtFloat);
        testLog(bufferFloat, iotaFloat, logFloat);
        testLog(bufferInt, iotaInt, logFloat);
        testCos(bufferFloat, iotaFloat, cosFloat);
        testCos(bufferInt, iotaInt, cosFloat);
        testSin(bufferFloat, iotaFloat, sinFloat);
        testSin(bufferInt, iotaInt, sinFloat);
        testAbs(bufferFloat, iotaFloat, absFloat);
        testAbs(bufferInt, iotaInt, absInt);
        testMax(bufferFloat, iotaFloat, valueFloat, maxFloat);
        testMax(bufferInt, iotaInt, valueInt, maxInt);
        testMin(bufferFloat, iotaFloat, valueFloat, minFloat);
        testMin(bufferInt, iotaInt, valueInt, minInt);
    }

    private static void rotations() throws Exception {
        int valueInt = 5;
        var iotaFloat = TestHelper.iotaArrayFloat(0, size);
        var iotaInt = TestHelper.iotaArrayInt(0, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);
        testRor(bufferFloat, iotaFloat, valueInt);
        testRol(bufferFloat, iotaFloat, valueInt);
        testRor(bufferInt, iotaInt, valueInt);
        testRol(bufferInt, iotaInt, valueInt);
    }

    private static void binary() throws Exception {
        int valueInt = 5;
        var iotaInt = TestHelper.iotaArrayInt(0, size);
        var bufferInt = SVMBuffer.fromArray(SPECIES, iotaInt);
        BinaryOperator<Integer> ashrFunc = (a,b) -> a >> b;
        BinaryOperator<Integer> lshlFunc = (a,b) -> a << b;
        BinaryOperator<Integer> andFunc = (a,b) -> a & b;
        BinaryOperator<Integer> orFunc = (a,b) -> a | b;

        testAshr(bufferInt, iotaInt, valueInt, ashrFunc);
        testLshl(bufferInt, iotaInt, valueInt, lshlFunc);
        testAnd(bufferInt, iotaInt, valueInt, andFunc);
        testOr(bufferInt, iotaInt, valueInt, orFunc);
    }

    private static void compare() throws Exception{
        float valueFloat = 5.0f;
        var iotaFloat = TestHelper.iotaArrayFloat(0, size);
        var bufferFloat = SVMBuffer.fromArray(SPECIES, iotaFloat);
        BinaryOperator<Float> gtFuncFloat = (a,b) -> a > b ? 1.0f : 0.0f;
        var mask = TestHelper.createMask(size);
        var bufferMask = SVMBuffer.fromArray(SPECIES, mask);

        testCompareGT(bufferFloat, iotaFloat, valueFloat, gtFuncFloat);
        testBlend(bufferFloat, bufferFloat, iotaFloat, iotaFloat, bufferMask, mask);
    }

    public static void testFromArray(float[] array) throws Exception {
        var buffer = SVMBuffer.fromArray(SPECIES, array);
        var bufferArray = new float[buffer.length];
        buffer.intoArray(bufferArray);
        TestHelper.AssertArray(bufferArray, array, "FromArrayFloat");
    }

    public static void testFromArray(int[] array) throws Exception {
        var buffer = SVMBuffer.fromArray(SPECIES, array);
        var bufferArray = new int[buffer.length];
        buffer.intoArray(bufferArray);
        TestHelper.AssertArray(bufferArray, array, "FromArrayInt");
    }

    public static void testIntoArray(SVMBuffer buffer, float[] array) throws Exception {
        var bufferArray = new float[buffer.length];
        buffer.intoArray(bufferArray);
        TestHelper.AssertArray(bufferArray, array, "toArrayFloat");
    }

    public static void testIntoArray(SVMBuffer buffer, int[] array) throws Exception {
        var bufferArray = new int[buffer.length];
        buffer.intoArray(bufferArray);
        TestHelper.AssertArray(bufferArray, array, "toArrayInt");
    }

    private static void testAdd(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.add(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAdd(SVMBuffer buffer, SVMBuffer summand, float[] a, int[] b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.add(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAdd(SVMBuffer buffer, SVMBuffer summand, int[] a, int[] b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.add(summand);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAdd(SVMBuffer buffer, int valueSVM, float[] a, int value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.add(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAdd(SVMBuffer buffer, int valueSVM, int[] a, int value, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.add(valueSVM);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAdd(SVMBuffer buffer, float valueSVM, float[] a, float value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.add(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAdd(SVMBuffer buffer, float valueSVM, int[] a, float value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.add(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "AddBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSub(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.sub(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSub(SVMBuffer buffer, SVMBuffer summand, float[] a, int[] b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.sub(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSub(SVMBuffer buffer, SVMBuffer summand, int[] a, int[] b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.sub(summand);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "SubBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSub(SVMBuffer buffer, int valueSVM, float[] a, int value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.sub(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSub(SVMBuffer buffer, int valueSVM, int[] a, int value, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.sub(valueSVM);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "SubBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSub(SVMBuffer buffer, float valueSVM, float[] a, float value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.sub(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSub(SVMBuffer buffer, float valueSVM, int[] a, float value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.sub(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "SubBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMul(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.mul(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "MulBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMul(SVMBuffer buffer, SVMBuffer summand, float[] a, int[] b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.mul(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "MulBufferFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b,op,info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMul(SVMBuffer buffer, SVMBuffer summand, int[] a, int[] b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.mul(summand);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "MulBufferIntInt";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMul(SVMBuffer buffer, int valueSVM, float[] a, int value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.mul(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "MulBufferFloatint";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value,op,info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMul(SVMBuffer buffer, int valueSVM, int[] a, int value, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.mul(valueSVM);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "MulBufferIntint";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMul(SVMBuffer buffer, float valueSVM, float[] a, float value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.mul(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "MulBufferFloatfloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value,op,info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMul(SVMBuffer buffer, float valueSVM, int[] a, float value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.mul(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "MulBufferIntfloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testDiv(SVMBuffer buffer, SVMBuffer summand, float[] a, float[] b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.div(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "DivBufferFloatFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testDiv(SVMBuffer buffer, SVMBuffer summand, float[] a, int[] b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.div(summand);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "DivBufferFloatInt";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testDiv(SVMBuffer buffer, SVMBuffer summand, int[] a, int[] b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.div(summand);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "DivBufferIntInt";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testDiv(SVMBuffer buffer, int valueSVM, float[] a, int value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.div(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "DivBufferFloatint";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testDiv(SVMBuffer buffer, int valueSVM, int[] a, int value, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.div(valueSVM);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "DivBufferIntint";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testDiv(SVMBuffer buffer, float valueSVM, float[] a, float value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.div(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "DivBufferFloatfloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testDiv(SVMBuffer buffer, float valueSVM, int[] a, float value, BinaryOperator<Float> op) throws Exception {
        var result = buffer.div(valueSVM);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "DivBufferIntfloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a,value, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testRol(SVMBuffer buffer, float[] value, int n) throws Exception {
        float[] resultArray = new float[size];
        var result = buffer.rol(n);
        result.intoArray(resultArray);
        var expected = TestHelper.RotateArrayLeft(value, n);
        TestHelper.AssertArray(resultArray, expected, "RolFloat");
    }

    private static void testRol(SVMBuffer buffer, int[] value, int n) throws Exception {
        int[] resultArray = new int[size];
        var result = buffer.rol(n);
        result.intoArray(resultArray);
        var expected = TestHelper.RotateArrayLeft(value, n);
        TestHelper.AssertArray(resultArray, expected, "RolInt");
    }

    private static void testRor(SVMBuffer buffer, float[] value, int n) throws Exception {
        float[] resultArray = new float[size];
        var result = buffer.ror(n);
        result.intoArray(resultArray);
        var expected = TestHelper.RotateArrayRight(value, n);
        TestHelper.AssertArray(resultArray, expected, "RolFloat");
    }

    private static void testRor(SVMBuffer buffer, int[] value, int n) throws Exception {
        int[] resultArray = new int[size];
        var result = buffer.ror(n);
        result.intoArray(resultArray);
        var expected = TestHelper.RotateArrayRight(value, n);
        TestHelper.AssertArray(resultArray, expected, "RolInt");
    }

    private static void testBroadcast(float value) throws Exception {
        float[] resultArray = new float[SVMBufferTests.size];
        var buffer = SVMBuffer.broadcast(SPECIES, value, SVMBufferTests.size);
        buffer.intoArray(resultArray);
        TestHelper.AssertArray(resultArray, value, "BroadcastFloat");
    }

    private static void testBroadcast(int value) throws Exception {
        int[] resultArray = new int[SVMBufferTests.size];
        var buffer = SVMBuffer.broadcast(SPECIES, value, SVMBufferTests.size);
        buffer.intoArray(resultArray);
        TestHelper.AssertArray(resultArray, value, "BroadcastInt");
    }

    private static void testFill(float[] value) throws Exception {
        var zeros = SVMBuffer.zero(SPECIES, value.length, Type.FLOAT);
        float[] resultArray = new float[SVMBufferTests.size];
        var buffer = zeros.fill(value);
        buffer.intoArray(resultArray);
        TestHelper.AssertArray(resultArray, value, "fillFloat");
    }

    private static void testFill(int[] value) throws Exception {
        var zeros = SVMBuffer.zero(SPECIES, value.length, Type.INT);
        int[] resultArray = new int[SVMBufferTests.size];
        var buffer = zeros.fill(value);
        buffer.intoArray(resultArray);
        TestHelper.AssertArray(resultArray, value, "fillInt");
    }

    private static void testRepeatFullBuffer(SVMBuffer buffer, float[] value, int n) throws Exception {
        float[] resultArray = new float[size * n];
        var results = buffer.repeatFullBuffer(n);
        results.intoArray(resultArray);
        var expected = TestHelper.repeatFullArray(value, n);
        TestHelper.AssertArray(resultArray, expected, "repeatFullBuffer");
    }

    private static void testRepeatEachNumber(SVMBuffer buffer, float[] value, int n) throws Exception {
        float[] resultArray = new float[size * n];
        var results = buffer.repeatEachNumber(n);
        results.intoArray(resultArray);
        var expected = TestHelper.repeatEachNumber(value, n);
        TestHelper.AssertArray(resultArray, expected, "repeatEachNumber");
    }

    private static void testZeroFloat() throws Exception {
        float[] resultArray = new float[size];
        var zeros = SVMBuffer.zero(SPECIES, size, Type.FLOAT);
        zeros.intoArray(resultArray);
        TestHelper.AssertArray(resultArray, 0.0f, "fillFloat");
    }

    private static void testZeroInt() throws Exception {
        int[] resultArray = new int[size];
        var zeros = SVMBuffer.zero(SPECIES, size, Type.INT);
        zeros.intoArray(resultArray);
        TestHelper.AssertArray(resultArray, 0, "fillInt");
    }

    private static void testSqrt(SVMBuffer buffer, float[] a, UnaryOperator<Float> op) throws Exception {

        var result = buffer.sqrt();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "SqrtFloat";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSqrt(SVMBuffer buffer, int[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.sqrt();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "SqrtInt";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testLog(SVMBuffer buffer, float[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.log();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "LogFloat";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testLog(SVMBuffer buffer, int[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.log();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "LogInt";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testCos(SVMBuffer buffer, float[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.cos();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "CosFloat";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testCos(SVMBuffer buffer, int[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.cos();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "CosInt";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testExp(SVMBuffer buffer, float[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.exp();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        float[] expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "ExpFloat";
        float delta = 1.1f;
        TestHelper.AssertArray(resultArray, expectedArray, delta, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testExp(SVMBuffer buffer, int[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.exp();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        float[] expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "ExpInt";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAbs(SVMBuffer buffer, float[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.abs();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "AbsFloat";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAbs(SVMBuffer buffer, int[] a, UnaryOperator<Integer> op) throws Exception {
        var result = buffer.abs();
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperatorInt(a, op);
        String info = "AbsInt";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSin(SVMBuffer buffer, float[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.sin();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "SinFloat";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testSin(SVMBuffer buffer, int[] a, UnaryOperator<Float> op) throws Exception {
        var result = buffer.sin();
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        var expectedArray = TestHelper.ArrayUnaryOperator(a, op);
        String info = "SinInt";
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMax(SVMBuffer buffer, float[] a, float b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.max(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "MaxFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMax(SVMBuffer buffer, int[] a, int b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.max(b);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "MaxInt";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMin(SVMBuffer buffer, float[] a, float b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.min(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "MinFloat";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testMin(SVMBuffer buffer, int[] a, int b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.min(b);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "MinInt";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAshr(SVMBuffer buffer, int[] a, int b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.ashr(b);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "Ashr";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testLshl(SVMBuffer buffer, int[] a, int b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.lshl(b);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "Ashr";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testAnd(SVMBuffer buffer, int[] a, int b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.and(b);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "Ashr";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testOr(SVMBuffer buffer, int[] a, int b, BinaryOperator<Integer> op) throws Exception {
        var result = buffer.or(b);
        var resultArray = new int[buffer.length];
        result.intoArray(resultArray);
        String info = "Ashr";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testCompareGT(SVMBuffer buffer, float[] a, float b, BinaryOperator<Float> op) throws Exception {
        var result = buffer.compareGT(b);
        var resultArray = new float[buffer.length];
        result.intoArray(resultArray);
        String info = "compareGT";
        var expectedArray = TestHelper.ArrayBinaryOperator(a, b, op, info);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, buffer, info);
    }

    private static void testBlend(SVMBuffer bufferA, SVMBuffer bufferB, float[] a, float[] b, SVMBuffer bufferMask, float[] mask) throws Exception {
        var result = bufferA.blend(bufferB, bufferMask);
        var resultArray = new float[bufferA.length];
        result.intoArray(resultArray);
        String info = "compareGT";
        var expectedArray = TestHelper.blend(a, b, mask);
        TestHelper.AssertArray(resultArray, expectedArray, info);
        TestHelper.checkBufferAddressUnequal(result, bufferA, info);
    }

    private static void sumReduceFloat(SVMBuffer buffer) throws Exception {
        var result = buffer.sumReduceFloat();
        var expected = size * (size + 1)/2;

        if (Math.abs(result - expected) > TestHelper.delta) {
            throw new RuntimeException("SumReduceFloat is: " + result + ", expected: " + expected);
        }
    }

    private static void sumReduceInt(SVMBuffer buffer) throws Exception {
        var result = buffer.sumReduceInt();
        var expected = size * (size + 1)/2;

        if (result != expected) {
            throw new RuntimeException("SumReduceInt is: " + result + ", expected: " + expected);
        }
    }
}
