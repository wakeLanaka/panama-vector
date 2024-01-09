
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

import jdk.incubator.vector.SVMBuffer;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/*
 * @test
 * @modules jdk.incubator.vector
 * @summary Helps for SVMBuffer tests
 */
public class TestHelper {

    public static final float delta = 0.001f;

    public static float[] ArrayBinaryOperator(float[] a, float[] b, BinaryOperator<Float> op, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)op.apply(Float.valueOf(a[i]), Float.valueOf(b[i]));
        }
        return c;
    }

    public static float[] ArrayBinaryOperator(float[] a, float b, BinaryOperator<Float> op, String info) {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)op.apply(Float.valueOf(a[i]), Float.valueOf(b));
        }
        return c;
    }

    public static float[] ArrayBinaryOperator(float[] a, int[] b, BinaryOperator<Float> op, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)op.apply(Float.valueOf(a[i]), Float.valueOf(b[i]));
        }
        return c;
    }

    public static float[] ArrayBinaryOperator(int[] a, float b, BinaryOperator<Float> op, String info) {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)op.apply(Float.valueOf(a[i]), Float.valueOf(b));
        }
        return c;
    }

    public static float[] ArrayBinaryOperator(float[] a, int b, BinaryOperator<Float> op, String info) {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)op.apply(Float.valueOf(a[i]), Float.valueOf(b));
        }
        return c;
    }

    public static int[] ArrayBinaryOperator(int[] a, int[] b, BinaryOperator<Integer> op, String info) throws Exception {
        checkArrayLength(a, b, info);
        var c = new int[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (int)op.apply(Integer.valueOf(a[i]), Integer.valueOf(b[i]));
        }
        return c;
    }

    public static int[] ArrayBinaryOperator(int[] a, int b, BinaryOperator<Integer> op, String info) {
        var c = new int[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (int)op.apply(Integer.valueOf(a[i]), Integer.valueOf(b));
        }
        return c;
    }

    public static float[] ArrayUnaryOperator(int[] a, UnaryOperator<Float> op) {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)op.apply(Float.valueOf(a[i]));
        }
        return c;
    }

    public static int[] ArrayUnaryOperatorInt(int[] a, UnaryOperator<Integer> op) {
        var c = new int[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (int)op.apply(Integer.valueOf(a[i]));
        }
        return c;
    }

    public static float[] ArrayUnaryOperator(float[] a, UnaryOperator<Float> op) {
        var c = new float[a.length];
        for (int i = 0; i < a.length; i++){
            c[i] = (float)op.apply(Float.valueOf(a[i]));
        }
        return c;
    }

    public static void checkArrayLength(float[] a, float[] b, String info) throws Exception {
        if (a.length != b.length) {
            throw new RuntimeException(info + ": not same array length!");
        }
    }

    public static void checkArrayLength(float[] a, int[] b, String info) throws Exception {
        if (a.length != b.length) {
            throw new RuntimeException(info + ": not same array length!");
        }
    }

    public static void checkArrayLength(int[] a, int[] b, String info) throws Exception {
        if (a.length != b.length) {
            throw new RuntimeException(info + ": not same array length!");
        }
    }

    public static float[] iotaArrayFloat(int offset, int size){
        var array = new float[size];
        for (int i = 0; i < size; i++){
            array[i] = (float)i + (float)offset;
        }
        return array;
    }

    public static int[] iotaArrayInt(int offset, int size){
        var array = new int[size];
        for (int i = 0; i < size; i++){
            array[i] = i + offset;
        }
        return array;
    }

    public static float[] onesFloat(int size){
        var array = new float[size];
        for (int i = 0; i < size; i++){
            array[i] = 1.0f;
        }
        return array;
    }

    public static int[] onesInt(int size){
        var array = new int[size];
        for (int i = 0; i < size; i++){
            array[i] = 1;
        }
        return array;
    }

    public static void AssertArray(float[] f1, float[] f2, String info) throws Exception {
        AssertArray(f1, f2, delta, info);
    }

    public static void AssertArray(float[] f1, float[] f2, float delta, String info) throws Exception {
        if (f1.length != f2.length) {
            throw new RuntimeException(info = "Float Array's do not have the same length!");
        }
        int n = f1.length;

        for (int i = 0; i < n; i++) {
            if (Math.abs(f1[i] - f2[i]) > delta) {
                throw new RuntimeException(info + ": Element " + i + " is not < " + delta);
            }
        }
    }

    public static void AssertArray(int[] f1, int[] f2, String info) throws Exception {
        if (f1.length != f2.length) {
            throw new RuntimeException(info = "Float Array's do not have the same length!");
        }
        int n = f1.length;

        for (int i = 0; i < n; i++) {
            if (f1[i] != f2[i]) {
                throw new RuntimeException(info + ": Element " + i + " is not <" + delta);
            }
        }
    }

    public static void AssertArray(float[] array, float value, String info) throws Exception {
        for (int i = 0; i < array.length; i++){
            if (Math.abs(array[i] - value) > delta) {
                throw new RuntimeException(info + ": Element " + i + " is not < delta!");
            }
        }
    }

    public static void AssertArray(int[] array, int value, String info) throws Exception {
        for (int i = 0; i < array.length; i++){
            if (array[i] != value) {
                throw new RuntimeException(info + ": Element " + i + " are not equal!");
            }
        }
    }

    public static void checkBufferAddressEqual(SVMBuffer result, SVMBuffer expected, String info) throws Exception {
        if(result.svmBuffer != expected.svmBuffer){
            throw new RuntimeException(info + ": svmBuffers are not the same!");
        }
    }

    public static void checkBufferAddressUnequal(SVMBuffer result, SVMBuffer expected, String info) throws Exception {
        if(result.svmBuffer == expected.svmBuffer){
            throw new RuntimeException(info + ": svmBuffers are the same!");
        }
    }

    public static float[] RotateArrayLeft(float[] array, int n){
        var c = new float[array.length];
        for(int i = 0; i < array.length; i++){
            c[i] = array[(i + n) % array.length];
        }
        return c;
    }

    public static int[] RotateArrayLeft(int[] array, int n){
        var c = new int[array.length];
        for(int i = 0; i < array.length; i++){
            c[i] = array[(i + n) % array.length];
        }
        return c;
    }

    public static float[] RotateArrayRight(float[] array, int n){
        var c = new float[array.length];
        for(int i = 0; i < array.length; i++){
            c[(i + n) % array.length] = array[i];
        }
        return c;
    }

    public static int[] RotateArrayRight(int[] array, int n){
        var c = new int[array.length];
        for(int i = 0; i < array.length; i++){
            c[(i + n) % array.length] = array[i];
        }
        return c;
    }

    public static float[] createMask(int size){
        var c = new float[size];
        for(int i = 0; i < c.length; i++){
            c[i] = i % 3 == 0 ? 1.0f : 0.0f;
        }
        return c;
    }

    public static float[] blend(float[] a, float[] b, float[] mask){
        var c = new float[a.length];
        for(int i = 0; i < c.length; i++){
            c[i] = a[i] + (b[i] - a[i]) * mask[i];
        }
        return c;
    }

    public static float[] repeatFullArray(float[] array, int n){
        var c = new float[array.length * n];
        for(int i = 0; i < c.length; i++){
            c[i] = array[i % array.length];
        }
        return c;
    }

    public static float[] repeatEachNumber(float[] array, int n){
        var c = new float[array.length * n];
        for(int i = 0; i < array.length; i++){
            for(int x = 0; x < n; x++){
                c[i * n + x] = array[i];
            }
        }
        return c;
    }
}
