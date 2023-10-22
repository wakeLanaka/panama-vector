package jdk.internal.vm.vector;

/**
    abcd
*/
public class GPUSupport {
    static {
        registerNatives();
    }

    /**
        abcd
    */
    public GPUSupport(){}
    /**
        abcd
        @return
    */
    public static native int initializeGPU();

    /**
        abcd
        @param memObjectNumber abc
        @return abc
    */
    public static native int[] intoArray(int memObjectNumber, int length, int[] output);

    /**
        abc
        @param first abc
        @param second abc
    */
    public static native int add(int first, int second, int lowerBound, int upperBound, int maxGroupSize);

    /**
        abcd
        @param abc
    */
    public static native int allocateArray(int[] array);

    /**
        abcd
        @param a
        @param b
        @param c
    */
    public static native void gpuAddition(int[] a, int[] b, int[] c);

    /**
        abcd
        @param a
        @param b
        @param c
    */
    public static native void gpuAdditionHostPtr(int[] a, int[] b, int[] c);

    /**
        abcd
        @param a
        @param b
        @param c
        @param n
    */
    public static native void gpuMatrix(float[] a, float[] b, float[] c, int n);

    /**
        abcd
        @return
    */
    public static native int[] test();

    private static native int registerNatives();
}
