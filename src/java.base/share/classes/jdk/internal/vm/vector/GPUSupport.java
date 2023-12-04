package jdk.internal.vm.vector;

/**
    abcd
*/
public class GPUSupport {
    static {
        registerNatives();
    }

    public GPUSupport(){}

    public static native int initializeGPU();

    public static native long CreateContext(long jDevice);

    public static native void ReleaseContext(long jContext);

    public static native long CreateProgram(long jContext, String jKernelString);

    public static native void ReleaseProgram(long jContext);

    public static native long CreateCommandQueue(long jContext, long jDevice);

    public static native void ReleaseCommandQueue(long jCommandQueue);

    public static native long CreateDevice();

    public static native void ReleaseDevice(long jDevice);

    public static native float[] Add(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, int length);

    public static native void Fma(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float[] jBuffer3, float[] jBuffer4, int length);

    public static native void MatrixFma(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float[] jBuffer3, int K, int N, int k, int length);

    public static native float ReduceAdd(long jProgram, long jContext, long jCommandQueue, float[] jBuffer, int length);

    public static native void Subtract(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float[] jBuffer3, int length);

    public static native void SubtractionMinuend(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float jMinuend, int length);

    public static native void SubtractionSubtrahend(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float jSubtrahend, int length);

    public static native void Multiply(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float jFactor, int length);

    public static native void Multiply(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float[] jBuffer3, int length);

    public static native void Sqrt(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, int length);

    public static native void Division(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float jDivisor, int length);

    public static native void Division(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float[] jBuffer3, int length);

    public static native void Log(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, int length);

    public static native void Exp(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, int length);

    public static native void Abs(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, int length);

    public static native void CompareGT(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float comparee, float[] jBuffer2, int length);

    public static native void Blend(long jProgram, long jContext, long jCommandQueue, float[] jBuffer1, float[] jBuffer2, float[] jMask, float[] jBuffer3, int length);

    public static native void gpuAddHostPtr(int[] a, int[] b, int[] c);

    private static native int registerNatives();
}
