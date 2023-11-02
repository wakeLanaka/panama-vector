package jdk.internal.vm.vector;

public class SVMBufferSupport {
    static {
        registerNatives();
    }

    private static native int registerNatives();

    public static native long CreateContext(long jDevice);

    public static native void ReleaseContext(long jContext);

    public static native long CreateProgram(long jContext, String jKernelString);

    public static native long ReleaseProgram(long jContext, String jKernelString);

    public static native long CreateCommandQueue(long jContext, long jDevice);

    public static native void ReleaseCommandQueue(long jCommandQueue);

    public static native long CreateDevice();

    public static native void ReleaseDevice(long jDevice);

    public static native void WriteSVMBuffer(long jCommandQueue, long jBuffer, int element, int length, int value);

    public static native long CopyFromArray(long jContext, long jCommandQueue, int[] jArray);

    public static native long CopyFromFloatArray(long jContext, long jCommandQueue, float[] jArray);

    public static native void CopyToArray(long jContext, long jCommandQueue, long jBuffer, int[] jArray);

    public static native void CopyToFloatArray(long jContext, long jCommandQueue, long jBuffer, float[] jArray);

    public static native void ReleaseSVMBuffer(long jCommandQueue, long jBuffer);

    public static native long CreateReadSVMBuffer(long jContext, int length);

    public static native long CreateWriteSVMBuffer(long jContext, int length);

    public static native int ReadSVMBuffer(long jCommandQueue, long jBuffer, int element, int length);

    public static native void AddSVMBuffer(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length);

    public static native void FmaSVMBuffer(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length);

    public static native void MatrixFmaSVMBuffer(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int K, int N, int k, int length);

    public static native float ReduceAdd(long jContext, long jProgram, long jCommandQueue, long jBuffer, int length);
}
