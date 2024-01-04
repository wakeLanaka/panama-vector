package jdk.internal.vm.vector;

public class SVMBufferSupport {
    static {
        registerNatives();
    }

    private static native int registerNatives();

    public static native long CreateContext(long jDevice);

    public static native void ReleaseContext(long jContext);

    public static native long CreateProgram(long jContext, String jKernelString);

    public static native void ReleaseProgram(long jContext);

    public static native long CreateCommandQueue(long jContext, long jDevice);

    public static native void ReleaseCommandQueue(long jCommandQueue);

    public static native long CreateDevice();

    public static native void ReleaseDevice(long jDevice);

    public static native int GetMaxWorkGroupSize(long jDevice);

    public static native long CreateReadWriteFloatSVMBuffer(long jContext, int length);

    public static native long CreateReadWriteIntSVMBuffer(long jContext, int length);

    public static native long CopyFromArray(long jContext, long jCommandQueue, int[] jArray);

    public static native long CopyFromArray(long jContext, long jCommandQueue, float[] jArray);

    public static native long Fill(long jContext, long jCommandQueue, long b1, float[] jArray);

    public static native long Fill(long jContext, long jCommandQueue, long b1, int[] jArray);

    public static native void CopyToArray(long jContext, long jCommandQueue, long jBuffer, int[] jArray);

    public static native void CopyToArray(long jContext, long jCommandQueue, long jBuffer, float[] jArray);

    public static native void CopyToArray(long jContext, long jCommandQueue, long jBuffer, float[] jArray, int bufferLength, int length, int offset);

    public static native void CopyToArray(long jContext, long jCommandQueue, long jBuffer, int[] jArray, int bufferLength, int length, int offset);

    public static native void ReleaseSVMBuffer(long jContext, long jCommandQueue, long jBuffer);

    public static native void ReleaseSVMBufferInt(long jContext, long jCommandQueue, long jBuffer);

    public static native void FmaSVMBuffer(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, long jBuffer4, int length);

    public static native void MatrixFmaSVMBuffer(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int K, int N, int k, int length);

    public static native float SumReduceFLOAT(long jContext, long jProgram, long jCommandQueue, long jBuffer, int maxWorkGroupSize, int length);

    public static native int SumReduceINT(long jContext, long jProgram, long jCommandQueue, long jBuffer, int maxWorkGroupSize, int length);

    public static native void executeKernelWithTypesFFF(long jProgram, long jCommandQueue, long jBuffer1, float jFactor, long jBuffer2, int length, String kernel);

    public static native void executeKernelWithTypesIFF(long jProgram, long jCommandQueue, long jBuffer1, float jFactor, long jBuffer2, int length, String kernel);

    public static native void executeKernelWithTypesFIF(long jProgram, long jCommandQueue, long jBuffer1, int jFactor, long jBuffer2, int length, String kernel);

    public static native void executeKernelWithTypesIII(long jProgram, long jCommandQueue, long jBuffer1, int jFactor, long jBuffer2, int length, String kernel);

    public static native void executeKernelWithTypesBufferFFF(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length, String kernel);

    public static native void executeKernelWithTypesBufferIFF(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length, String kernel);

    public static native void executeKernelWithTypesBufferFIF(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length, String kernel);

    public static native void executeKernelWithTypesBufferIII(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length, String kernel);

    public static native void executeKernelWithTypesBufferFF(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int length, String kernel);

    public static native void executeKernelWithTypesBufferIF(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int length, String kernel);

    public static native void executeKernelWithTypesBufferFI(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int length, String kernel);

    public static native void executeKernelWithTypesBufferII(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int length, String kernel);

    public static native void executeKernelWithTypesBufferIIII(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int size,int length, String kernel);
    public static native void executeKernelWithTypesBufferFFFI(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int size,int length, String kernel);
    public static native void executeKernelWithTypesBufferFIFI(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int size, int length, String kernel);
    public static native void executeKernelWithTypesBufferIFFI(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int size, int length, String kernel);

    public static native void Blend(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jMask, long jBuffer3, int length);

    public static native void MultiplyRange(long jProgram, long jCommandQueue, long b1, int index1, long b2, int index2, long b3, int amount);

    public static native void ExecuteKernel(long jKernel, long jCommandQueue, int length);

    public static native long CreateKernel(long jKernel);

    public static native void SetKernelArgument(long jKernel, long buffer, int argumentNumber);

    public static native void SetKernelArgument(long jKernel, int value, int argumentNumber);

    public static native void SetKernelArgument(long jKernel, float value, int argumentNumber);

    public static native void ToInt(long jProgram, long jCommandQueue, long b1, long b2, int length);

    public static native void ToFloat(long jProgram, long jCommandQueue, long b1, long b2, int length);

    public static native void MultiplyArea(long jProgram, long jCommandQueue, long b1, long b2, long b3, int offset, int thisWidth, int factorWidth, int length);

    public static native void EachAreaFMA(long jProgram, long jCommandQueue, long b1, long b2, long b3, int width, int kernelWidth, int length);
}
