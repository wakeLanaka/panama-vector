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

    public static native void ReleaseSVMBuffer(long jContext, long jCommandQueue, long jBuffer);

    public static native void ReleaseSVMBufferInt(long jContext, long jCommandQueue, long jBuffer);

    public static native void AddSVMBuffer(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length);

    public static native void FmaSVMBuffer(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, long jBuffer4, int length);

    public static native void MatrixFmaSVMBuffer(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int K, int N, int k, int length);

    public static native float SumReduce(long jContext, long jProgram, long jCommandQueue, long jBuffer, int maxWorkGroupSize, int length);

    public static native void Subtract(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length);

    public static native void Subtract(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, float jMinuend, int length);

    public static native void MultiplyFFF(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, float jFactor, int length);

    public static native void MultiplyIFF(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, float jFactor, int length);

    public static native void MultiplyFFI(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int jFactor, int length);

    public static native void MultiplyIII(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int jFactor, int length);

    public static native void Multiply(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length);

    public static native void MultiplyInt(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length);

    public static native void Sqrt(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int length);

    public static native void Division(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, float jDivisor, int length);

    public static native void Division(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jBuffer3, int length);

    public static native void Log(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int length);

    public static native void Exp(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int length);

    public static native void Abs(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, int length);

    public static native void CompareGT(long jProgram, long jCommandQueue, long jBuffer1, float comparee, long jBuffer2, int length);

    public static native void Blend(long jProgram, long jCommandQueue, long jBuffer1, long jBuffer2, long jMask, long jBuffer3, int length);

    public static native void BlackScholes(long jProgram, long jCommandQueue, float sig, float r, long xBuffer, long callBuffer, long putBuffer, long tBuffer, long s0, int length);

    public static native void Sin(long jProgram, long jCommandQueue, long b1, long b2, int length);

    public static native void Cos(long jProgram, long jCommandQueue, long b1, long b2, int length);

    public static native void MultiplyInPlaceRepeat(long jProgram, long jCommandQueue, long b1, long b2, int size, int length);

    public static native void MultiplyRange(long jProgram, long jCommandQueue, long b1, int index1, long b2, int index2, long b3, int amount);

    public static native void DFT(long jProgram, long jCommandQueue, long b1, long b2, long b3, long b4, int size);

    public static native void ForSum(long jProgram, long jCommandQueue, long b1, long b2, float v1, int length);

    public static native void ExecuteKernel(long jKernel, long jCommandQueue, int length);

    public static native long CreateKernel(long jKernel);

    public static native void SetKernelArgument(long jKernel, long buffer, int argumentNumber);

    public static native void SetKernelArgument(long jKernel, int value, int argumentNumber);

    public static native void SetKernelArgument(long jKernel, float value, int argumentNumber);

    public static native void Repeat1(long jProgram, long jCommandQueue, long b1, long b2, int repetition, int length);

    public static native void Repeat2(long jProgram, long jCommandQueue, long b1, long b2, int repetition, int length);

    public static native void Ashr(long jProgram, long jCommandQueue, long b1, long b2, int amount, int length);

    public static native void Lshl(long jProgram, long jCommandQueue, long b1, long b2, int amount, int length);

    public static native void ToInt(long jProgram, long jCommandQueue, long b1, long b2, int length);

    public static native void ToFloat(long jProgram, long jCommandQueue, long b1, long b2, int length);

    public static native void And(long jProgram, long jCommandQueue, long b1, long b2, int value, int length);

    public static native void Or(long jProgram, long jCommandQueue, long b1, long b2, long b3, int length);

    public static native void Max(long jProgram, long jCommandQueue, long b1, long b2, float value, int length);

    public static native void Min(long jProgram, long jCommandQueue, long b1, long b2, float value, int length);

    public static native void Ror(long jProgram, long jCommandQueue, long b1, long b2, int amount, int length);

    public static native void Rol(long jProgram, long jCommandQueue, long b1, long b2, int amount, int length);

    public static native void MultiplyArea(long jProgram, long jCommandQueue, long b1, long b2, long b3, int offset, int thisWidth, int factorWidth, int length);

    public static native void EachAreaFMA(long jProgram, long jCommandQueue, long b1, long b2, long b3, int width, int kernelWidth, int length);
}
