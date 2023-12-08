package jdk.incubator.vector;

import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer.OpenCLInformation;

/**
 *  Class to execute for-loops on the GPU using KernelBuilder
 */
public class ForKernelBuilder {

    private static GPUInformation info = new OpenCLInformation();

    private final KernelIndex index = new KernelIndex();

    /**
     *  Represents the body of the for-loop
     */
    public KernelBuilder body = null;

    private final int offset;

    private final int limit;

    private final int step;

    /**
     *  Returns the index of this for-loop
     *  @return index
     */
    public KernelIndex getIndex(){
        return index;
    }

    private ForKernelBuilder(int offset, int limit, int step){
        this.offset = offset;
        this.limit = limit;
        this.step = step;
    }

    /**
     *  FactoryMethod for creating ForKernelBuilder objects
     *  @param offset of the for-loop
     *  @param limit of the for-loop
     *  @param step after each for-loop iteration
     *  @return new ForKernelBuilder
     */
    public static ForKernelBuilder For(int offset, int limit, int step){
        var fkb = new ForKernelBuilder(offset, limit, step);
        fkb.body = new KernelBuilder(limit);
        return fkb;
    }

    /**
     *  Represents the end of the for-loop
     */
    public void End(){
        StringBuilder forSignature = new StringBuilder("for(int " + index.getIndex() + " = " + offset + "; " + index.getIndex() + " < " + limit + "; " + index.getIndex() + " +=" + step + "){");
        System.out.println("End:\t" + this.body.getKernelString());
        forSignature.append(this.body.getKernelString() + "}");
        this.body.ExecKernel(info, forSignature);
    }

    /**
     *  Returns the gpu information
     *  @return GPUInformation
     */
    public GPUInformation getInfo(){
        return ForKernelBuilder.info;
    }
}
