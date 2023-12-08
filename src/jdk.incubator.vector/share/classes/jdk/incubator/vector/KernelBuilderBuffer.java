package jdk.incubator.vector;

import jdk.internal.vm.vector.SVMBufferSupport;

/**
    *  Representation of a SVMBuffer for the KernelBuilder class
    */
public class KernelBuilderBuffer {

    private long svmbuffer = 0;

    private final Class<?> type;

    private KernelBuilderBuffer(long svmbuffer, Class<?> type){
        this.svmbuffer = svmbuffer;
        this.type = type;
    }

    /**
     *  Get the content type of the SVMBuffer
     *  @return content type of SVMBuffer
     */
    public Class<?> getType(){
        return type;
    }

    // /**
    //     *  ABC
    //     */
    // public KernelBuilderBuffer(){}



    /**
        *  Loads a SVMBuffer from an array of type {@code float[]}
        *  @param info informations for the gpu
        *  @param array to represent by the buffer
        *  @return the SVMBuffer loaded from the array
        */
    public static KernelBuilderBuffer fromArray(GPUInformation info, float[] array){
        long svm = SVMBufferSupport.CopyFromArray(info.GetContext(), info.GetCommandQueue(), array);
        return new KernelBuilderBuffer(svm, array.getClass().getComponentType());
    }

    /**
     *  HMM
     */
    public void printType(){
        switch(this.type.toString()){
            case "float" -> System.out.println("float yee");
            default -> {}
        }
    }
}
