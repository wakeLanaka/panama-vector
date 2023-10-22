package jdk.incubator.vector;


import jdk.internal.vm.vector.GPUSupport;
import jdk.incubator.vector.GPUSpecies;
import java.util.function.*;

import static jdk.internal.vm.vector.VectorSupport.*;

/**
    abcd
*/
public class GPUVector {

    /**
        abc
    */
    public int MemObjectNumber = 0;

    /**
        abc
    */
    public int length = 0;

    /**
        abc
    */
    private int offset = 0;


    /**
        abc
    */
    private GPUVector(int memObjectNumber) {
        this.MemObjectNumber = memObjectNumber;
    }

    // /**
    //     abc
    // */
    // public GPUVector() {}
    /**
        abc
        @return abc
    */
    public static int[] test() {
        return GPUSupport.test();
    }

    /**
        abc
        @param species asdf
        @param array asdf 
        @param offset asdf
        @return asdf
    */
    public static GPUVector fromArray(GPUSpecies species, int[] array, int offset) {
        int memObjectNumber = GPUSupport.allocateArray(array);
        var gpuVector = new GPUVector(memObjectNumber);
        gpuVector.offset = offset;
        gpuVector.length = array.length;
        return gpuVector;
    }

    /**
        abc
        @param vector abc
        @return abc
    */
    public GPUVector add(GPUVector vector) {
        var memObjectNumber = GPUSupport.add(this.MemObjectNumber, vector.MemObjectNumber, this.offset, this.length, IntGPUSpecies.GetMaxGroupSize());
        return new GPUVector(memObjectNumber);
    }

    /**
        abc
        @param array abc
        @param i abc
    */
    public void intoArray(int[] array, int i) {
        GPUSupport.intoArray(this.MemObjectNumber, array.length, array);
    }

    /**
        abc
        @param a abc
        @param b abc
        @param c abc
    */
    public static void gpuAddition(int[] a, int[] b, int[] c) {
        GPUSupport.gpuAddition(a, b, c);
    }

    /**
        abc
        @param a abc
        @param b abc
        @param c abc
    */
    public static void gpuAdditionHostPtr(int[] a, int[] b, int[] c) {
        GPUSupport.gpuAdditionHostPtr(a, b, c);
    }

    /**
        abc
        @param a abc
        @param b abc
        @param c abc
        @param n abc
    */
    public static void gpuMatrix(float[] a, float[] b, float[] c, int n) {
        GPUSupport.gpuMatrix(a, b, c, n);
    }

    /**
        abc
    */
    static final class IntGPUSpecies implements GPUSpecies  {

        static private int arrayLength = 0;
        static private int maxGroupSize = 0;

        private IntGPUSpecies() {
            IntGPUSpecies.maxGroupSize = GPUSupport.initializeGPU();
        }

        /**
            abc
        */
        public int loopBound(int length) {
            IntGPUSpecies.arrayLength = length;
            return length;
        }

        /**
            abc
        */
        public int length(int i) {
            return IntGPUSpecies.arrayLength - i;
        }


        /**
            abc
            @return abc
        */
        public static int GetMaxGroupSize() {
            return IntGPUSpecies.maxGroupSize;
        }
    }

    /**
        abc
    */
    public static final GPUSpecies SPECIES_PREFERRED
        = new IntGPUSpecies();
}
