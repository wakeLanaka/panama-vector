package jdk.incubator.vector;

/**
    abc
*/
public interface GPUSpecies {

    /**
     * abc
     * @param length the input length
     * @return the largest multiple of the vector length not greater
     *         than the given length
     */
    int loopBound(int length);

    /**
     * abc
     * @param  i abc
     * @return the number of vector lanes
     */
    int length(int i);
}
