package jdk.incubator.vector;

/**
 *  Class to represent the Index of a for-loop
 */
public class KernelIndex {

    private static int counter = 0;

    /**
     *  Returns the current for-loop index
     *  @return name of the index
     */
    public String getIndex(){
        return "t" + counter;
    }

    /**
     *  Create a new KernelIndex
     */
    public KernelIndex(){
        counter++;
    }
}
