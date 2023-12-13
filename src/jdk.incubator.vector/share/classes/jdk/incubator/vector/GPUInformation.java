package jdk.incubator.vector;

/**
 *  Interface for managing different gpu frameworks / libraries
 */
public interface GPUInformation {

    /**
     *  Deallocates the context
     */
    public void ReleaseContext();

    /**
     *  Deallocates the command queue
     */
    public void ReleaseCommandQueue();

    /**
     *  Deallocates the device
     */
    public void ReleaseDevice();

    /**
     *  Get the current context
     *  @return the current context
     */
    public long GetContext();

    /**
     *  Get current command queue
     *  @return the current command queue
     */
    public long GetCommandQueue();

    /**
     *  Get the current program
     *  @return the current program
     */
    public long GetProgram();

    /**
     *  Get the maximal work group size
     *  @return the maximal work group size for this device
     */
    public int GetMaxWorkGroupSize();
}
