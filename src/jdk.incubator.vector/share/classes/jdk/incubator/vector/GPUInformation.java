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
}
