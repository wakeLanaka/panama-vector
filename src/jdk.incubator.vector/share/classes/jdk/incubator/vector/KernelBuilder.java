package jdk.incubator.vector;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import jdk.incubator.vector.GPUInformation;
import jdk.internal.vm.vector.SVMBufferSupport;

/**
 *  Throw error if cos body is not correct type
 */
public class KernelBuilder {

    private static int localVariableNumber = 0;

    private static int variableNameCounter = 0;

    private Map<Object, String> objectVariables = new LinkedHashMap<>();

    /**
     *  kernel
     */
    public StringBuilder kernelString = new StringBuilder();

    private long program = 0;

    private int threads = 0;

    /**
     *  Creates a KernelBuilder
     *  @param threads to be executed on the GPU
     */
    public KernelBuilder(int threads){
        this.threads = threads;
        KernelBuilder.variableNameCounter++;
    }

    private KernelBuilder(){
        KernelBuilder.variableNameCounter++;
    }


    private KernelBuilder(Map<Object, String> objectVariables){
        KernelBuilder.variableNameCounter++;
        this.objectVariables = objectVariables;
    }


    private String getVariableName(Object object){
        String variableName = objectVariables.get(object);

        if (variableName == null){
            variableName = "a" + variableNameCounter;
            KernelBuilder.variableNameCounter++;
            objectVariables.put(object, variableName);
        }
        return variableName;
    }

    /**
     *  Get the current kernel
     *  @return current kernel
     */
    public String getKernelString(){
        return kernelString.toString();
    }

    /**
     *  Use the opencl global_id(0) variable
     *  @return new KernelBuilder
     */
    public KernelStatement Iota(){
        var statement = new KernelStatement(Integer.TYPE, this.kernelString, "i");
        return statement;
    }

    /**
     *  Creates a new variable representing the value
     *  @param value of the variable
     *  @return new KernelBuilder representing the variable
     */
    public KernelStatement Var(float value){
        var variableName = getVariableName(value);
        var statement = new KernelStatement(Float.TYPE, this.kernelString, variableName);
        return statement;
    }

    /**
     *  Creates a new variable representing the buffer element
     *  @param buffer containing the elements
     *  @return new KernelBuilder representing the buffer
     */
    public KernelStatement Var(SVMBuffer buffer){
        var bufferVariable = getVariableName(buffer);
        var rhs = bufferVariable + "[i]";
        return new KernelStatement(Float.TYPE, this.kernelString, rhs);
    }

    /**
     *  Creates a new variable representing the buffer element
     *  @param buffer containing the elements
     *  @param index of the element
     *  @return new KernelBuilder representing the buffer
     */
    public KernelStatement Var(SVMBuffer buffer, String index){
        var bufferVariable = getVariableName(buffer);
        var rhs = bufferVariable + "[" + index + "]";
        return new KernelStatement(Float.TYPE, this.kernelString, rhs);
    }

    /**
     *  Creates a new variable representing the value
     *  @param value of the variable
     *  @return new KernelBuilder representing the variable
     */
    public KernelStatement Var(int value){
        var variableName = getVariableName(value);
        var statement = new KernelStatement(Integer.TYPE, this.kernelString, variableName);
        return statement;
    }

    /**
     *  Executes the KernelBuilder on the GPU
     *  @param info of the GPU
     *  @param kernelBuilder to be executed on the GPU 
     */
    public void ExecKernel(GPUInformation info, KernelBuilder kernelBuilder){
        ExecKernel(info, kernelBuilder.kernelString);
    }

    /**
     *  Executes the kernelString on the GPU
     *  @param info of the GPU
     *  @param kernelString to be executed on the GPU
     */
    public void ExecKernel(GPUInformation info, StringBuilder kernelString){
        if (this.program == 0){
            createKernel(kernelString, this.objectVariables.keySet());
            this.program = SVMBufferSupport.CreateProgram(info.GetContext(), kernelString.toString());
        }

        long kernel = SVMBufferSupport.CreateKernel(program);

        setKernelArguments(kernel, this.objectVariables.keySet());

        resetKernelBuilder();

        SVMBufferSupport.ExecuteKernel(kernel, info.GetCommandQueue(), this.threads);
    }

    private void createKernel(StringBuilder str, Set<Object> variables){
        str.insert(0, "{int i = get_global_id(0); ");
        str.insert(0, createKernelSignature(this.objectVariables.keySet()));
        str.append("}");
    }

    private String createKernelSignature(Set<Object> arguments){
        String kernelSignature = "__kernel void exec(";
        for(var obj : arguments){
            switch(obj){
                case SVMBuffer buffer -> kernelSignature += "__global float * " + this.getVariableName(obj) + ",";
                case Float value -> kernelSignature += "float " + this.getVariableName(obj) + ",";
                case Integer value -> kernelSignature += "int " + this.getVariableName(obj) + ",";
                default -> throw new AssertionError("Object is not of a supported type");
            }
        }
        kernelSignature = kernelSignature.substring(0, kernelSignature.length() - 1) + ")";
        return kernelSignature;
    }

    private void setKernelArguments(long kernel, Set<Object> arguments){
        int argumentNumber = 0;
        for(var obj : arguments){
            setKernelArgument(kernel, obj, argumentNumber);
            argumentNumber++;
        }
    }

    private void setKernelArgument(long kernel, Object object, int argumentNumber){
        switch(object){
            case SVMBuffer buffer -> { SVMBufferSupport.SetKernelArgument(kernel, buffer.svmBuffer, argumentNumber);}
            case Integer value -> { SVMBufferSupport.SetKernelArgument(kernel, value.intValue(), argumentNumber);}
            case Float value -> { SVMBufferSupport.SetKernelArgument(kernel, value.floatValue(), argumentNumber);}
            default -> throw new AssertionError("Object is not of a supported type");
        }
    }

    private void resetKernelBuilder(){
        objectVariables = new LinkedHashMap<>();
        KernelBuilder.variableNameCounter = 0;
        KernelBuilder.localVariableNumber = 0;
    }

    /**
    *  Assigns builder to buffer
    *  @param buffer to be set
    *  @param builder right hand side of the addition assignment operator
    *  @return new KernelStatement representing the assignment operator
    */
    public KernelBuilder Assign(SVMBuffer buffer, KernelStatement builder){
        var bufferVariable = getVariableName(buffer);
        this.kernelString.append(bufferVariable + "[i] = " + builder.localName + ";");
        return this;
    }

    /**
    *  Adds and Assigns builder to buffer
    *  @param buffer to be set
    *  @param builder right hand side of the addition assignment operator
    */
    public void AddAssign(SVMBuffer buffer, KernelStatement builder){
        var bufferVariable = getVariableName(buffer);
        this.kernelString.append(bufferVariable);
        this.kernelString.append("[i] += ");
        this.kernelString.append(builder.localName);
        this.kernelString.append(";");
    }

    /**
     *  Representation of a single statement in the kernel
     */
    public class KernelStatement {

        private final String localName;

        private StringBuilder kernelString;

        private final Class<?> type;

        private Class<?> getType(Class<?> a, Class<?> b){
            if (a == Double.TYPE || b == Double.TYPE) {
                return Double.TYPE;
            } else if (a == Float.TYPE || b == Float.TYPE) {
                return Float.TYPE;
            } else if(a == Long.TYPE || b == Long.TYPE){
                return Long.TYPE;
            } else if(a == Integer.TYPE || b == Integer.TYPE){
                return Integer.TYPE;
            }
            return Short.TYPE;
        }

        /**
         *  KernelStatement constructor
         *  @param type of the KernelStatement
         *  @param kernelString string of the current kernel
         *  @param rhs of the statement
         */
        public KernelStatement(Class<?> type, StringBuilder kernelString, String rhs){
            this.type = type;
            this.localName = getLocalVariableName();
            this.kernelString = kernelString;
            kernelString.append(type.toString() + " " + this.localName + " = " + rhs + ";");
        }

        private String getLocalVariableName(){
            var name = "l" + KernelBuilder.localVariableNumber;
            KernelBuilder.localVariableNumber++;
            return name;
        }

        /**
        *  Multiplies this with the buffer element
        *  @param buffer containing the values
        *  @return new KernelStatement representing the multiplication
        */
        public KernelStatement Mul(SVMBuffer buffer){
            var bufferVariable = getVariableName(buffer);
            var rhs = this.localName + " * " + bufferVariable + "[i]";
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Multiplies this with the buffer element
        *  @param buffer containing the values
        *  @param index accesing the buffer
        *  @return new KernelStatement representing the multiplication
        */
        public KernelStatement Mul(SVMBuffer buffer, Integer index){
            var bufferVariable = getVariableName(buffer);
            var indexVariable = getVariableName(index);
            var rhs = this.localName + " * " + bufferVariable + "["+ indexVariable + "]";
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Multiplies this with the buffer element
        *  @param buffer containing the values
        *  @param index accessing the buffer
        *  @return new KernelStatement representing the multiplication
        */
        public KernelStatement Mul(SVMBuffer buffer, KernelIndex index){
            var bufferVariable = getVariableName(buffer);
            var rhs = this.localName + " * " + bufferVariable + "["+ index.getIndex() + "]";
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Multiplies this with the buffer element
        *  @param buffer containing the values
        *  @param index accessing the buffer
        *  @return new KernelStatement representing the multiplication
        */
        public KernelStatement Mul(SVMBuffer buffer, String index){
            var bufferVariable = getVariableName(buffer);
            var rhs = this.localName + " * " + bufferVariable + "["+ index + "]";
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Multiplies this with the KernelIndex
        *  @param value to be multiplied
        *  @return new KernelStatement representing the multiplication
        */
        public KernelStatement Mul(KernelIndex value){
            var resultType = getType(this.type, Integer.TYPE);
            var rhs = this.localName + " * " + value.getIndex();
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Multiplies this with the value
        *  @param value to be multiplied
        *  @return new KernelStatement representing the multiplication
        */
        public KernelStatement Mul(Integer value){
            var intVariable = getVariableName(value);
            var rhs = this.localName + " * " + intVariable;
            var resultType = getType(this.type, Integer.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Multiplies this with the value
        *  @param value to be multiplied
        *  @return new KernelStatement representing the multiplication
        */
        public KernelStatement Mul(Float value){
            var floatVariable = getVariableName(value);
            var rhs = this.localName + " * " + floatVariable;
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Multiplies this with the value
        *  @param value to be multiplied
        *  @return new KernelStatement representing the multiplication
        */
        public KernelStatement Mul(KernelStatement value){
            var rhs = this.localName + " * " + value.localName;
            var resultType = getType(this.type, value.type);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Divides this with the value
        *  @param value to be divided
        *  @return new KernelStatement representing the division
        */
        public KernelStatement Div(Integer value){
            var intVariable = getVariableName(value);
            var rhs = this.localName + " / " + intVariable;
            var resultType = getType(this.type, Integer.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Divides this with the value
        *  @param value to be divided
        *  @return new KernelStatement representing the division
        */
        public KernelStatement Div(Float value){
            var floatVariable = getVariableName(value);
            var rhs = this.localName + " / " + floatVariable;
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Divides this with the value
        *  @param value to be divided
        *  @return new KernelStatement representing the division
        */
        public KernelStatement Div(KernelStatement value){
            var rhs = this.localName + " / " + value.localName;
            var resultType = getType(this.type, value.type);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Divides this with the value
        *  @param value to be divided
        *  @return new KernelStatement representing the division
        */
        public KernelStatement Div(SVMBuffer value){
            var bufferVariable = getVariableName(value);
            var rhs = this.localName + " / " + bufferVariable + "[i]";
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Adds value to this
        *  @param value to be added
        *  @return new KernelStatement representing the addition
        */
        public KernelStatement Add(Float value){
            var floatVariable = getVariableName(value);
            var rhs = this.localName + " + " + floatVariable;
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Adds value to this
        *  @param value to be added
        *  @return new KernelStatement representing the addition
        */
        public KernelStatement Add(Integer value){
            var floatVariable = getVariableName(value);
            var rhs = this.localName + " + " + floatVariable;
            var resultType = getType(this.type, Integer.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Adds value to this
        *  @param value to be added
        *  @return new KernelStatement representing the addition
        */
        public KernelStatement Add(SVMBuffer value){
            var bufferVariable = getVariableName(value);
            var rhs = this.localName + " + " + bufferVariable + "[i]";
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Adds value to this
        *  @param value to be added
        *  @return new KernelStatement representing the addition
        */
        public KernelStatement Add(KernelStatement value){
            var rhs = this.localName + " + " + value.localName;
            var resultType = getType(this.type, value.type);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Subtracts value from this
        *  @param value to be subtracted
        *  @return new KernelStatement representing the subtraction
        */
        public KernelStatement Sub(Float value){
            var floatVariable = getVariableName(value);
            var rhs = this.localName + " - " + floatVariable;
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Subtracts value from this
        *  @param value to be subtracted
        *  @return new KernelStatement representing the subtraction
        */
        public KernelStatement Sub(Integer value){
            var floatVariable = getVariableName(value);
            var rhs = this.localName + " - " + floatVariable;
            var resultType = getType(this.type, Integer.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Subtracts value from this
        *  @param value to be subtracted
        *  @return new KernelStatement representing the subtraction
        */
        public KernelStatement Sub(SVMBuffer value){
            var bufferVariable = getVariableName(value);
            var rhs = this.localName + " - " + bufferVariable + "[i]";
            var resultType = getType(this.type, Float.TYPE);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        /**
        *  Subtracts value from this
        *  @param value to be subtracted
        *  @return new KernelStatement representing the subtraction
        */
        public KernelStatement Sub(KernelStatement value){
            var rhs = this.localName + " - " + value.localName;
            var resultType = getType(this.type, value.type);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }


        /**
        *  Calculates the Cosinus of this
        *  @return new KernelStatement representing the result of the cosinus
        */
        public KernelStatement Cos(){
            var rhs = "cos(" + this.localName + ")";
            return new KernelStatement(Float.TYPE, this.kernelString, rhs);
        }

        /**
        *  Calculates the Sinus of this
        *  @return new KernelStatement representing the result of the sinus
        */
        public KernelStatement Sin(){
            var rhs = "sin(" + this.localName + ")";
            return new KernelStatement(Float.TYPE, this.kernelString, rhs);
        }

        /**
        *  Calculates the logarithm of this
        *  @return new KernelStatement representing the result of the logarithm
        */
        public KernelStatement Log(){
            var rhs = "log(" + this.localName + ")";
            return new KernelStatement(Float.TYPE, this.kernelString, rhs);
        }

        /**
        *  Calculates the exponential function of this
        *  @return new KernelStatement representing the result of the exponential function
        */
        public KernelStatement Exp(){
            var rhs = "exp(" + this.localName + ")";
            return new KernelStatement(Float.TYPE, this.kernelString, rhs);
        }

        /**
        *  Calculates the square-root of this
        *  @return new KernelStatement representing the result of the square-root
        */
        public KernelStatement Sqrt(){
            var rhs = "sqrt(" + this.localName + ")";
            return new KernelStatement(Float.TYPE, this.kernelString, rhs);
        }

        /**
        *  Calculates the absolut value of this
        *  @return new KernelStatement representing the absolut value
        */
        public KernelStatement Abs(){
            var rhs = "fabs(" + this.localName + ")";
            return new KernelStatement(Float.TYPE, this.kernelString, rhs);
        }

        /**
        *  Compares this with the value
        *  @param value to be compared against
        *  @return new KernelStatement representing comparison
        */
        public KernelStatement CompareGT(float value){
            var floatVariable = getVariableName(value);
            var rhs = "0.0f; if(" + this.localName + " > " + floatVariable + "){";
            var statement = new KernelStatement(Float.TYPE, this.kernelString, rhs);
            statement.kernelString.append(statement.localName + "= 1.0f;}");
            return statement;
        }

        /**
         *  Returns the linear blend of x (this) and y and a implemented as: x + (y - x) * a
         *  @param value y
         *  @param mask a
         *  @return new KernelStatement representing the opencl mix function
         */
        public KernelStatement Blend(KernelStatement value, KernelStatement mask){
            var rhs = "mix(" + this.localName + ", " + value.localName + ", " + mask.localName + ")";
            var resultType = getType(this.type, value.type);
            return new KernelStatement(resultType, this.kernelString, rhs);
        }

        @Override
        public String toString(){
            return this.localName + ": " + this.kernelString.toString();
        }
    }
}
