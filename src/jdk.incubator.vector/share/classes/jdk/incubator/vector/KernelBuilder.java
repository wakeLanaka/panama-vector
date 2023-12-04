package jdk.incubator.vector;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import jdk.internal.vm.vector.SVMBufferSupport;
import jdk.incubator.vector.GPUInformation;

/**
 * How do I know if it is first operation after = ? Then the first * has to be removed
 * How do I let the user use I?
 * How do I distinguish between the types of the internal variables?
 * How do I get the length?
 * How do I know the types for the local variable?
 * How do I distinguish between array and values?
 */
public class KernelBuilder {

    // private Map<Integer, Object> objects = new HashMap<>();
    /**
     *  ABC
     */
    public LinkedHashSet<Object> objects = new LinkedHashSet<Object>();

    /**
     *  ABC
     */
    public Map<Integer, String> objectVariables = new HashMap<>();

    private static int localVariableNumber = 0;


    private long program = 0;

    private static int variableNameCounter = 0;

    /**
     *  ABC
     */
    public int length = 0;

    /**
     *  Abc
     */
    public String kernelBody = "";

    /**
     *  ABC
     */
    public String kernelHeader = "";

    /**
     *  ABC
     */
    public String kernel = "";

    /**
     *  ABC
     */
    public String localName = "";

    /**
     *  Abc
     *  @param b1 abc
     *  @param b2 abc
     *  @param size abc
     */
    public KernelBuilder(SVMBuffer b1, SVMBuffer b2, int size){
        kernel = "__kernel void exec(__global float * A, __global float * B, int k){int i = get_global_id(0);";
    }


    /**
     *  Abc
     *  @param kernelBody abc
     */
    public KernelBuilder(String kernelBody){
        KernelBuilder.variableNameCounter++;
        localName = getLocalVariableName();
        this.kernelHeader = "__kernel void exec(";
        this.kernelBody = kernelBody;
    }

    /**
     *  Abc
     *  @param kernelBody abc
     *  @param objects abc
     *  @param objectVariables abc
     */
    public KernelBuilder(String kernelBody, LinkedHashSet<Object> objects, Map<Integer, String> objectVariables){
        KernelBuilder.variableNameCounter++;
        this.objects = objects;
        this.objectVariables = objectVariables;
        localName = getLocalVariableName();
        this.kernelHeader = "__kernel void exec(";
        this.kernelBody = kernelBody;
    }

    // /**
    //  *  Abc
    //  *  @param kernelBody abc
    //  */
    // public KernelBuilder(String kernelBody){
    //     // this.variableNameCounter = 0;
    //     this.kernelHeader = "__kernel void exec(";
    //     this.kernelBody = kernelBody;
    // }

    /**
     *  ABC
     *  @return abc
     */
    public KernelBuilder Iota(){
        var kb = new KernelBuilder("");
        kb.localName = "i";
        return kb;
    }

    /**
     *  ABC
     *  @param length abc
     *  @return abc
     */
    public static KernelBuilder CreateHeader(int length){
        var kb = new KernelBuilder("");
        kb.length = length;
        return kb;
    }

    /**
     *  ABC
     *  @return abc
     */
    public static KernelBuilder CreateDef(){
        String kernelBody = "float x1 = ";
        return new KernelBuilder(kernelBody);
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @return abc
     */
    public KernelBuilder SetAdd(SVMBuffer b1){
        this.length = b1.length;
        var v1 = this.getVariableName(b1);
        if(this.program != 0){
            return this;
        }
        this.kernelBody += v1 + "[i] += ";
        return this;
    }

    /**
     *  ABC
     *  @param kb abc
     *  @return abc
     */
    public KernelBuilder Combine(KernelBuilder kb){
        if(this.program != 0){
            return this;
        }

        // this.objects.addAll(kb.objects);
        for(var obj : this.objects){
            if (obj instanceof SVMBuffer){
                this.kernelHeader += "__global float * " + kb.getVariableName(obj) + ",";
            } else {
                this.kernelHeader += "int " + kb.getVariableName(obj) + ",";
            }
        }

        this.kernelHeader += "{int i = get_global_id(0); " + this.kernelBody + ";}";
        // this.kernelHeader += "{int i = get_global_id(0); " + this.kernelBody + "; if(i == 15){float angle = i * 2.0f * 3.141592653589793f * a1/16; printf(\"angle[%d]: %.8f\\n\",i, angle);}}";
        // this.kernelHeader += "{int i = get_global_id(0); " + this.kernelBody + "; printf(\"a0: %.2f\\n\", a0[i]); printf(\"a2: %.2f\\n\", a2[i]); printf(\"a1: %d\\n\", a1);}";
        return this;
    }

    private void createHeader(){
        for(var obj : this.objects){
            switch(obj){
                case SVMBuffer buffer -> this.kernelHeader += "__global float * " + this.getVariableName(obj) + ",";
                case Float value -> this.kernelHeader += "float " + this.getVariableName(obj) + ",";
                case Integer value -> this.kernelHeader += "int " + this.getVariableName(obj) + ",";
                default -> {}
            }
        }
        this.kernelHeader = this.kernelHeader.substring(0, this.kernelHeader.length() - 1) + ")";
    }

    private void setKernelArgument(long kernel, Object object, int argumentNumber){
        switch(object){
            case SVMBuffer buffer -> SVMBufferSupport.SetKernelArgument(kernel,buffer.svmBuffer, argumentNumber);
            case Integer value -> SVMBufferSupport.SetKernelArgument(kernel, value.intValue(), argumentNumber);
            case Float value -> SVMBufferSupport.SetKernelArgument(kernel, value.floatValue(), argumentNumber);
            default -> System.out.println("Error");
        }
    }

    /**
     *  ABC
     *  @param object abc
     *  @return abc
     */
    public String getVariableName(Object object){
        String variableName = objectVariables.get(object.hashCode());

        if (variableName == null){
            variableName = "a" + variableNameCounter;
            KernelBuilder.variableNameCounter++;
            objectVariables.put(object.hashCode(), variableName);
            objects.add(object);
        }
        // System.out.println("variable: " + variableName);
        return variableName;
    }

    /**
     *  abc
     *  @return abc
     */
    public KernelBuilder Cos(){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + "cos(" + this.localName + ");";
        return kb;

        // if(this.program != 0){
        //     return this;
        // }
        // int index = this.kernelBody.indexOf('=');
        // if(index == -1){
        //     this.kernelBody = "cos(" + this.kernelBody + ")";
        // } else{
        //     this.kernelBody = this.kernelBody.substring(0,index + 1) + "cos(" + this.kernelBody.substring(index + 1) + ")";
        // }
        // return this;
    }

    /**
     *  ABC
     *  @param f1 abc
     *  @return abc
     */
    public KernelBuilder MultiplyI(float f1){
        if(this.program != 0){
            return this;
        }
        this.kernelBody += "i * " + f1;
        return this;
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @return abc
     */
    public KernelBuilder Multiply(SVMBuffer b1){
        var v1 = getVariableName(b1);
        if(this.program != 0){
            return this;
        }
        this.kernelBody += " * " + v1 + "[i]";
        return this;
    }

    /**
     *  ABC
     *  @param f1 abc
     *  @return abc
     */
    public KernelBuilder Var(float f1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(f1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + v1 + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @return abc
     */
    public KernelBuilder Var(SVMBuffer b1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(b1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + v1 + "[i];";
        return kb;
    }

    /**
     *  ABC
     *  @param i1 abc
     *  @return abc
     */
    public KernelBuilder Var(int i1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(i1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "int " + kb.localName + "=" + v1 + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @param i1 abc
     *  @return abc
     */
    public KernelBuilder Mul(SVMBuffer b1, Integer i1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(b1);
        var v2 = getVariableName(i1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + this.localName + " * " + v1 + "["+ v2 + "]" + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @param s1 abc
     *  @return abc
     */
    public KernelBuilder Mul(SVMBuffer b1, String s1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(b1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + this.localName + " * " + v1 + "["+ s1 + "]" + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @param kb1 abc
     *  @return abc
     */
    public static KernelBuilder AddAssign(SVMBuffer b1, KernelBuilder kb1){
        var kb = new KernelBuilder("", kb1.objects, kb1.objectVariables);
        var v1 = kb.getVariableName(b1);

        kb.kernelBody = kb1.kernelBody + v1 + "[i] += " + kb1.localName + ";";
        return kb;
    }

    private String getLocalVariableName(){
        var name = "l" + KernelBuilder.localVariableNumber;
        KernelBuilder.localVariableNumber++;
        return name;
    }

    /**
     *  ABC
     *  @param f1 abc
     *  @return abc
     */
    public KernelBuilder MulArr(float f1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(f1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + this.localName + " * " + v1 + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param kb1 abc
     *  @return abc
     */
    public KernelBuilder MulArr(KernelBuilder kb1){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + this.localName + " * " + kb1.localName + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @return abc
     */
    public KernelBuilder MulArr(SVMBuffer b1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(b1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + this.localName + " * " + v1 + "[i];";
        return kb;
    }

    /**
     *  ABC
     *  @param i1 abc
     *  @return abc
     */
    public KernelBuilder Mul(Integer i1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(i1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + this.localName + " * " + v1 + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param s1 abc
     *  @return abc
     */
    public KernelBuilder Mul(String s1){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + this.localName + " * " + s1 + ";";
        return kb;
    }


    /**
     *  ABC
     *  @param f1 abc
     *  @return abc
     */
    public KernelBuilder Mul(Float f1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(f1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += this.kernelBody + "float " + kb.localName + "=" + this.localName + " * " + v1 + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param f1 abc
     *  @return abc
     */
    public KernelBuilder AddArr(Float f1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(f1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += this.kernelBody + "float " + kb.localName + "=" + this.localName + " + " + v1 + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param kb1 abc
     *  @return abc
     */
    public KernelBuilder AddArr(KernelBuilder kb1){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += kb1.kernelBody + this.kernelBody + "float " + kb.localName + "=" + this.localName + " + " + kb1.localName + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param kb1 abc
     *  @return abc
     */
    public KernelBuilder SubArr(KernelBuilder kb1){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += this.kernelBody + "float " + kb.localName + "=" + this.localName + " - " + kb1.localName + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param i1 abc
     *  @return abc
     */
    public KernelBuilder Div(Integer i1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(i1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += this.kernelBody + "float " + kb.localName + "=" + this.localName + " / " + v1 + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param kb1 abc
     *  @return abc
     */
    public KernelBuilder Div(KernelBuilder kb1){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += this.kernelBody + "float " + kb.localName + "=" + this.localName + " / " + kb1.localName + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param kb1 abc
     *  @return abc
     */
    public KernelBuilder DivArr(KernelBuilder kb1){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += kb1.kernelBody + this.kernelBody + "float " + kb.localName + "=" + this.localName + " / " + kb1.localName + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param f1 abc
     *  @return abc
     */
    public KernelBuilder Div(Float f1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(f1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += this.kernelBody + "float " + kb.localName + "=" + this.localName + " / " + v1 + ";";
        return kb;
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @return abc
     */
    public KernelBuilder DivArr(SVMBuffer b1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(b1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += this.kernelBody + "float " + kb.localName + "=" + this.localName + " / " + v1 + "[i];";
        return kb;
    }

    /**
     *  ABC
     *  @param b1 abc
     *  @return abc
     */
    public KernelBuilder Div(SVMBuffer b1){
        var kb = new KernelBuilder("", objects, objectVariables);
        var v1 = getVariableName(b1);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += this.kernelBody + "float " + kb.localName + "=" + this.localName + " / " + v1 + "[i];";
        return kb;
    }

    /**
     *  ABC
     *  @return abc
     */
    public KernelBuilder LogArr(){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + "log(" + this.localName + ");";
        return kb;
    }

    /**
     *  ABC
     *  @return abc
     */
    public KernelBuilder ExpArr(){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + "exp(" + this.localName + ");";
        return kb;
    }

    /**
     *  ABC
     *  @return abc
     */
    public KernelBuilder SqrtArr(){
        var kb = new KernelBuilder("", objects, objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody = this.kernelBody + "float " + kb.localName + "=" + "sqrt(" + this.localName + ");";
        return kb;
    }

    /**
     *  ABC
     *  @param info abc
     *  @param kb abc
     */
    public void ExecKernel(GPUInformation info, KernelBuilder kb){
        long kernel = 0;
        this.objects = kb.objects;
        this.objectVariables = kb.objectVariables;

        if (this.program == 0){
            createHeader();
            String kernelString = this.kernelHeader + "{int i = get_global_id(0); " + kb.kernelBody + "}";
            // System.out.println("ExecKernel: " + kernelString);
            this.program = SVMBufferSupport.CreateProgram(info.GetContext(), kernelString);
        }

        // System.out.println("ExecKernel: " + this.objects.size());
        // var values = kb.objectVariables.values().toArray();
        // for(int i = 0; i < values.length; i++){
        //     System.out.println("val:" + values[i]);
        // }

        kernel = SVMBufferSupport.CreateKernel(this.program);
        var arr = objects.toArray();

        int argumentNumber = 0;
        for(var obj : arr){
            setKernelArgument(kernel, obj, argumentNumber);
            argumentNumber++;
        }

        objects = new LinkedHashSet<Object>();
        objectVariables = new HashMap<>();

        KernelBuilder.variableNameCounter = 0;
        KernelBuilder.localVariableNumber = 0;

        SVMBufferSupport.ExecuteKernel(kernel, info.GetCommandQueue(), this.length);
    }

    // /**
    //  *  ABC
    //  *  @return abc
    //  */
    // public KernelBuilder line2(){
    //     this.kernel += "B[i] += A[k] * cos(angle); printf(\"B[i]: %.6f\\n\", B[i]); printf(\"A[i]: %.6f\\n\", A[i]);}";
    //     return this;
    // }

    // /**
    //  *  ABC
    //  *  @param v1 abc
    //  *  @return abc
    //  */
    // public KernelBuilder MultiplyI(float v1){
    //     this.kernel += "float f1 = i * " + v1 + ";";
    //     return this;
    // }

    // /**
    //  *  ABC
    //  *  @return abc
    //  */
    // public KernelBuilder MultiplyT(){
    //     this.kernel += "float f2 = t * f1;";
    //     return this;
    // }

    // /**
    //  *  ABC
    //  *  @param v1 abc
    //  *  @return abc
    //  */
    // public KernelBuilder Divide(float v1){
    //     this.kernel += "float angle = f2 /" + v1 + ";";
    //     return this;
    // }

    // /**
    //  *  ABC
    //  *  @return abc
    //  */
    // public KernelBuilder AddSum(){
    //     this.kernel += "sum += A[t] * cos(angle);";
    //     return this;
    // }

    // /**
    //  *  ABC
    //  *  @return abc
    //  */
    // public KernelBuilder Set(){
    //     this.kernel += "B[i] = sum;";
    //     return this;
    // }

    /**
     *  ABC
     *  @param offset abc
     *  @param limit abc
     *  @param step abc
     *  @param kb1 abc
     *  @return abc
     */
    public KernelBuilder For(int offset, int limit, int step, KernelBuilder kb1){
        var kb = new KernelBuilder("", kb1.objects, kb1.objectVariables);
        if(this.program != 0){
            return this;
        }
        kb.kernelBody += "for(int t = " + offset + "; t < " + limit + "; t +=" + step + "){" + kb1.kernelBody + "}";
        return kb;
    }
}
