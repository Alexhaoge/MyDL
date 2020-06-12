package mydl.layer;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor_size;

/**
 * The {@code Layer} class defines the abstract of all layer class.
 */
public abstract class Layer implements Iterable<String>{

    /**
     * A {@code Map<String, Tensor>} including all parameters of the layer.
     */
    protected Map<String, Tensor> paras = new HashMap<String, Tensor>();
    
    /**
     * A {@code Map<String, Tensor>} including all gradients of the layer.
     */
    protected Map<String, Tensor> grads = new HashMap<String, Tensor>();

    /**
     * Input tensor size, if the layer has parameters.
     */
    protected Tensor_size inSize;

    /**
     * Output tensor size, if the layer has parameters.
     */
    protected Tensor_size outSize;

    /**
     * Forward propagation, producre the output tensor corresponding to the input tensor
     * @param input the input tensor
     * @return output Tensor
     */
    public abstract Tensor forward(Tensor input);

    /**
     * Backward propagation, produce the gradient through this layer
     * @param grad the gradient tensor from last layer
     * @return the gradient tensor of this layer
     */
    public abstract Tensor backward(Tensor grad); 

    /**
     * Get a specific parameter of this layer.
     * @param name {@link String}. The name of parameter.
     * @return {@link Tensor}. 
     */
    public Tensor get_para(String name){
        return paras.get(name);
    }

    /**
     * Get the gradient of a specific parameter in this layer.
     * @param name {@link String}. The name of parameter(gradient).
     * @return {@link Tensor}. 
     */
    public Tensor get_grad(String name){
        return grads.get(name);
    }

    /**
     * Set a specific parameter of this layer.
     * @param name {@link String}. The name of parameter.
     * @param para {@link Tensor}. New parameter value.
     */
    public void set_para(String name, Tensor para){
        paras.put(name, para);
    }

    /**
     * Get the input tensor size of this layer if it has parameter.
     * @return {@link Tensor_size}
     */
    public Tensor_size getInputSize(){
        return inSize;
    }

    /**
     * Get the output tensor size of this layer if it has parameter.
     * @return {@link Tensor_size}
     */
    public Tensor_size getOutputSize(){
        return outSize;
    }

    /**
     * Implement the {@code Iterable<String>} interface.
     * This is mainly used by {@link mydl.optimizer.Optimizer}
     */
    public Iterator<String> iterator(){
        return paras.keySet().iterator();
    }
}