package mydl.layer;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import mydl.tensor.Tensor;

/**
 * The {@code Layer} class defines the abstract layer.
 */
public abstract class Layer implements Iterable<String>{

    /**
     * A {@code Map<String, Tensor>} including all parameters of the layer.
     */
    Map<String, Tensor> paras = new HashMap<String, Tensor>();
    
    /**
     * A {@code Map<String, Tensor>} including all gradients of the layer.
     */
    Map<String, Tensor> grads = new HashMap<String, Tensor>();

    /**
     * Forward propagation, producre the output tensor corresponding to the input tensor
     * @param inputs the input tensor
     * @return output Tensor
     */
    public abstract Tensor forward(Tensor inputs);

    /**
     * Backward propagation, produce the gradient through this layer
     * @param grad the gradient tensor from last layer
     * @return the gradient tensor of this layer
     */
    public abstract Tensor backward(Tensor grad); 

    public Tensor get_para(String name){
        return paras.get(name);
    }

    public Tensor get_grad(String name){
        return grads.get(name);
    }

    public void set_para(String name, Tensor para){
        paras.put(name, para);
    }

    public Iterator<String> iterator(){
        return paras.keySet().iterator();
    }
}