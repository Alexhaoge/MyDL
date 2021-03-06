package mydl.layer;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor_size;

/**
 * The {@code Layer} class defines the abstract of all layer class.
 * <p> Because our {@link Tensor} is not as flexible as those Tensor class in Python,
 * in model training, during a single mini-batch training, the gradient inside the layers 
 * will accumulate and be set to zero after the {@link mydl.optimizer.Optimizer} update the parameters.
 * Therefore during the backward propagation, the new gradients of the parameters in
 * this layer will be added to the old gradients, not replace them. 
 */
public abstract class Layer implements Iterable<String>, Serializable{

    private static final long serialVersionUID = 1649087768762844258L;

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
     * Backward propagation, produce the gradient through this layer.
     * <p> Because our {@link Tensor} is not as flexible as those Tensor class in Python,
     * in model training, during a single mini-batch training, the gradient inside the layers 
     * will accumulate and be set to zero after the {@link mydl.optimizer.Optimizer} update the parameters.
     * Therefore during the backward propagation, the new gradients of the parameters in
     * this layer will be added to the old gradients, not replace them. 
     * <p><b>Note</b>: During the backward propagation, the new gradients of the parameters in
     * this layer will be added to the old gradients, not replace them. 
     * @param grad the gradient tensor from last layer
     * @return the gradient tensor of this layer
     * @see mydl.model.Model#train_on_batch
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
     * Set a specific gradient of this layer.
     * @param name {@link String}. The name of gradient.
     * @param grad {@link Tensor}. New gradient value.
     */
    public void set_grad(String name, Tensor grad){
        grads.put(name, grad);
    }
    
    /**
     * Set all the gradients in this layer to zero tensor.
     * This is used after the optimization of a single batch.
     * @see mydl.model.Model#train_on_batch
     */
    public void clean_grads(){
        Iterator<String> itname = grads.keySet().iterator();
        while(itname.hasNext()){
            String name = itname.next();
            grads.get(name).set_zero();
        }
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
     * Implement the {@code Iterable<String>} interface so that you
     * can use an iterator to visit all the parameters and gradients.
     */
    public Iterator<String> iterator(){
        return paras.keySet().iterator();
    }
}