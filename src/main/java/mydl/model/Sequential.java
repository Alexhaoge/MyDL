package mydl.model;

import java.util.ArrayList;

import mydl.layer.Layer;
import mydl.tensor.Tensor;

/**
 * The {@code Sequetial} class is a model where all the layers stack linearly.
 */
public class Sequential extends Model{
    /**
     * ArrayList of layers
     */
    public ArrayList<Layer> layers;

    /**
     * Forward propagation of Sequential model.
     * @param inputs
     * @return output of sequential model
     */
    public Tensor forward(Tensor inputs){
        int size = layers.size();
        for(int i = 0; i < size ; i++)
            inputs = layers.get(i).forward(inputs);
        return inputs;
    }

    /**
     * Backward propagation of Sequential model.
     * @param grad 
     * @return total grad
     */
    public Tensor backward(Tensor grad){
        int size = layers.size();
        for(int i = size - 1; i >= 0 ; i++)
            grad = layers.get(i).backward(grad);
        return grad;
    }

    public void params_and_grads(){
        
    }
}