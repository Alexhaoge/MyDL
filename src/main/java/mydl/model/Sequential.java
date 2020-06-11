package mydl.model;

import mydl.layer.Layer;
import mydl.loss.Loss;
import mydl.optimizer.Optimizer;
import mydl.tensor.Tensor;

/**
 * The {@code Sequetial} class is a model where all the layers stack linearly.
 * It is similar to {@code tf.keras.Sequential}
 * @see <a href="https://keras.io/api/models/sequential/">https://keras.io/api/models/sequential/</a>
 */
public class Sequential extends Model{

    /**
     * Forward propagation of Sequential model.
     * @param inputs
     * @return output of sequential model
     */
    protected Tensor forward(Tensor inputs){
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
    protected Tensor backward(Tensor grad){
        int size = layers.size();
        for(int i = size - 1; i >= 0 ; i++)
            grad = layers.get(i).backward(grad);
        return grad;
    }

    public void compile(Optimizer _opt, Loss _loss){
        opt = _opt;
        loss = _loss;
    }

    /**
     * Adds a layer instance on top of the layer stack.
     * @param _layer Layer instance to add.
     */
    public void add(Layer _layer){
        layers.add(_layer);
    }

    /**
     * Removes the last layer in the model.
     * @throws IndexOutOfBoundsException if there are no layers in the model.
     */
    public void pop() throws IndexOutOfBoundsException{
        if(layers.isEmpty())
            throw new IndexOutOfBoundsException("Empty model");
        else layers.remove(layers.size()-1);
    }
}