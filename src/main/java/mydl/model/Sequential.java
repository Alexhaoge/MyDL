package mydl.model;

import mydl.layer.Layer;
import mydl.loss.Loss;
import mydl.optimizer.Optimizer;
import mydl.tensor.Tensor;
import mydl.tensor.Tensor_size;

/**
 * The {@code Sequetial} class is a model where all the layers stack linearly.
 * It is similar to {@code tf.keras.Sequential}
 * @see <a href="https://keras.io/api/models/sequential/">https://keras.io/api/models/sequential/</a>
 */
public class Sequential extends Model{

    private static final long serialVersionUID = 7590736785691222057L;

    /**
     * Forward propagation of Sequential model.
     * 
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

    /**
     * Compile this sequential model.
     * <p>Optimizer and loss function will be added while input size and output size of
     * every two adjacent layers will be checked to see if they fit.
     * @param _opt {@link Optimizer} to add.
     * @param _loss {@link Loss} function to add.
     * @throws RuntimeException if the tensor sizes of two adjacent layers do not fit.
     */
    public void compile(Optimizer _opt, Loss _loss) throws RuntimeException {
        super.compile(_opt, _loss);
        //check size
        Tensor_size lastsize = null;
        for(int i=0; i < layers.size() - 1; i++){
            if(layers.get(i).getInputSize() == null) continue;
            if(lastsize != null && lastsize.equals(layers.get(i+1).getInputSize()) != true)
                throw new RuntimeException("The output size of Layer-"+Integer.toString(i)
                    +" does not match with the input size of Layer-"+Integer.toString(i+1));
            lastsize = layers.get(i).getOutputSize();
        }
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