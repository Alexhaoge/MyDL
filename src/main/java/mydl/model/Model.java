package mydl.model;

import java.util.ArrayList;

import mydl.layer.Layer;
import mydl.loss.Loss;
import mydl.optimizer.Optimizer;
import mydl.tensor.Tensor;

/**
 * The {@code Model} class the abstract of all model.
 * <p> Currently only {@link mydl.model.Sequential} is implemented, 
 * but we also write this abstract class to declare what needs to be implement for a user-defined model class.
 */
public abstract class Model{
    
    /**
     * ArrayList of layers
     */
    public ArrayList<Layer> layers;
    
    /**
     * Optimizer of this model
     */
    public Optimizer opt;
    
    /**
     * Loss function of this model
     */
    public Loss loss;
    
    /**
     * Forward propagation
     * @param inputs input tensor
     * @return output tensor
     */
    protected abstract Tensor forward(Tensor inputs);
    
    /**
     * Backward propagation.
     * @param grad gradient tensor calculated by loss function
     * @return gradient of input tensor
     */
    protected abstract Tensor backward(Tensor grad);

    /**
     * Compile this model. Optimizer and loss function will be added while input size and output size of every two adjacent layers will be checked to see if they fit.
     * @param _opt opimizer to add
     * @param _loss loss function to add
     * @throws Exception if the tensor sizes of two adjacent layers do not fit
     */
    public abstract void compile(Optimizer _opt, Loss _loss)
        throws Exception;

    
    public void fit(ArrayList<Tensor> features,
            ArrayList<Tensor> tags, int epochs, 
            int batch_size, boolean shuffle) throws Exception{
        
    }

    /**
     * Generates output predictions for input samples.
     * @param inputs {@link ArrayList} of input samples
     * @return {@link ArrayList} of predictions
     */
    public ArrayList<Tensor> predict(ArrayList<Tensor> inputs){
        ArrayList<Tensor> results = new ArrayList<Tensor>();
        for(int i = 0; i < inputs.size(); i++)
            results.add(forward(inputs.get(i)));
        return results;
    }
}