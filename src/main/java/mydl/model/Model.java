package mydl.model;

import java.util.ArrayList;
import java.util.Collections;

import mydl.layer.Layer;
import mydl.loss.Loss;
import mydl.optimizer.Optimizer;
import mydl.tensor.Tensor;
import mydl.utils.Data;

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
     * Compile this model. 
     * <p> Optimizer and loss function will be added while input size and 
     * output size of every two adjacent layers will be checked to see if they fit.
     * @param _opt opimizer to add
     * @param _loss loss function to add
     * @throws Exception if the tensor sizes of two adjacent layers do not fit
     */
    public abstract void compile(Optimizer _opt, Loss _loss)
        throws Exception;

    
    /**
     * 
     * @param inputs
     * @param targets
     * @param epochs
     * @param batch_size
     * @param shuffle
     * @param verbose
     * @throws IllegalArgumentException
     * @throws IndexOutOfBoundsException
     */
    public void fit(ArrayList<Tensor> inputs, ArrayList<Tensor> targets, 
    int epochs, int batch_size, boolean shuffle, boolean verbose)
    throws IllegalArgumentException, IndexOutOfBoundsException {    
        if(batch_size < 1 || epochs < 1)
            throw new IllegalArgumentException("batch size and epoch must be positive integers");
        if(inputs.size()!=targets.size())
            throw new IndexOutOfBoundsException("sample size does not match");
        
        ArrayList<Data> train = Data.to_data(inputs, targets, batch_size);
        if(shuffle) Collections.shuffle(train);
        for(int epoch = 1; epoch <= epochs; epoch++){
            double epoch_loss = 0.0;
            for(int i=0; i<batch_size ; i++){
                Tensor predicted = forward(train.get(i).input);
                epoch_loss += loss.loss(predicted, train.get(i).target);
                Tensor grad = loss.grad(predicted, train.get(i).target);
                backward(grad);
                opt.step(this);
            }
            if(verbose) System.out.println("epoch="+epoch+", loss="+epoch_loss);
        }
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