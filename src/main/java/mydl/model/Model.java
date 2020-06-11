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
    protected Optimizer opt;
    
    /**
     * Loss function of this model
     */
    protected Loss loss;
    
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

    //Runs a single gradient update on a single batch of data.
    public void train_on_batch(ArrayList<Data> inputs, ArrayList<Data> targets){
        
    }
    
    /**
     * Trains the model for a fixed number of epochs (iterations on a dataset).
     * @param inputs {@link ArrayList} of input tensor
     * @param targets {@link ArrayList} of target tensor
     * @param epochs Positive Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. 
     * @param batch_size Positive Integer.
     * @param shuffle Boolean. Whether to shuffle the training data before each epoch.
     * @param verbose Boolean. Whether to show loss and epoch when fitting.
     * @throws IllegalArgumentException if the batch_size or epochs less than 1.
     * @throws IndexOutOfBoundsException if the input size does not match with the target size.
     * @throws IllegalStateException if the model was never compiled.
     */
    public void fit(ArrayList<Tensor> inputs, ArrayList<Tensor> targets, 
    int epochs, int batch_size, boolean shuffle, boolean verbose)
    throws IllegalArgumentException, IndexOutOfBoundsException,
    IllegalStateException {    
        //check before train
        if(opt == null || loss == null)
            throw new IllegalStateException("this model is not compiled");
        if(batch_size < 1 || epochs < 1)
            throw new IllegalArgumentException("batch size and epoch must be positive integers");
        if(inputs.size()!=targets.size())
            throw new IndexOutOfBoundsException("sample size does not match");
        //load data
        ArrayList<Data> train = Data.to_data(inputs, targets, batch_size);
        if(shuffle) Collections.shuffle(train);
        //epoch
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
     * @param inputs {@link ArrayList} of tensor. Input samples
     * @return {@link ArrayList} of tensor. Predictions
     */
    public ArrayList<Tensor> predict(ArrayList<Tensor> inputs){
        ArrayList<Tensor> results = new ArrayList<Tensor>();
        for(int i = 0; i < inputs.size(); i++)
            results.add(forward(inputs.get(i)));
        return results;
    }

    /**
     * Get the optimizer of this model.
     * @return A {@link Optimizer} instance. 
     * Note that it is not a copy but a reference.
     */
    public Optimizer get_optim(){
        return opt;
    }

    /**
     * Get the loss function of this model.
     * @return A {@link Lost} instance.
     * Note that it is not a copy but a reference.
     */
    public Loss get_loss(){
        return loss;
    }
}