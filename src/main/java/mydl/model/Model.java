package mydl.model;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import mydl.layer.Layer;
import mydl.loss.Loss;
import mydl.optimizer.Optimizer;
import mydl.tensor.Tensor;
import mydl.utils.Data;

/**
 * The {@code Model} class is the abstract of all model.
 * <p> Currently only {@link Sequential} model is implemented, 
 * but we also write this abstract class to declare what needs to be implement for a user-defined model class.
 */
public abstract class Model implements Serializable{
    
    private static final long serialVersionUID = -4595657931755877646L;

    /**
     * ArrayList of layers
     */
    public ArrayList<Layer> layers = new ArrayList<Layer>();
    
    /**
     * Optimizer of this model
     */
    protected Optimizer opt;
    
    /**
     * Loss function of this model
     */
    protected Loss loss;
    
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

    /**
     * Get a specific layer of this model according to the index.
     * @param index Integer.
     * @return {@link Layer} class.
     */
    public Layer get_Layer(int index){
        return layers.get(index);
    }

    /**
     * Modify a specific layer in the model
     * @param index Integer. The index of layer to modifty.
     * @param _layer {@link Layer} class. The new layer.
     */
    public void set_layer(int index, Layer _layer){
        layers.set(index, _layer);
    }

    /**
     * Forward propagation.
     * @param inputs Input tensor.
     * @return Output tensor.
     */    
    protected abstract Tensor forward(Tensor inputs);
    
    /**
     * Backward propagation.
     * @param grad Gradient tensor calculated by loss function.
     * @return Gradient of input tensor.
     */
    protected abstract Tensor backward(Tensor grad);

    /**
     * Compile this model. 
     * <p> Optimizer and loss function will be added.
     * @param _opt opimizer to add
     * @param _loss loss function to add
     * @throws RuntimeException if the tensor sizes of two adjacent layers do not fit.
     * @apiNote Tensor size check is not implemented in {@link Model} but in {@link Sequential}.
     */
    public void compile(Optimizer _opt, Loss _loss) throws RuntimeException{
        opt = _opt;
        loss = _loss;
    }

    /**
     * Set all the gradients of layers' parameters to zero.
     * Called after a mini-batch gradient update.
     * @see Model#train_on_batch
     */
    protected void clean_grad(){
        for(int i = 0; i < layers.size(); i++){
            layers.get(i).clean_grads();
        }
    }

    /**
     * Runs a single gradient update on a single batch of data.
     * @param data {@link List} of {@link Data}. The batch sample.
     * @return The total loss of the batch.
     */
    public double train_on_batch(List<Data> data){
        double batch_loss = 0 ;
        for(int i = 0; i < data.size(); i++){
            Tensor predicted = forward(data.get(i).input);
            batch_loss += loss.loss(predicted, data.get(i).target);
            Tensor grad = loss.grad(predicted, data.get(i).target);
            backward(grad);
        }
        opt.step(this, data.size());
        clean_grad();
        return batch_loss;
    }
    
    /**
     * Trains the model for a fixed number of epochs (iterations on a dataset).
     * @param inputs {@link ArrayList} of input tensor
     * @param targets {@link ArrayList} of target tensor
     * @param epochs Positive Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. 
     * @param batch_size Positive Integer. Number of samples in a mini-batch.
     * @param shuffle Boolean. Whether to shuffle the training data before fitting.
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
        ArrayList<Data> train = Data.to_data(inputs, targets, shuffle);
        //epoch
        for(int epoch = 1; epoch <= epochs; epoch++){
            double epoch_loss = 0;
            for(int i = 0; i < train.size(); i += batch_size)
                epoch_loss += train_on_batch(train.subList(i, Math.min(i+batch_size, train.size())));
            if(verbose) System.out.println("epoch="+epoch+", loss="+epoch_loss);
        }
    }

    /**
     * Fit method with training set in the form of {@link ArrayList} of {@link Data}.
     * @param train {@link ArrayList} of {@link Data}. The training set.
     * @param epochs Positive Integer. Number of epochs to train the model. An epoch is an iteration over the entire input and target data provided. 
     * @param batch_size Positive Integer. Number of samples in a mini-batch.
     * @param shuffle Boolean. Whether to shuffle the training data before fitting.
     * @param verbose Boolean. Whether to show loss and epoch when fitting.
     * @throws IllegalArgumentException if the batch_size or epochs less than 1.
     * @throws IllegalStateException if the model was never compiled.
     */
    public void fit(ArrayList<Data> train, int epochs, int batch_size, boolean shuffle, 
    boolean verbose) throws IllegalArgumentException, IllegalStateException{
        if(opt == null || loss == null)
            throw new IllegalStateException("this model is not compiled");
        if(batch_size < 1 || epochs < 1)
            throw new IllegalArgumentException("batch size and epoch must be positive integers");
        if(shuffle) Collections.shuffle(train);
        for(int epoch = 1; epoch <= epochs; epoch++){
            double epoch_loss = 0;
            for(int i = 0; i < train.size(); i += batch_size)
                epoch_loss += train_on_batch(train.subList(i, Math.min(i+batch_size, train.size())));
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
     * Save a model by {@link Serializable} interface.
     * @param filepath String. Relative or absolute path of the model file.
     */
    public void saveModel(String filepath){
        try(ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filepath))){
            out.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Model loadModel(String filepath) throws ClassNotFoundException, IOException {
        ObjectInputStream in = new ObjectInputStream(new FileInputStream(filepath));
        Model model = (Model) in.readObject();
        in.close();
        return model;
    }
}