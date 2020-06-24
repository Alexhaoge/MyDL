package mydl.loss;

import mydl.tensor.Tensor;

/**
 * The cross entropy loss function for multi-class categorizaion. Input must not be Tensor of size 1.
 * <p>{@code Loss = - Σactual·ln(predicted)}
 */
public class CategoricalCrossentropy extends Loss{

    /**
     * Forward propagation.
     * @param predicted the output tensor calculated by the neural network
     * @param actual the actual target tensor
     * @return the loss value
     * @exception IndexOutOfBoundsException if the output size of model does not match with the target size.
     * @exception IndexOutOfBoundsException if the output is Tensor of size 1.
     * @see {@link Loss#loss}
     */
    public double loss(Tensor predicted, Tensor actual){
        if(predicted.total_size() == 1)
            throw new IndexOutOfBoundsException("The ouput size of Tensor should not be 1");
        else if(predicted.size().equals(actual.size()))
            return -(actual.dot_mul(predicted.ln()).sum());
        else
            throw new IndexOutOfBoundsException("The output size of the model does not match with the target size");
    }
    
    /**
     * Backward propagation.
     * @param predicted the output tensor calculated by the neural network
     * @param actual the actual result
     * @return the gradient of the loss function
     * @exception IndexOutOfBoundsException if the output size of model does not match with the target size.
     * @exception IndexOutOfBoundsException if the output is Tensor of size 1.
     * @see {@link Loss#grad}
     */
    public Tensor grad(Tensor predicted, Tensor actual){
        if(predicted.total_size() == 1)
            throw new IndexOutOfBoundsException("The ouput size of Tensor should not be 1");
        else if(predicted.size().equals(actual.size()))
            return actual.divided(predicted).subtracted(0);
        else
            throw new IndexOutOfBoundsException("The output size of the model does not match with the target size");
    }
}