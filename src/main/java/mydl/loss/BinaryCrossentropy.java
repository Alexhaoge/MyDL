package mydl.loss;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor1D;

/**
 * The cross entropy loss function of binary classification. Input must be Tensor1D of size 2.
 * <p>{@code Loss = - Σactual·ln(predicted)}
 */
public class BinaryCrossentropy extends Loss {

    private static final long serialVersionUID = 4575604140988685172L;

    /**
     * Forward propagation.
     * 
     * @param predicted the output tensor calculated by the neural network
     * @param actual    the actual target tensor
     * @return the loss value
     * @exception IndexOutOfBoundsException if the output size of model does not
     *                                      match with the target size.
     * @exception IndexOutOfBoundsException if the output is not Tensor1D of size 2.
     * @see Loss#loss
     */
    public double loss(Tensor predicted, Tensor actual){
        if(!predicted.size().equals(actual.size()))
            throw new IndexOutOfBoundsException("The output size of the model does not match with the target size");
        else if(predicted instanceof Tensor1D && predicted.size.Tensor_length[0]==2)
            return -(actual.dot_mul(predicted.ln()).sum());
        else
            throw new IndexOutOfBoundsException("Invalid Class Number: expected 2, get "
                +Integer.toString(predicted.size.Tensor_length[0]));
    }
    
    /**
     * Backward propagation.
     * @param predicted the output tensor calculated by the neural network
     * @param actual the actual result
     * @return the gradient of the loss function
     * @exception IndexOutOfBoundsException if the output size of model does not match with the target size.
     * @exception IndexOutOfBoundsException if the output is not Tensor1D of size 2.
     * @see Loss#grad
     */
    public Tensor grad(Tensor predicted, Tensor actual){
        if(!predicted.size().equals(actual.size()))
            throw new IndexOutOfBoundsException("The output size of the model does not match with the target size");
        else if(predicted instanceof Tensor1D && predicted.size.Tensor_length[0]==2)
            return actual.divided(predicted).subtracted(0);
        else
            throw new IndexOutOfBoundsException("Invalid Class Number: expected 2, get "
                +Integer.toString(predicted.size.Tensor_length[0]));
    }

}