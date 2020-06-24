package mydl.loss;

import java.io.Serializable;

import mydl.tensor.Tensor;

/**
 * Class {@code Loss} is the abstract class of loss function.
 * <p>Any loss function should have a loss method to calculate the loss value,
 * and a grad method to calculate the gradient of loss function.
 */
public abstract class Loss implements Serializable{

    private static final long serialVersionUID = 2248220377984436945L;

    /**
     * Calculate the loss value for forward propagation.
     * 
     * @param predicted the output tensor calculated by the neural network
     * @param actual    the actual result
     * @return the loss value
     * @exception IndexOutOfBoundsException if the output size of model does not
     *                                      match with the target size.
     */
    public abstract double loss(Tensor predicted, Tensor actual) throws IndexOutOfBoundsException;

    /**
     * Calculate the gradient for backward propagation.
     * @param predicted the output tensor calculated by the neural network
     * @param actual the actual result
     * @return the gradient of the loss function
     * @exception IndexOutOfBoundsException if the output size of model does not match with the target size.
     */
    public abstract Tensor grad(Tensor predicted, Tensor actual) throws IndexOutOfBoundsException;

}