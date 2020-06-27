package mydl.loss;

import mydl.tensor.Tensor;

/**
 * The {@code MSE} is the mean squared error.
 * <p>{@code Loss = sum((predict - actual)^2) / size}
 */
public class MSE extends Loss{

    private static final long serialVersionUID = 1921867246675752750L;

    public double loss(Tensor predicted, Tensor actual) {
        if(predicted.size().equals(actual.size()))
            return (predicted.subtract(actual).pow(2)).sum()/predicted.total_size();
        else
            throw new IndexOutOfBoundsException("The output size of the model does not match with the target size");
    }
    
    public Tensor grad(Tensor predicted, Tensor actual){
        if(predicted.size().equals(actual.size()))
            return predicted.subtract(actual).dot_mul(2).divided(predicted.total_size());
        else
            throw new IndexOutOfBoundsException("The output size of the model does not match with the target size");
    }
}