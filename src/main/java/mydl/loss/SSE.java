package mydl.loss;

import mydl.tensor.Tensor;

/**
 * The {@code SSE} is the sum of the squared error.<p>
 * {@code Loss = sum((predict - actual)^2)}
 */
public class SSE extends Loss{

    /**
     * 
     */
    public double loss(Tensor predicted, Tensor actual){
        if(predicted.size().equals(actual.size()))
            return (predicted.subtract(actual).pow(2)).sum();
        else
            throw new IndexOutOfBoundsException("The output size of the model does not match with the target size");
    }
    
    /**
     * 
     */
    public Tensor grad(Tensor predicted, Tensor actual){
        return predicted.subtract(actual).dot_mul(2);
    }

}