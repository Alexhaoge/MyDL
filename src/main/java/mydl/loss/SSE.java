package mydl.loss;

import mydl.tensor.Tensor;

/**
 * The {@code SSE} is the sum of the squared error.
 * MSE Loss = sum((predict - actual)^2)
 */
public class SSE extends Loss{

    /**
     * 
     */
    public double loss(Tensor predicted, Tensor actual){
        return (predicted.subtract(actual).pow(2)).sum();
    }
    
    /**
     * 
     */
    public Tensor grad(Tensor predicted, Tensor actual){
        return predicted.subtract(actual).dot_mul(2);
    }

}