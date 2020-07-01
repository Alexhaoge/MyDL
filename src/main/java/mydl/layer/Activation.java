package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The {@code Activation} class is the abstract of all activation function.
 * <p>All activation function should implement {@code func} and {@derivative} method for forward and backward propagation.
 */
public abstract class Activation extends Layer{
    
    private static final long serialVersionUID = -7903859352154021415L;

    /**
     * record the input of this 
     */
    protected Tensor input;

    /**
     * The exact function of this activation.
     * @param x Input tensor.
     * @return Output tensor of this function.
     */
    protected abstract Tensor func(Tensor x);
    
    /**
     * The derivative of this function.
     * @param x
     * @return
     */
    protected abstract Tensor derivative(Tensor x);

    /**
     * forward propagation
     * @param input input tensor
     * @return output tensor
     */
    public Tensor forward(Tensor inputs){
        input = inputs;
        return func(inputs);
    }

    /**
     * Backward propagation.
     * <p>If y = f(x) and x = g(z) then dy / dz= f'(x) * g'(z)
     * @param grad input gradient
     * @return gradient with activation
     */
    public Tensor backward(Tensor grad){
        return derivative(input).dot_mul(grad);
    }

}

