package mydl.layer;

import java.util.function.Function;

import mydl.layer.Layer;
import mydl.tensor.Tensor;

/**
 * The {@code Activation} class is the abstract of all activation function 
 */
public abstract class Activation extends Layer{
    /**
     * _f is the exact function 
     */
    protected Function<Tensor, Tensor> _f;
    /**
     * _df is the derivative of {@link _f}
     */
    protected Function<Tensor, Tensor> _df;
    
    /**
     * intial constructor of activation
     * @param __f activation function, implementing {@link java.util.function.Function}
     * @param __df gradient function, implementing {@link java.util.function.Function}
     */
    Activation(Function<Tensor, Tensor> __f, Function<Tensor, Tensor> __df){
        _f = __f;
        _df = __df;
    }

    /**
     * forward propagation
     * @param inputs input tensor
     * @return output tensor
     */
    public Tensor forward(Tensor inputs){
        return _f.apply(inputs);
    }

    /**
     * Backward propagation.
     * <p>If y = f(x) and x = g(z) then dy / dz= f'(x) * g'(z)
     * @param grad input gradient
     * @return gradient with activation
     */
    public Tensor backward(Tensor grad){
        return (_df.apply(grad)).cross_mul(grad);
    }
}

