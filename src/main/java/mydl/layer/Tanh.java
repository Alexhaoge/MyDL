package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The tanh activation function.
 */
public class Tanh extends Activation{

    private static final long serialVersionUID = -1539479041875751847L;

    @Override
    protected Tensor func(Tensor x) {
        return x.tanh();
    }

    @Override
    protected Tensor derivative(Tensor x) {
        return (x.tanh()).pow(2).subtracted(1);
    }
}