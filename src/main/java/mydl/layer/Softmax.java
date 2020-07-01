package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The softmax activation function.
 */
public class Softmax extends Activation{

    private static final long serialVersionUID = 4879347903969095269L;

    @Override
    protected Tensor func(Tensor x) {
        return x.softmax();
    }

    @Override
    protected Tensor derivative(Tensor x) {
        return x.softmax().subtracted(1).dot_mul(x.softmax());
    }
}