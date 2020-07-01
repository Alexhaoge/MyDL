package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The sigmoid activation function.
 */
public class Sigmoid extends Activation{

    private static final long serialVersionUID = 7217291702514777349L;

    @Override
    protected Tensor func(Tensor x) {
        return x.sigmoid();
    }

    @Override
    protected Tensor derivative(Tensor x) {
        return x.sigmoid().dot_mul(x.sigmoid().subtracted(1));
    }
}