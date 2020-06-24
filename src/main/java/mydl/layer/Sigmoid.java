package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The sigmoid activation function.
 */
public class Sigmoid extends Activation{
    public Sigmoid() {
        super((Tensor x) -> x.sigmoid(), 
            (Tensor x) -> x.sigmoid().dot_mul(x.sigmoid().subtracted(1))
        );
    }
}