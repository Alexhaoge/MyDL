package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The sigmoid activation function.
 */
public class Sigmoid extends Activation{

    private static final long serialVersionUID = 7217291702514777349L;

    public Sigmoid() {
        super((Tensor x) -> x.sigmoid(), 
            (Tensor x) -> x.sigmoid().dot_mul(x.sigmoid().subtracted(1))
        );
    }
}