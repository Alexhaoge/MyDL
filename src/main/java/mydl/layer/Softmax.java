package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The softmax activation function.
 */
public class Softmax extends Activation{

    private static final long serialVersionUID = 4879347903969095269L;
    
    public Softmax() {
        super((Tensor x) -> x.softmax(),
            (Tensor x) -> x.softmax().subtract(1)
        );
    }
}