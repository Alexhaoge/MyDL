package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The tanh activation function.
 */
public class Tanh extends Activation{

    private static final long serialVersionUID = -1539479041875751847L;

    public Tanh() {
        super((Tensor x) -> x.tanh(), 
            (Tensor x) -> (x.tanh()).pow(2).subtracted(1)       
        );
    }    
}