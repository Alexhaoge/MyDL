package mydl.layer;

import mydl.tensor.Tensor;

/**
 * The ReLU activation function.<p>
 * {@code ReLU(x) = t(x > 0)}<p>
 * {@code ReLU(x) = 0(x <= 0)}
*/
public class ReLU extends Activation{
    
    private static final long serialVersionUID = 2623235547473023051L;

    private double t;

    /*
     * {@code ReLU(x) = 1(x > 0)}<p>
     * {@code ReLU(x) = 0(x <= 0)}
     */
    public ReLU() {
        t = 1.0;
    }

    /**
     * {@code ReLU(x) = t(x > 0)}<p>
     * {@code ReLU(x) = 0(x <= 0)}
     * @param _t
     */
    public ReLU(double _t) {
        t = _t;
    }

    @Override
    protected Tensor func(Tensor x) {
        return x.relu(t);
    }

    @Override
    protected Tensor derivative(Tensor x) {
        return x.DiffReLU(t);
    }

}