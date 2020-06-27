package mydl.layer;

import org.ejml.MatrixDimensionException;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor_size;

/**
 * The {@code Reshape} layer only reshape the input tensor.
 */
public class Reshape extends Layer {

    private static final long serialVersionUID = -4483217955150549202L;

    /**
     * Constructor.
     * @param input Input tensor shape.
     * @param output Output tensor shape.
     * @throws MatrixDimensionException if the total size of input and output does not match
     */
    public Reshape(Tensor_size input, Tensor_size output) throws MatrixDimensionException {
        if(input.total_size() != output.total_size())
            throw new MatrixDimensionException("Cannot reshape: total input size and total output size does not match");
        this.inSize = input.clone();
        this.outSize = output.clone();
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.reshape(outSize);
    }

    @Override
    public Tensor backward(Tensor grad) {
        return grad.reshape(inSize);
    }
    
}