package mydl.layer;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor_size;

/**
 * The {@code Dense} class defines a regular densely-connected NN layer,
 *  similar to Dense in Keras.
 * <p>{@code  output = dot(input, kernel) + bias}
 * <p>If the input to the layer has a rank greater than 2, 
 * then Dense computes the dot product between the input 
 * and the kernel along the last axis of the input and axis 1 of the kernel.
 * @see <a href="https://keras.io/api/layers/core_layers/dense/">https://keras.io/api/layers/core_layers/dense/</a>
 */
public class Dense extends Layer {

    private static final long serialVersionUID = 2503538020897130923L;

    /**
     * A tensor record the last input.
     */
    protected Tensor _input;

    /**
     * Constructor of Dense class.
     * @param input_size A Tensor_size object. The input tensor size.
     * @param units Positive integer. The number of units in the dense layer.
     */
    public Dense(Tensor_size input_size, int units){
        this.inSize = input_size;
        this.outSize = input_size.clone();
        outSize.Tensor_length[outSize.size-1] = units;
        paras.put("kernel", Tensor.random(new Tensor_size(inSize.Tensor_length[inSize.size-1],units)));
        paras.put("bias", Tensor.random(new Tensor_size(units)));
    }

    public Tensor forward(Tensor input){
        _input = input;//not necessary to be clone
        return input.cross_mul(paras.get("kernel")).add(paras.get("bias"));
    }

    public Tensor backward(Tensor grad){
        grads.put("kernel", grads.get("kernel").add(_input.transpose().cross_mul(grad)));
        grads.put("bias", grads.get("bias").add(grad));
        return grad.cross_mul(paras.get("kernel").transpose());
    }
}