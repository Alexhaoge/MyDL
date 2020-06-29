package mydl.layer;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor_size;

/**
 * The {@code Dense} class defines a regular densely-connected NN layer.
 * <p>{@code  output = input x kernel + bias}
 * <p>If the input to the layer has a dimensionality greater than 2, then Dense 
 * computes the matrix multiplication between the input and the kernel along 
 * the last dimension of the {@code input} and the second last dimension of 
 * the {@code kernel}.
 * <p>For example, if the {@code input} tensor shape is {@code Tensor_size(3, 4, 5)} and 
 * {@code units=2}, then the {@code kernel} tensor shape is {@code Tensor_size(3, 5, 2)}
 * and the {@code bias} and the {@code output} tensor shape is {@code Tensor_size(3, 4, 2)}.
 * 
 * @apiNote The rules for caculating high dimension tensor is different from keras, 
 * which is more rather like Pytorch. It is awful for mixing up different styles, 
 * but this is only a pre-release. We will decide a more stable api in formal release.
 * @version v1.0-alpha 
 */
public class Dense extends Layer {

    private static final long serialVersionUID = 2503538020897130923L;

    /**
     * A tensor record the last input.
     */
    protected Tensor _input;

    /**
     * Constructor of Dense class. The ouput size is same to the input size 
     * except the last dimension, and size of last dimension is {@code units}.
     * @param input_size A Tensor_size object. The input tensor size.
     * @param units Positive integer. The number of units in the dense layer.
     */
    public Dense(Tensor_size input_size, int units){
        this.inSize = input_size.clone();
        this.outSize = inSize.clone();
        this.outSize.Tensor_length[outSize.size-1] = units;
        Tensor_size kernel = inSize.clone();
        if(kernel.size == 1){
            kernel.size = 2;
            kernel.Tensor_length[1] = units;
        }else{
            kernel.Tensor_length[kernel.size-2] = kernel.Tensor_length[kernel.size-1];
            kernel.Tensor_length[kernel.size-1] = units;
        }
        paras.put("kernel", Tensor.random(kernel));
        paras.put("bias", Tensor.random(new Tensor_size(outSize)));
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