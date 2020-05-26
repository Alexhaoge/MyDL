package mydl.utils;

import mydl.tensor.Tensor;

/**
 * An interface to represent a specific function that ouput a tensor according to the input tensor.
 * This is mainly used in activation function. You need to implement this throught a class or a lambda expression.
 */
@Deprecated
public interface func {
    Tensor call(Tensor inputs);
}