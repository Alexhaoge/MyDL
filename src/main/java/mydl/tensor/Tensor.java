package mydl.tensor;

/**
 * The {@code Tensor} class is the basic datatype for neural network.
 * It is similiar to numpy.ndarray in Python.
 */
public abstract class Tensor {

    //public abstract Tensor(int x);

    public abstract Tensor add(Tensor x);

    public abstract Tensor subtract(Tensor x);

    /**
     * this - x
     */
    public abstract Tensor subtract(double x);

    public Tensor subtract(int x){
        return subtract((double)x);
    }

    /**
     * x - this
     */
    public abstract Tensor subtracted(double x);

    public Tensor subtracted(int x){
        return subtracted((double)x);
    }

    public abstract Tensor dot_mul(Tensor x);

    public abstract Tensor dot_mul(double x);

    /**
     * 感觉这里单独实现不复用double对效率和精度更好一些
     */
    public abstract Tensor dot_mul(int x);

    /**
     * 矩阵乘法
     */
    public abstract Tensor cross_mul(Tensor x);

    public abstract Tensor sigmoid();

    public Tensor tanh(){
        return sigmoid().dot_mul(2).subtract(1);
    }

    public abstract Tensor pow(double x);

    public abstract Tensor pow(int x);

    /**
     * return the sum of every dimension
     * @return double, sum of every dimesion of this tensor
     */
    public abstract double sum();

    public abstract Tensor reshape(int x, int y);
    public abstract Tensor reshape(int x);
}