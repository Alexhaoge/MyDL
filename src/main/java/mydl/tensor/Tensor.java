package mydl.tensor;

import java.io.Serializable;

/**
 * The {@code Tensor} class is the basic datatype for neural network.
 * It is similiar to numpy.ndarray in Python.
 */
public abstract class Tensor implements Serializable, Cloneable {
    /**
     *
     */
    private static final long serialVersionUID = 1L;
    /**
     * size类型你来决定
     */
    Tensor_size size;
    /**
     * 根据size生成一个随机数的tensor
     * @param size A class contains dim and length[]
     *             dim 1,2,3 means tensor1d, tensor1d and tensor1d
     *             length[0],length[1],length[2] means rownum, colnum, N
     * @return
     */
    public static Tensor random(Tensor_size size){
        switch (size.getDim()){
            case 1:{
                int length = size.getTensor_length()[0];
                Tensor1D res = new Tensor1D( length );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = Math.random();
                }
                return res;
            }
            case 2:{
                int[] length = size.getTensor_length();
                Tensor2D res = new Tensor2D( length[0], length[1] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = Math.random();
                }
                return res;
            }
            case 3:{
                int[] length = size.getTensor_length();
                Tensor3D res = new Tensor3D( length[0], length[1], length[2]);
                for (int i = 0; i < res.darray.size(); i++){
                    for (int j = 0; j < res.darray.get( i ).data.length; j++){
                        res.darray.get( i ).data[j] = Math.random();
                    }
                }
                return res;
            }
        }
        System.err.println("Input errors!");
        return null;
    }

    /**
     * 返回一个当前Tensor的拷贝
     */
    public abstract Tensor clone();

    public abstract Tensor reshape(Tensor_size new_size);

    // return type undecided
    public abstract Object size();

    // 矩阵转置
    public abstract Tensor transpose();

    public abstract Tensor add(Tensor x);

    public abstract Tensor subtract(Tensor x);

    /**
     * this - x
     * @param x
     * @return
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

    /**
     * this / x 按位被除，请务必和divide区分开
     * @param x
     * @return
     */
    public Tensor divided(double x){ return dot_mul( 1.0/x ); };

    public Tensor divided(int x){
        return divided((double)x);
    }

    /**
     * x / this 按位除，请务必和divided区分开
     * @param x
     * @return
     */
    public abstract Tensor divide(double x);

    public Tensor divide(int x){
        return divide((double)x);
    }

    /**
     * sigmoid函数，按位求
     * @return
     */
    public abstract Tensor sigmoid();

    public Tensor tanh(){
        return sigmoid().dot_mul(2).subtract(1);
    }

    public abstract Tensor pow(double x);

    public abstract Tensor pow(int x);

    /**
     * 所有数的和
     * @return double, sum of every dimesion of this tensor
     */
    public abstract double sum();

    /**
     * 求和，压缩第axis维
     * @param axis
     * @return
     */
    public abstract Tensor sum(int axis);
    public abstract Tensor sum(int axis, int... _axis);//压缩某些维，先不用实现
}
