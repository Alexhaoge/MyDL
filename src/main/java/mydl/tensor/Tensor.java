package mydl.tensor;
import org.ejml.MatrixDimensionException;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * The {@code Tensor} class is the basic datatype for neural network.
 * It is similiar to numpy.ndarray in Python.
 */
public abstract class Tensor implements Serializable{

    private static final long serialVersionUID = -5694231439125166069L;

    /**
     * size类型你来决定
     */
    public Tensor_size size;

    /**
     * Generate a tensor with pseudorandom, uniformly distributed
     * {@code double} value between {@code floor} and {@code cell}.
     * @param _size {@link Tensor_size}. The shape of the tensor.
     * @param floor Upper
     * @param ceil 均匀随机分布上确界
     * @return
     */
    public static Tensor random_uniform(Tensor_size _size, double floor, double ceil){
        Random ran = new Random();
        switch (_size.getSize()){
            case 1:{
                int length = _size.getTensor_length()[0];
                Tensor1D res = new Tensor1D( length );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextDouble()*(ceil-floor)+floor;
                }
                return res;
            }
            case 2:{
                int[] length = _size.getTensor_length();
                Tensor2D res = new Tensor2D( length[0], length[1] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextDouble()*(ceil-floor)+floor;
                }
                return res;
            }
            case 3:{
                int[] length = _size.getTensor_length();
                Tensor3D res = new Tensor3D( length[0], length[1], length[2]);
                for (int i = 0; i < res.darray.size(); i++){
                    for (int j = 0; j < res.darray.get( i ).data.length; j++){
                        res.darray.get( i ).data[j] = ran.nextDouble()*(ceil-floor)+floor;
                    }
                }
                return res;
            }
            default:
                throw new MatrixDimensionException("Input tensor length error.");
        }
    }

    /**
     * Generate a tensor with pseudorandom, uniformly distributed
     * {@code double} value between {@code 0.0} and {@code 1.0}.
     * @param _size {@link Tensor_size}. The shape of the tensor.
     * @return A random tensor.
     */
    public static Tensor random(Tensor_size _size){
        Random ran = new Random();
        switch (_size.getSize()){
            case 1:{
                int length = _size.getTensor_length()[0];
                Tensor1D res = new Tensor1D( length );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextDouble();
                }
                return res;
            }
            case 2:{
                int[] length = _size.getTensor_length();
                Tensor2D res = new Tensor2D( length[0], length[1] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextDouble();
                }
                return res;
            }
            case 3:{
                int[] length = _size.getTensor_length();
                Tensor3D res = new Tensor3D( length[0], length[1], length[2]);
                for (int i = 0; i < res.darray.size(); i++){
                    for (int j = 0; j < res.darray.get( i ).data.length; j++){
                        res.darray.get( i ).data[j] = ran.nextDouble();
                    }
                }
                return res;
            }
            default:
                throw new MatrixDimensionException("Input tensor length error.");
        }
    }

    /**
     * Generate a tensor with random number by normal distribution{@code N(mean,std)}.
     * @param _size {@link Tensor_size}. The shape of the tensor.
     * @param mean Mean of the normal distribution.
     * @param std Standard deviation of the normal distribution.
     * @return A random tensor.
     */
    public static Tensor random_normdist(Tensor_size _size, double mean, double std){
        Random ran = new Random();
        switch (_size.getSize()){
            case 1:{
                int length = _size.getTensor_length()[0];
                Tensor1D res = new Tensor1D( length );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextGaussian() * std + mean;
                }
                return res;
            }
            case 2:{
                int[] length = _size.getTensor_length();
                Tensor2D res = new Tensor2D( length[0], length[1] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextGaussian() * std + mean;
                }
                return res;
            }
            case 3:{
                int[] length = _size.getTensor_length();
                Tensor3D res = new Tensor3D( length[0], length[1], length[2]);
                for (int i = 0; i < res.darray.size(); i++){
                    for (int j = 0; j < res.darray.get(i).data.length; j++){
                        res.darray.get(i).data[j] = ran.nextGaussian() * std + mean;
                    }
                }
                return res;
            }
            default:
                throw new MatrixDimensionException("Input tensor length error.");
        }
    }

    /**
     * Generate a tensor with random number by standard normal distribution{@code N(0,1)}.
     * @param _size {@link Tensor_size}. The shape of the tensor.
     * @return A random tensor.
     */
    public static Tensor random_normdist(Tensor_size _size){
        return Tensor.random_normdist(_size, 0, 1);
    }

    /**
     * Generate a zero tensor.
     * @param _size {@link Tensor_size}. The shape of the tensor.
     * @return A zero tensor.
     */
    public static Tensor zero(Tensor_size _size){
        switch (_size.getSize()){
            case 1:{
                int length = _size.getTensor_length()[0];
                Tensor1D res = new Tensor1D( length );
                res.darray.zero();
                return res;
            }
            case 2:{
                int[] length = _size.getTensor_length();
                Tensor2D res = new Tensor2D( length[0], length[1] );
                res.darray.zero();
                return res;
            }
            case 3:{
                int[] length = _size.getTensor_length();
                Tensor3D res = new Tensor3D( length[0], length[1], length[2]);
                for (int i = 0; i < res.darray.size(); i++){
                    res.darray.get(i).zero();
                }
                return res;
            }
            default:
                throw new MatrixDimensionException("Input tensor length error.");
        }

    }

    /**
     * 把一个tensor清零，且不新建tensor直接返回当前的
     * @return this tensor
     */
    public abstract Tensor set_zero();

    /**
     * Clone. Return a deep copy of the current Tensor.
     */
    public abstract Tensor clone();

    public abstract Tensor reshape(Tensor_size new_size);

    /**
     * Get the shape of this tensor.
     * @return A {@link Tensor_size} object indicating the shape of tensor.
     */
    public abstract Tensor_size size();

    public int total_size(){
        int _total = 1;
        for(int i = 0; i < this.size.getSize(); i++)
            _total *= this.size.Tensor_length[i];
        return _total;
    }

    /**
     * Res = this^T
     * @return tensor
     */
    public abstract Tensor transpose();

    /**
     * Res_{i} = this_{i} + t2_{i}
     * @param t2 addend, tensor
     * @return tensor Res
     */
    public abstract Tensor add(Tensor t2);

    /**
     * Res_{i} = this_{i} - t2_{i}
     * @param t2 s sbtracted, tensor
     * @return tensor Res
     */
    public abstract Tensor subtract(Tensor t2);

    /**
     * Res_{i} = this_{i} - x
     * @param x minus, double variable
     * @return tensor Res
     */
    public abstract Tensor subtract(double x);

    /**
     * Res_{i} = this_{i} - x
     * @param x minus, int variable
     * @return tensor Res
     */
    public Tensor subtract(int x){
        return subtract((double)x);
    }

    /**
     * Res = x - this_{i}
     * @param x subtracted, double variable
     * @return tensor Res
     */
    public abstract Tensor subtracted(double x);

    /**
     * Res = x - this_{i}
     * @param x subtracted, int variable
     * @return tensor res
     */
    public Tensor subtracted(int x){
        return subtracted((double)x);
    }

    /**
     * Res_{i} = this_{i}*t1_{i}
     * @param t1 multiplier, tensor
     * @return tensor Res
     */
    public abstract Tensor dot_mul(Tensor t1);


    /**
     * Res_{i} = this_{i}*x
     * @param x multiplier, double
     * @return tensor
     */
    public abstract Tensor dot_mul(double x);

    /**
     * Res_{i} = this_{i}*x
     * @param x multiplier, int
     * @return tensor
     */
    public abstract Tensor dot_mul(int x);

    /**
     * Matrix multiplication of {@code this} and {@code x}.
     * @param x the tensor to be multiplied.
     * @throws MatrixDimensionException if this tensor does not
     * have compactible shape for matrix multiply with {@code x}.
     * @apiNote Matrix multiplication does not satisfy the commutative law,
     * so {@code a.cross_mul(b)} is different from {@code b.cross_mul(a)}.
     */
    public abstract Tensor cross_mul(Tensor x) throws MatrixDimensionException;

    /**
     * Res_{i} = this_{i} / x
     * @param x divisor, double
     * @return tensor
     */
    public Tensor divided(double x){ return dot_mul( 1.0/x ); };

    /**
     * Res_{i} = this_{i} / x
     * @param x divisor, int
     * @return tensor
     */
    public Tensor divided(int x){
        return divided((double)x);
    }

    /**
     * Res_{i} = this_{i} / x_{i}
     * @param x divisor, tensor
     * @return tensor
     */
    public abstract Tensor divided(Tensor x);

    /**
     * Res_{i} = x / this_{i}
     * @param x dividend, double
     * @return tensor
     */
    public abstract Tensor divide(double x);

    /**
     * Res_{i} = x / this_{i}
     * @param x dividend, int
     * @return ternsor
     */
    public Tensor divide(int x){
        return divide((double)x);
    }

    /**
     * Res_{i} = sigmoid(this_{i})
     * @return tensor
     */
    public abstract Tensor sigmoid();

    /**
     * Res_{i} = tanh(this_{i})
     * @return tensor
     */
    public Tensor tanh(){
        return sigmoid().dot_mul(2).subtract(1);
    }

    /**
     * Res_{i} = Relu(this_{i}, lambda = t)
     * @param t lambda, double
     * @return tensor
     */
    public abstract Tensor relu(double t);

    /**
     * Res_{i} = Relu(this_{i}, lambda = 1)
     * @return tensor
     */
    public Tensor relu(){
        return this.relu(1.0);
    }

    /**
     * Res_{i} = this_{i}^x
     * @param x, double
     * @return tensor
     */
    public abstract Tensor pow(double x);

    /**
     * Res_{i} = this_{i}^x
     * @param x, int
     * @return tensor
     */
    public abstract Tensor pow(int x);

    /**
     * Res_{i} = log_e (this_{i})
     * @return tensor
     */
    public abstract Tensor ln();

    /**
     * Res = \sum{Res_{i}}
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

    /**
     * Res_{i} = sgn(this_{i})
     * @return tensor
     */
    public abstract Tensor sgn();

    /**
     * Res_{i} = t(this_{i} > 0)
     * Res_{i} = 0(this_{i} <= 0)
     * @param t double
     * @return
     */
    public abstract Tensor DiffReLU(double t);
    public Tensor DiffReLU(){
        return this.DiffReLU(1.0);
    }

    /**
     * Res_{i} = softmax(this_{i})
     * @return tensor
     */
    public abstract Tensor softmax();
    public boolean equals(Object obj) {
        if (obj instanceof Tensor) {
            if (obj instanceof Tensor1D && this instanceof Tensor1D) {
                if (((Tensor1D) obj).size() == this.size()) {
                    return (Arrays.equals(((Tensor1D) this).getData().data, ((Tensor1D) obj).getData().data));
                }
            }else if (obj instanceof Tensor2D && this instanceof Tensor2D) {
                if (((Tensor2D) obj).size() == this.size()) {
                    return (Arrays.equals(((Tensor2D) this).getData().data, ((Tensor2D) obj).getData().data));
                }
            }else if (obj instanceof Tensor3D && this instanceof Tensor3D) {
                if (((Tensor3D) obj).size() == this.size()) {
                    boolean temp = true;
                    for (int i = 0; i < ((Tensor3D) this).darray.size() && temp == true; i++) {
                        temp = (temp && Arrays.equals( ((Tensor3D) this).darray.get(i).data , ((Tensor3D) obj).darray.get( i ).data ));
                    }
                    return temp;
                }
            }
        }
        return false;
    }
}
