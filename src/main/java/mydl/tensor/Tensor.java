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
     * @return An uniformly distributed tensor.
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
     * Set this tensor to zero.
     * @return This tensor. Modified
     */
    public abstract Tensor set_zero();

    /**
     * Clone. Return a deep copy of the current Tensor.
     */
    public abstract Tensor clone();

    /**
     * Reshape this tensor.
     * @param new_size The new {@code Tensor_size} of this tensor. 
     * @return a new Tensor.
     */
    public abstract Tensor reshape(Tensor_size new_size);

    /**
     * Get the shape of this tensor.
     * @return A <b>copy</b> of {@link Tensor_size} indicating the shape of tensor.
     */
    public Tensor_size size(){
        return this.size.clone();
    }

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
     * {@code this_i + t2_i}
     * @param t2 Tensor to add.
     * @return A new tensor.
     */
    public abstract Tensor add(Tensor t2);

    /**
     * {@code this_i + x}
     * @param x Double to add.
     * @return A new tensor.
     */
    public abstract Tensor add(double x);

    /**
     * {@code this_i - t2_i}
     * @param t2 Tensor.
     * @return A new tensor.
     */
    public abstract Tensor subtract(Tensor t2);

    /**
     * Res_{i} = this_{i} - x
     * @param x minus, double variable
     * @return tensor Res
     */
    public abstract Tensor subtract(double x);

    /**
     * {@code res_i = this_i * x}
     * @param x minus, int variable
     * @return tensor Res
     */
    public Tensor subtract(int x){
        return subtract((double)x);
    }

    /**
     * {@code Res_i = x- this_i}
     * @param x subtracted, double variable
     * @return tensor Res
     */
    public abstract Tensor subtracted(double x);

    /**
     * {@code Res_i = x - this_i}
     * @param x subtracted, int variable
     * @return tensor res
     */
    public Tensor subtracted(int x){
        return subtracted((double)x);
    }

    /**
     * Element-wise multiplication. {@code this_i * t_i}
     * @param t Tensor to multiply.
     * @return A new tensor.
     */
    public abstract Tensor dot_mul(Tensor t);


    /**
     * {@code Res_i = this_i * x}
     * @param x Double to multiply.
     * @return A new tensor.
     */
    public abstract Tensor dot_mul(double x);

    /**
     * {@code Res_i = this_i * x}
     * @param x Integer to multiply.
     * @return A new tensor.
     */
    public abstract Tensor dot_mul(int x);

    /**
     * Matrix multiplication of {@code this} and {@code x}.
     * <p><b>Note</b>: Matrix multiplication does not satisfy the commutative law,
     * so {@code a.cross_mul(b)} is different from {@code b.cross_mul(a)}.
     * @param x the tensor to be multiplied.
     * @return A new tensor. Result of the multiplication.
     * @throws MatrixDimensionException if this tensor does not
     * have compactible shape for matrix multiply with {@code x}.
     */
    public abstract Tensor cross_mul(Tensor x) throws MatrixDimensionException;

    /**
     * {@code Res_i = this_i / x}
     * @param x Double. Divisor.
     * @return A new tensor.
     */
    public Tensor divided(double x){ return dot_mul( 1.0/x ); };

    /**
     * {@code Res_i = this_i / x}
     * @param x divisor, int
     * @return A new tensor.
     */
    public Tensor divided(int x){
        return divided((double)x);
    }

    /**
     * {@code Res_i = this_i / x_i}
     * @param x divisor, tensor
     * @return A new tensor.
     */
    public abstract Tensor divided(Tensor x);

    /**
     * {@code Res_i = x / this_i}
     * <p><b>Note:</b> Please note the difference with {@link Tensor#divided}.
     * @param x dividend, double
     * @return A new tensor.
     */
    public abstract Tensor divide(double x);

    /**
     * {@code Res_i = x / this_i}
     * @param x dividend, int
     * @return A new tensor.
     */
    public Tensor divide(int x){
        return divide((double)x);
    }

    /**
     * {@code sigmoid(this_i)}
     * @return A new tensor.
     */
    public abstract Tensor sigmoid();

    /**
     * {@code tanh(this_i)}
     * @return A new tensor.
     */
    public abstract Tensor tanh();

    
    /**
     * {@code ReLU(x) = t(x > 0)}
     * {@code ReLU(x) = 0(x <= 0)}
     * @param t Double.
     * @return A new tensor.
     */
    public abstract Tensor relu(double t);

    /**
     * {@code ReLU(x) = 1(x > 0)}
     * {@code ReLU(x) = 0(x <= 0)}
     * @return A new tensor.
     */
    public Tensor relu(){
        return this.relu(1.0);
    }

    /**
     * {@code sgn(this_i)}
     * @return A new tensor.
     */
    public abstract Tensor sgn();

    /**
     * {@code Res_i = t (this_i < 0)}
     * {@code Res_i = 0 (this_i >= 0)}
     * @param t Double.
     * @return A new Tensor. The derivative of ReLU(t).
     */
    public abstract Tensor DiffReLU(double t);


    public Tensor DiffReLU(){
        return this.DiffReLU(1.0);
    }

    /**
     * {@code Res_i = softmax(this_i)}
     * @return A new tensor.
     */
    public abstract Tensor softmax();

    /**
     * {@code Res_i = this_i ^ x}
     * @param x Double.
     * @return A new tensor.
     */
    public abstract Tensor pow(double x);

    /**
     * {@code Res_i = this_i ^ x}
     * @param x Integer.
     * @return A new tensor.
     */
    public abstract Tensor pow(int x);

    /**
     * {@code Res_i = ln(this_i)}
     * @return A new tensor.
     */
    public abstract Tensor ln();

    /**
     * Return the sum of all elements in this tensor.
     * @return Double. Sum of all elements in this tensor.
     */
    public abstract double sum();

    // /**
    //  * Sum over the axis.
    //  * @param axis Integer. The dimension to be sum up.
    //  * @return A new tensor.
    //  */
    // public abstract Tensor sum(int axis);
    // public abstract Tensor sum(int axis, int... _axis);//压缩某些维，先不用实现

    /**
     * Get the maximum element in this tensor.
     * @return Double. The maximum element.
     */
    public abstract double elementMax();

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
