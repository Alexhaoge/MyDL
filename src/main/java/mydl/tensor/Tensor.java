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
                Tensor1D res = new Tensor1D( _size.Tensor_length[0] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextDouble()*(ceil-floor)+floor;
                }
                return res;
            }
            case 2:{
                Tensor2D res = new Tensor2D( _size.Tensor_length[0], 
                    _size.Tensor_length[1] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextDouble()*(ceil-floor)+floor;
                }
                return res;
            }
            case 3:{
                Tensor3D res = new Tensor3D( _size.Tensor_length[0], 
                    _size.Tensor_length[1], _size.Tensor_length[2]);
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
                Tensor1D res = new Tensor1D( _size.Tensor_length[0] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextDouble();
                }
                return res;
            }
            case 2:{
                Tensor2D res = new Tensor2D( _size.Tensor_length[0], 
                    _size.Tensor_length[1] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextDouble();
                }
                return res;
            }
            case 3:{
                Tensor3D res = new Tensor3D( _size.Tensor_length[0], 
                    _size.Tensor_length[1], _size.Tensor_length[2]);
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
                Tensor1D res = new Tensor1D( _size.Tensor_length[0] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextGaussian() * std + mean;
                }
                return res;
            }
            case 2:{
                Tensor2D res = new Tensor2D( _size.Tensor_length[0], 
                    _size.Tensor_length[1] );
                for (int i = 0; i < res.darray.data.length; i++){
                    res.darray.data[i] = ran.nextGaussian() * std + mean;
                }
                return res;
            }
            case 3:{
                Tensor3D res = new Tensor3D( _size.Tensor_length[0], 
                    _size.Tensor_length[1], _size.Tensor_length[2]);
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
                Tensor1D res = new Tensor1D( _size.Tensor_length[0] );
                res.darray.zero();
                return res;
            }
            case 2:{
                Tensor2D res = new Tensor2D( _size.Tensor_length[0], 
                    _size.Tensor_length[1] );
                res.darray.zero();
                return res;
            }
            case 3:{
                Tensor3D res = new Tensor3D( _size.Tensor_length[0], 
                    _size.Tensor_length[1], _size.Tensor_length[2]);
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
     * Matrix multiplication of {@code this} and {@code x}.
     * @param x the tensor to be multiplied.
     * @throws MatrixDimensionException if this tensor does not 
     * have compactible shape for matrix multiply with {@code x}.
     * @apiNote Matrix multiplication does not satisfy the commutative law, 
     * so {@code a.cross_mul(b)} is different from {@code b.cross_mul(a)}.
     */
    public abstract Tensor cross_mul(Tensor x) throws MatrixDimensionException;

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
     * this / x 按位除法，注意是两个tensor每一位做除法，请务必和divide区分开
     * @param x
     * @return
     */
    public abstract Tensor divided(Tensor x);

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

    /**
     * Relu
     * @param t 
     * @return
     */
    public abstract Tensor relu(double t);
    public Tensor relu(){
        return this.relu(1.0);
    }

    public abstract Tensor sgn();

    public abstract Tensor DiffReLU(double t);
    
    public Tensor DiffReLU(){
        return this.DiffReLU(1.0);
    }

    public abstract Tensor pow(double x);

    public abstract Tensor pow(int x);

    //ln(this),自然对数
    public abstract Tensor ln();

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
    
    public abstract double max();

    public abstract double min();

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
