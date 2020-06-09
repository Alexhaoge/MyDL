package mydl.tensor;
import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;


public class Tensor1D extends Tensor {
    DMatrixRMaj darray = new DMatrixRMaj();

    // The default construction method gives a one-colums matrix(an array).
    public Tensor1D(double[] data) {
        int[] length = new int[1];
        length[0] = data.length;
        this.size = new Tensor_size( 1, length );
        this.darray = new DMatrixRMaj(data);
    }

    //Cations! If the input is not an array, the result may be strange.
    public Tensor1D(DMatrixRMaj darray) {
        int[] length = new int[1];
        length[0] = darray.data.length;
        this.size = new Tensor_size( 1, length );
        this.darray = new DMatrixRMaj(darray.data);
    }

    public Tensor1D(int length) {
        int[]len = new int[1];
        len[0] = length;
        this.size = new Tensor_size( 1, len);
        this.darray = new DMatrixRMaj(length);
    }

    public Tensor1D(Tensor1D t1) {
        this.size = t1.size;
        this.darray = new DMatrixRMaj(t1.darray);
    }

    public static Tensor add (Tensor1D t1, double addtion) {
        Tensor1D res = new Tensor1D(t1);
        CommonOps_DDRM.add( res.darray, addtion );
        return res;
    }

    public Tensor add (double addtion) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.add(res.darray, addtion );
        return res;
    }

    public static Tensor subtract (Tensor1D t1, double minuend) {
        return Tensor1D.add( t1, (-1)*minuend );
    }

    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    public static Tensor substracted (Tensor1D t1, double substract) {
        return t1.subtracted( substract );
    }

    public Tensor subtracted (double substract) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.add( res.darray, substract );
        return res;
    }

    public static Tensor dot_mul (Tensor1D t1, double times) {
        Tensor1D res = new Tensor1D( t1 );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    public Tensor dot_mul (double times) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    // 实际上，scale函数只有double 类型传入，所以...单独拿出int并不能优化 除非 调用ejml底层的代码

    public static Tensor dot_mul(Tensor1D t1, int int_times) {
        Tensor1D res = new Tensor1D( t1 );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    public Tensor dot_mul(int int_times) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    /**
     * The method cross_mul means res_{i, j} = t1_{i, j}*t2_{i, j}
     * @param t1
     * @param t2
     */

    public static Tensor cross_mul(Tensor1D t1, Tensor1D t2) {
        Tensor1D res = new Tensor1D( t1.darray.getNumElements() );
        CommonOps_DDRM.elementMult( t1.darray, t2.darray, res.darray );
        return res;
    }

    /**
     * The method cross_mul means res_{i, j} = this_{i, j}*t2_{i, j}
     * @param t2
     */

    public Tensor cross_mul(Tensor1D t2) throws MatrixDimensionException {
        if (this.darray.getNumElements() != t2.darray.getNumElements()) {
            System.err.println("Length of two arrays differs.");
            return null;
        }
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementMult( res.darray, t2.darray, res.darray );
        return res;
    }

    /**
     *The methond tensor_mul means res = this*t2^{T}
     * @param t2 (n*1 array and t2^{T} means 1*a array)
     * @return gives a Tensor2D result.
     */
    public Tensor tensor_mul(Tensor1D t2) {
        if (this.darray.getNumElements() != t2.darray.getNumElements()) {
            System.err.println("Length of two arrays differs.");
            return null;
        }
        else {
            Tensor2D res = new Tensor2D( this.darray.getNumElements(), this.darray.getNumElements() );
            CommonOps_DDRM.multTransB(this.darray, t2.darray, res.darray);
            return res;
        }
    }

    public static Tensor tensor_mul(Tensor1D t1, Tensor1D t2) {
        if (t1.darray.getNumElements() != t2.darray.getNumElements()) {
            System.err.println("Length of two arrays differs.");
            return null;
        }
        else {
            Tensor2D res = new Tensor2D( t1.darray.getNumElements(), t2.darray.getNumElements() );
            CommonOps_DDRM.multTransB(t1.darray, t2.darray, res.darray);
            return res;
        }
    }

    public Tensor pow(double pow) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    public Tensor pow(int pow) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    public static Tensor pow(Tensor1D t1 , double pow) {
        Tensor1D res = new Tensor1D( t1.darray.getNumElements() );
        CommonOps_DDRM.elementPower( pow, t1.darray, res.darray );
        return res;
    }
    public Tensor sigmoid() {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.elementPower( Math.E, res.darray, res.darray );
        CommonOps_DDRM.add(res.darray, 1);
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }

    public static Tensor sigmoid(Tensor1D t1) {
        Tensor1D res = new Tensor1D( t1 );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.elementPower( Math.E, res.darray, res.darray );
        CommonOps_DDRM.add(res.darray, 1);
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }

    public static double sum(Tensor1D t1) {
        return CommonOps_DDRM.elementSum( t1.darray );
    }

    public double sum() {
        return Tensor1D.sum( this );
    }

    public Tensor reshape (Tensor_size new_size) {
        if (size.dim == 1){
            Tensor1D res = new Tensor1D( this );
            res.darray.reshape( new_size.Tensor_length[0], 1 );
            res.size = new_size;
        }else{
            System.err.println("Input errors!");
            return null;
        }

    }

    public DMatrixRMaj getData() {
        return this.darray;
    }

    public Tensor divide(double dividend) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.divide( dividend, res.darray );
        return res;
    }

    public static Tensor divide(double dividend, Tensor1D t1) {
        Tensor1D res = new Tensor1D( t1);
        CommonOps_DDRM.divide( dividend, t1.darray, res.darray );
        return res;
    }

    public Tensor tanh() {
        Tensor1D res = new Tensor1D( this );
        Tensor1D res1 = new Tensor1D( this );
        CommonOps_DDRM.scale(2, res1.darray );
        CommonOps_DDRM.scale( -1, res1.darray );
        CommonOps_DDRM.elementPower( Math.E, res1.darray, res1.darray );
        CommonOps_DDRM.add(res1.darray, 1);
        CommonOps_DDRM.divide( 1, res1.darray );
        Tensor1D res2 = new Tensor1D( this );
        CommonOps_DDRM.scale(-2, res2.darray );
        CommonOps_DDRM.scale( -1, res2.darray );
        CommonOps_DDRM.elementPower( Math.E, res2.darray, res2.darray );
        CommonOps_DDRM.add(res2.darray, 1);
        CommonOps_DDRM.divide( 1, res2.darray );
        CommonOps_DDRM.divide( 1, res2.darray );
        CommonOps_DDRM.scale( -1, res2.darray );
        CommonOps_DDRM.add( res1.darray, res2.darray, res.darray );
        return res;
    }
}