package mydl.tensor;
import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;


public class Tensor1D extends Tensor{
    DMatrixRMaj darray = new DMatrixRMaj();

    // The default construction method gives a one-colums matrix(an array).
    public Tensor1D(double[] data) {
        this.darray = new DMatrixRMaj(data);
    }

    //Cations! If the input is not an array, the result may be strange.
    public Tensor1D(DMatrixRMaj darray) {
        this.darray = new DMatrixRMaj(darray.data);
    }

    public Tensor1D(int length) {
        this.darray = new DMatrixRMaj(length);
    }

    public Tensor1D(Tensor1D t1) {
        this.darray = new DMatrixRMaj(t1.darray);
    }

    public static Tensor add (Tensor1D t1, double addtion) {
        Tensor1D res = new Tensor1D(t1);
        CommonOps_DDRM.add( res.darray, addtion );
        return res;
    }

    public Tensor add (double addtion) {
        CommonOps_DDRM.add(this.darray, addtion );
        return this;
    }

    public static Tensor subtract (Tensor1D t1, double minuend) {
        return Tensor1D.add( t1, (-1)*minuend );
    }

    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    public static Tensor substracted (Tensor1D t1, double substract) {
        Tensor1D res = new Tensor1D( t1 );
        res.dot_mul( -1 );
        return (Tensor1D) Tensor1D.add( res , substract );
    }

    public Tensor subtracted (double substract) {
        this.dot_mul( -1 );
        return this.add(substract);
    }

    public static Tensor dot_mul (Tensor1D t1, double times) {
        Tensor1D res = new Tensor1D( t1 );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    public Tensor dot_mul (double times) {
        CommonOps_DDRM.scale( times, this.darray );
        return this;
    }

    // 实际上，scale函数只有double 类型传入，所以...单独拿出int并不能优化 除非 调用ejml底层的代码

    public static Tensor dot_mul(Tensor1D t1, int int_times) {
        Tensor1D res = new Tensor1D( t1 );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    public Tensor dot_mul(int int_times) {
        CommonOps_DDRM.scale( int_times, this.darray );
        return this;
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
        CommonOps_DDRM.elementMult( this.darray, t2.darray, this.darray );
        return this;
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

    public static Tensor tensor_mul(Tensor1D t1,Tensor1D t2) {
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
        CommonOps_DDRM.elementPower( pow, this.darray, this.darray );
        return this;
    }

    public Tensor pow(int pow) {
        CommonOps_DDRM.elementPower( pow, this.darray, this.darray );
        return this;
    }

    public static Tensor pow(Tensor1D t1 ,double pow) {
        Tensor1D res = new Tensor1D( t1.darray.getNumElements() );
        CommonOps_DDRM.elementPower( pow, t1.darray, res.darray );
        return res;
    }
    public Tensor sigmoid() {
        this.dot_mul( -1 );
        this.pow(Math.E);
        this.add(1);
        return this.devide( 1 );
    }

    public static Tensor sigmoid(Tensor1D t1) {
        Tensor1D res = new Tensor1D( (Tensor1D) Tensor1D.pow( t1, Math.E ) );
        return res;
    }

    public static double sum(Tensor1D t1) {
        return CommonOps_DDRM.elementSum( t1.darray );
    }

    public double sum() {
        return Tensor1D.sum( this );
    }

    @Override
    public Tensor reshape (int x, int y) {
        return null;
    }


    public Tensor reshape (int x) {
        this.darray.reshape( x, 1, true );
        return this;
    }
    public DMatrixRMaj getData() {
        return this.darray;
    }

    public Tensor devide(double devidend) {
        CommonOps_DDRM.divide( devidend, this.darray );
        return this;
    }
    public static Tensor devide(double devidend, Tensor1D t1) {
        Tensor1D res = new Tensor1D( t1);
        CommonOps_DDRM.divide( devidend, t1.darray, res.darray );
        return res;
    }

    public Tensor tanh() {
        Tensor1D res1 = new Tensor1D( this );
        res1.dot_mul( 2 );
        res1.sigmoid();
        Tensor1D res2 = new Tensor1D( this );
        res2.dot_mul( -2 );
        res2.sigmoid();
        res2.dot_mul( -1 );
        CommonOps_DDRM.add( res1.darray, res2.darray, this.darray );
        return this;
    }
}