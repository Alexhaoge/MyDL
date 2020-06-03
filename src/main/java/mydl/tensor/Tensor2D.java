package mydl.tensor;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;


public class Tensor2D extends Tensor{
    DMatrixRMaj darray = new DMatrixRMaj();

    /**
     * The default construction method gives a rownum*colnum matrix.
     * (rownum always rank first).
     */
    public Tensor2D(double[][] data) {
        this.darray = new DMatrixRMaj(data);
    }

    public Tensor2D(double[] data) {
        this.darray = new DMatrixRMaj(data);
    }

    /**
     * The origin code of DMatrixRMaj.copy() which means deep clone.
     * public DMatrixRMaj copy() {
     * return new DMatrixRMaj(this);
     *    }
     */

    public Tensor2D(DMatrixRMaj darray) {
        this.darray = darray.copy();
    }

    public Tensor2D(int rownum, int colnum) {
        this.darray = new DMatrixRMaj(rownum, colnum);
    }

    public Tensor2D(Tensor2D t1) {
        this.darray = new DMatrixRMaj(t1.darray);
    }

    public static Tensor add (Tensor2D t1, double addtion) {
        Tensor2D res = new Tensor2D(t1);
        CommonOps_DDRM.add( res.darray, addtion );
        return res;
    }

    public Tensor add (double addtion) {
        CommonOps_DDRM.add(this.darray, addtion );
        return this;
    }

    public static Tensor subtract (Tensor2D t1, double minuend) {
        return Tensor2D.add( t1, (-1)*minuend );
    }

    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    public static Tensor substracted (Tensor2D t1, double substract) {
        Tensor2D res = new Tensor2D( t1 );
        res.dot_mul( -1 );
        return (Tensor2D) Tensor2D.add( res , substract );
    }

    public Tensor subtracted (double substract) {
        this.dot_mul( -1 );
        return this.add(substract);
    }

    public static Tensor dot_mul (Tensor2D t1, double times) {
        Tensor2D res = new Tensor2D( t1 );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    public Tensor dot_mul (double times) {
        CommonOps_DDRM.scale( times, this.darray );
        return this;
    }

    public Tensor tensor_mul(Tensor2D t2) {
            CommonOps_DDRM.mult(this.darray, t2.darray, this.darray);
            return this;
    }

    public static Tensor tensor_mul(Tensor2D t1, Tensor2D t2) {
        Tensor2D res = new Tensor2D( t1.darray.getNumRows(), t2.darray.getNumCols() );
        CommonOps_DDRM.mult(t1.darray, t2.darray, res.darray);
        return res;
    }


    public Tensor devide(double devidend) {
        CommonOps_DDRM.divide( devidend, this.darray );
        return this;
    }

    public static Tensor devide(double devidend, Tensor2D t1) {
        Tensor2D res = new Tensor2D( t1);
        CommonOps_DDRM.divide( devidend, t1.darray, res.darray );
        return res;
    }

    // 实际上，scale函数只有double 类型传入，所以...单独拿出int并不能优化 除非 调用ejml底层的代码

    public static Tensor dot_mul(Tensor2D t1, int int_times) {
        Tensor2D res = new Tensor2D( t1 );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    public Tensor dot_mul(int int_times) {
        CommonOps_DDRM.scale( int_times, this.darray );
        return this;
    }



    public static Tensor cross_mul(Tensor2D t1, Tensor2D t2) {
        Tensor2D res = new Tensor2D( t1.darray.getNumRows(), t2.darray.getNumCols() );
        CommonOps_DDRM.mult( t1.darray, t2.darray, res.darray );
        return res;
    }

    public Tensor cross_mul(Tensor2D t2) {
        CommonOps_DDRM.mult( this.darray, t2.darray, this.darray );
        return this;
    }

    public Tensor pow(double pow) {
        CommonOps_DDRM.elementPower( pow, this.darray, this.darray );
        return this;
    }

    public Tensor pow(int pow) {
        CommonOps_DDRM.elementPower( pow, this.darray, this.darray );
        return this;
    }

    public static Tensor pow(Tensor2D t1 ,double pow) {
        Tensor2D res = new Tensor2D( t1.darray.getNumRows(), t1.darray.getNumCols() );
        CommonOps_DDRM.elementPower( pow, t1.darray, res.darray );
        return res;
    }
    public Tensor sigmoid() {
        this.dot_mul( -1 );
        this.pow(Math.E);
        this.add(1);
        return this.devide( 1 );
    }

    public static Tensor sigmoid(Tensor2D t1) {
        Tensor2D res = new Tensor2D( (Tensor2D) Tensor2D.pow( t1, Math.E ) );
        return res;
    }

    public static double sum(Tensor2D t1) {
        return CommonOps_DDRM.elementSum( t1.darray );
    }

    public double sum() {
        return Tensor2D.sum( this );
    }

    public Tensor reshape (int x, int y) {
        this.darray.reshape( x, y, true );
        return this;
    }


    public Tensor reshape (int x) {
        return null;
    }
    public DMatrixRMaj getData() {
        return this.darray;
    }
    public Tensor tanh() {
        Tensor2D res1 = new Tensor2D( this );
        res1.dot_mul( 2 );
        res1.sigmoid();
        Tensor2D res2 = new Tensor2D( this );
        res2.dot_mul( -2 );
        res2.sigmoid();
        res2.dot_mul( -1 );
        CommonOps_DDRM.add( res1.darray, res2.darray, this.darray);
        return this;
    }
}