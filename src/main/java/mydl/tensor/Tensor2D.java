package mydl.tensor;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;


public class Tensor2D extends Tensor {
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
        Tensor2D res = new Tensor2D(this);
        CommonOps_DDRM.add(res.darray, addtion );
        return res;
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
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.add( res.darray, substract );
        return res;
    }

    public static Tensor dot_mul (Tensor2D t1, double times) {
        Tensor2D res = new Tensor2D( t1 );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    public Tensor dot_mul (double times) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    public Tensor tensor_mul(Tensor2D t2) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.mult(res.darray, t2.darray, res.darray);
        return res;
    }

    public static Tensor tensor_mul(Tensor2D t1, Tensor2D t2) {
        Tensor2D res = new Tensor2D( t1.darray.getNumRows(), t2.darray.getNumCols() );
        CommonOps_DDRM.mult(t1.darray, t2.darray, res.darray);
        return res;
    }


    public Tensor devide(double devidend) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.divide( devidend, res.darray );
        return res;
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
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }



    public static Tensor cross_mul(Tensor2D t1, Tensor2D t2) {
        Tensor2D res = new Tensor2D( t1.darray.getNumRows(), t2.darray.getNumCols() );
        CommonOps_DDRM.mult( t1.darray, t2.darray, res.darray );
        return res;
    }

    public Tensor cross_mul(Tensor2D t2) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.mult( res.darray, t2.darray, res.darray );
        return res;
    }

    public Tensor pow(double pow) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    public Tensor pow(int pow) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    public static Tensor pow(Tensor2D t1 , double pow) {
        Tensor2D res = new Tensor2D( t1.darray.getNumRows(), t1.darray.getNumCols() );
        CommonOps_DDRM.elementPower( pow, t1.darray, res.darray );
        return res;
    }
    public Tensor sigmoid() {
        CommonOps_DDRM.scale( -1, this.darray );
        CommonOps_DDRM.elementPower( Math.E, this.darray, this.darray );
        CommonOps_DDRM.add(this.darray, 1);
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }

    public static Tensor sigmoid(Tensor2D t1) {
        CommonOps_DDRM.scale( -1, t1.darray );
        CommonOps_DDRM.elementPower( Math.E, t1.darray, t1.darray );
        CommonOps_DDRM.add(t1.darray, 1);
        Tensor2D res = new Tensor2D( t1 );
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }
/**
 *@param t1 , input matrix
 *@param i , i = 1 means sum by row, i = 2 means sum by column.
 */
    public static Tensor sum(Tensor2D t1, int i) {
        Tensor2D res = new Tensor2D( t1 );
        switch (i){
            case 1:{
                CommonOps_DDRM.sumRows ( res.darray, res.darray );
                break;
            }
            case 2: {
                CommonOps_DDRM.sumCols( res.darray, res.darray );
                break;
            }
            default:{
                System.err.println("Input error, i = 1 or i = 2.");
            }
        }
        return res;
    }

    public Tensor sum(int i) {
        Tensor2D res = new Tensor2D( this );
        switch (i){
            case 1:{
                CommonOps_DDRM.sumRows ( res.darray, res.darray );
                break;
            }
            case 2: {
                CommonOps_DDRM.sumCols( res.darray, res.darray );
                break;
            }
            default:{
                System.err.println("Input error, i = 1 or i = 2.");
            }
        }
        return res;
    }

    public double sum() {
        return CommonOps_DDRM.elementSum( this.darray );
    }
    public static double sum(Tensor2D t1){
        return CommonOps_DDRM.elementSum( t1.darray );
    }

    public Tensor reshape (int x, int y) {
        Tensor2D res = new Tensor2D( this );
        res.darray.reshape( x, y, true );
        return res;
    }


    public DMatrixRMaj getData() {
        return this.darray;
    }
    public Tensor tanh() {
        Tensor2D res = new Tensor2D( this );
        Tensor2D res1 = new Tensor2D( this );
        CommonOps_DDRM.scale(2, res1.darray );
        CommonOps_DDRM.scale( -1, res1.darray );
        CommonOps_DDRM.elementPower( Math.E, res1.darray, res1.darray );
        CommonOps_DDRM.add(res1.darray, 1);
        CommonOps_DDRM.divide( 1, res1.darray );
        Tensor2D res2 = new Tensor2D( this );
        CommonOps_DDRM.scale(-2, res2.darray );
        CommonOps_DDRM.scale( -1, res2.darray );
        CommonOps_DDRM.elementPower( Math.E, res2.darray, res2.darray );
        CommonOps_DDRM.add(res2.darray, 1);
        CommonOps_DDRM.divide( 1, res2.darray );
        CommonOps_DDRM.scale( -1, res2.darray );
        CommonOps_DDRM.add( res1.darray, res2.darray, res.darray );
        return res;
    }
}