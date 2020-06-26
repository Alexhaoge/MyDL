package mydl.tensor;
import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;


public class Tensor1D extends Tensor {

    private static final long serialVersionUID = 743000385145097795L;

    /**
     * DMatrixRMaj for storing data.
     * @see {@link org.ejml.data.DMatrixRMaj}
     */
    DMatrixRMaj darray = new DMatrixRMaj();

    /**
     * The default construction method gives a one-colums matrix(an array).
     */
    public Tensor1D(double[] data) {
        this.size = new Tensor_size( data.length );
        this.darray = new DMatrixRMaj(data);
    }

    /**
     * Cations! If the input is not an array, the result may be strange.
     * @see {@link Tensor}
     */
    public Tensor1D(DMatrixRMaj darray) {
        this.size = new Tensor_size( darray.data.length );
        this.darray = new DMatrixRMaj(darray.data);
    }

    public Tensor1D(int length) {
        this.size = new Tensor_size( length );
        this.darray = new DMatrixRMaj(length);
    }

    public Tensor1D(Tensor1D t1) {
        this.size = new Tensor_size( t1.size.getTensor_length() );
        this.darray = new DMatrixRMaj(t1.darray);
    }

    /**
     * Res_{i} = t1_{i} + addtion
     * @param t1
     * @param addtion
     * @return
     */
    public static Tensor add (Tensor1D t1, double addtion) {
        Tensor1D res = new Tensor1D(t1);
        CommonOps_DDRM.add( res.darray, addtion );
        return res;
    }

    /**
     * Res_{i} = this_{i} + addtion
     * @param addtion
     * @return
     */
    public Tensor add (double addtion) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.add(res.darray, addtion );
        return res;
    }

    /**
     * Res_{i} = t1_{i} - minuend
     * @param t1
     * @param minuend
     * @return
     */
    public static Tensor subtract (Tensor1D t1, double minuend) {
        return Tensor1D.add( t1, (-1)*minuend );
    }

    /**
     * Res_{i} = this_{i} - minuend
     * @param minuend
     * @return
     */
    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    /**
     * Res_{i} = substract - t1_{i}
     * @param t1
     * @param subtract
     * @return
     */
    public static Tensor subtracted (Tensor1D t1, double subtract) {
        return t1.subtracted( subtract );
    }

    /**
     * Res_{i} = substract - this_{i}
     * @param subtract
     * @return
     */
    public Tensor subtracted (double subtract) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.add( res.darray, subtract );
        return res;
    }

    /**
     * Res_{i} = t1_{i}*times
     * @param t1
     * @param times
     * @return
     */
    public static Tensor dot_mul (Tensor1D t1, double times) {
        Tensor1D res = new Tensor1D( t1 );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    /**
     * Res_{i} = this_{i}*times
     * @param times
     * @return
     */
    public Tensor dot_mul (double times) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    // 实际上，scale函数只有double 类型传入，所以...单独拿出int并不能优化 除非 调用ejml底层的代码

    /**
     * Res_{i} = t1_{i}*int_times
     * @param t1
     * @param int_times
     * @return
     */
    public static Tensor dot_mul(Tensor1D t1, int int_times) {
        Tensor1D res = new Tensor1D( t1 );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    /**
     * Res_{i} = this_{i}*int_times
     * @param int_times
     * @return
     */
    public Tensor dot_mul(int int_times) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    /**
     * Res_{i} = t1_{i}*t2_{i}
     * @param t1
     * @param t2
     * @return
     */
    public static Tensor dot_mul(Tensor1D t1, Tensor1D t2) {
        Tensor1D res = new Tensor1D( t1.darray.getNumElements() );
        CommonOps_DDRM.elementMult( t1.darray, t2.darray, res.darray );
        return res;
    }

    /**
     * Res_{i} = this_{i}*t2_{i}
     * @param t2
     * @return
     */
    public Tensor dot_mul(Tensor t2){
        if(t2 instanceof Tensor1D){
            Tensor1D res = new Tensor1D( this );
            CommonOps_DDRM.elementMult( res.darray, ((Tensor1D) t2).darray, res.darray );
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor sizes differ.");
        }
    }

    /**
     *The methond cross_mul means res = this*t2^{T}
     * @param t2 (n*1 array and t2^{T} means 1*a array)
     * @return gives a Tensor2D result.
     */
    public Tensor cross_mul(Tensor t2) {
        if (t2 instanceof Tensor1D){
            Tensor2D res = new Tensor2D( this.darray.getNumElements(), this.darray.getNumElements() );
            CommonOps_DDRM.multTransB(this.darray, ((Tensor1D) t2).darray, res.darray);
            return res;
        }else{
            throw new MatrixDimensionException("Tensor sizes differ.");
        }
    }

    /**
     * Res_{i} = this_{i}/t1_{i}
     * @param t1
     * @return
     */
    public Tensor divided (Tensor t1) {
        Tensor1D res = new Tensor1D(this);
        CommonOps_DDRM.elementDiv( res.darray, ((Tensor1D)t1).darray );
        return res;
    }

    /**
     * Res_{n1 x n2} = t1_{n1 x 1}*{t2_{n2 x 1}}^T
     * @param t1
     * @param t2
     * @return
     */
    public static Tensor cross_mul(Tensor1D t1, Tensor1D t2) {
        Tensor2D res = new Tensor2D( t1.darray.getNumElements(), t2.darray.getNumElements() );
        CommonOps_DDRM.multTransB(t1.darray, t2.darray, res.darray);
        return res;
    }

    /**
     * Res_{i} = this_{i}^pow
     * @param pow
     * @return
     */
    public Tensor pow(double pow) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    /**
     * Res_{i} = this_{i}^pow
     * @param pow
     * @return
     */
    public Tensor pow(int pow) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    /**
     * Res_{i} = ln(this_{i})
     * @return
     */
    public Tensor ln () {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementLog( res.darray, res.darray );
        return res;
    }

    /**
     * Res_{i} = t1_{i}^pow
     * @param t1
     * @param pow
     * @return
     */
    public static Tensor pow(Tensor1D t1 , double pow) {
        Tensor1D res = new Tensor1D( t1.darray.getNumElements() );
        CommonOps_DDRM.elementPower( pow, t1.darray, res.darray );
        return res;
    }

    /**
     * Res_{i} = sigmoid(this_{i})
     * @return
     */
    public Tensor sigmoid() {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.elementPower( Math.E, res.darray, res.darray );
        CommonOps_DDRM.add(res.darray, 1);
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }

    /**
     * Res_{i} = sigmoid(t1_{i})
     * @param t1
     * @return
     */
    public static Tensor sigmoid(Tensor1D t1) {
        Tensor1D res = new Tensor1D( t1 );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.elementPower( Math.E, res.darray, res.darray );
        CommonOps_DDRM.add(res.darray, 1);
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }

    /**
     * Res = \sum{t1_{i}}
     * @param t1
     * @return
     */
    public static double sum(Tensor1D t1) {
        return CommonOps_DDRM.elementSum( t1.darray );
    }

    /**
     * Res = \sum{t1_{i}}
     * @return
     */
    public double sum() {
        return Tensor1D.sum( this );
    }

    public Tensor sum (int axis) {
        return null;
    }

    public Tensor sum (int axis, int... _axis) {
        return null;
    }

    /**
     * this_{i} = 0
     * @return
     */
    public Tensor set_zero () {
        Tensor1D res = new Tensor1D( this );
        res.darray.zero();
        return res;
    }

    /**
     * Deep clone: Res = this
     * @return
     */
    public Tensor clone () {
        return new Tensor1D( this );
    }

    /**
     * Res: rownum, colnum, N: new_size.Tensor_length[0], [1], [2]
     * @param new_size
     * @return
     */
    public Tensor reshape (Tensor_size new_size) {
        switch (new_size.size){
            case 1: {
                Tensor1D res = new Tensor1D( this );
                res.darray.reshape( new_size.getTensor_length()[0], 1 );
                res.size = new_size;
                return res;
            }
            case 2:{
                DMatrixRMaj d1 = new DMatrixRMaj(this.darray.data);
                d1.reshape( new_size.getTensor_length()[0], new_size.getTensor_length()[1] );
                Tensor2D res = new Tensor2D( d1 );
                return res;
            }
            case 3:{
                ArrayList<DMatrixRMaj> d1 = new ArrayList<>();
                d1.add(this.darray);
                Tensor3D res = new Tensor3D( d1 );
                return res;
            }
            default:{
                throw new MatrixDimensionException("Dimension errors");
            }
        }
    }

    /**
     * Res = rownum of this
     * @return
     */
    public Tensor_size size () {
        Tensor_size res = new Tensor_size( this.darray.data.length );
        return res;
    }

    /**
     * Res = this^T
     * @return
     */
    public Tensor transpose () {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.transpose( res.darray );
        return res;
    }

    /**
     * Res_{i} = this_{i}+t2_{i}
     * @param t2
     * @return
     */
    public Tensor add (Tensor t2) {
        if (t2 instanceof Tensor1D){
            Tensor1D res = new Tensor1D( this );
            CommonOps_DDRM.add(res.darray, ((Tensor1D) t2).darray, res.darray);
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor sizes differ.");
        }

    }

    /**
     * Res_{i} = this_{i}-t2_{i}
     * @param t2
     * @return
     */
    public Tensor subtract (Tensor t2) {
        if (t2 instanceof Tensor1D){
            Tensor1D res = new Tensor1D( this );
            CommonOps_DDRM.subtract(res.darray, ((Tensor1D) t2).darray, res.darray);
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor sizes differ.");
        }
    }

    /**
     * Res = (DMatrixRMaj) this
     * @return
     */
    public DMatrixRMaj getData() {
        return this.darray;
    }

    /**
     * Res_{i} = dividend / this_{i}
     * @param dividend
     * @return
     */
    public Tensor divide(double dividend) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.divide( dividend, res.darray );
        return res;
    }

    /**
     * Res_{i} = dividend / t1_{i}
     * @param dividend
     * @param t1
     * @return
     */
    public static Tensor divide(double dividend, Tensor1D t1) {
        Tensor1D res = new Tensor1D( t1);
        CommonOps_DDRM.divide( dividend, t1.darray, res.darray );
        return res;
    }

    /**
     * Res_{i} = tanh(this_{i})
     * @return
     */
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

    /**
     * Res_{i} = t*this_{i} (this_{i} > 0)
     * Res_{i} = 0 (this_{i} <= 0)
     * @param t
     * @return
     */
    public Tensor relu(double t) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.abs(this.darray, res.darray);
        CommonOps_DDRM.add( 0.5*t, this.darray, 0.5*t, res.darray, res.darray );
        return res;
    }

    /**
     * Res_{i} = t (this_{i} > 0)
     * Res_{i} = 0 (this_{i} <= 0)
     * @param t
     * @return
     */
    public Tensor DiffReLU(double t) {
        Tensor1D res = new Tensor1D( this );
        for (int i = 0; i < res.darray.getData().length; i++) {
            if (res.darray.getData()[i] > 0) {
                res.darray.getData()[i] = t;
            }else {
                res.darray.getData()[i] = 0;
            }
        }
        return res;
    }

    /**
     * Res_{i} = 1 (this_{i} > 0)
     * Res_{i} = 0 (this_{i} = 0)
     * Res_{i} = -1 (this_{i} < 0)
     * @return
     */
    public Tensor sgn() {
        Tensor1D res = new Tensor1D( this );
        for (int i = 0; i < res.darray.getData().length; i++) {
            if (res.darray.getData()[i] > 0) {
                res.darray.getData()[i] = 1;
            }else if (res.darray.getData()[i] < 0){
                res.darray.getData()[i] = -1;
            }
        }
        return  res;
    }
}