package mydl.tensor;
import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;

public class Tensor2D extends Tensor {

    private static final long serialVersionUID = 2602694190691785623L;

    /**
     * DMatrixRMaj for storing data.
     * @see {@link org.ejml.data.DMatrixRMaj}
     */
    public DMatrixRMaj darray = new DMatrixRMaj();

    /**
     * The default construction method gives a rownum*colnum matrix.
     * (rownum always rank first).
     */
    public Tensor2D(double[][] data) {
        this.size = new Tensor_size(data[0].length, data.length);
        this.darray = new DMatrixRMaj(data[0].length, data.length);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                this.darray.set( j, i, data[i][j]);
            }
        }
    }

    /**
     * Construct a Tensor2D, rownum = data.length and colnum = 1.
     * @param data
     */
    public Tensor2D(double[] data) {
        this.size = new Tensor_size(data.length, 1);
        this.darray = new DMatrixRMaj(data);
    }

    /**
     * The origin code of DMatrixRMaj.copy() which means deep clone.
     * public DMatrixRMaj copy() {
     * return new DMatrixRMaj(this);
     *    }
     */

    public Tensor2D(DMatrixRMaj darray) {
        this.size = new Tensor_size( darray.getNumRows(), darray.getNumCols() );
        this.darray = darray.copy();
    }

    /**
     * The default construction method gives a rownum*colnum matrix.
     * @param rownum
     * @param colnum
     */
    public Tensor2D(int rownum, int colnum) {
        this.size = new Tensor_size( rownum, colnum );
        this.darray = new DMatrixRMaj(rownum, colnum);
    }

    /**
     * Construct a same Tensor2D(Deep clone).
     * @param t1
     */
    public Tensor2D(Tensor2D t1) {
        this.size = new Tensor_size( t1.size.getTensor_length() );
        this.darray = new DMatrixRMaj(t1.darray);
    }

    /**
     * Res_{i, j} = t1_{i, j} + addtion
     * @param t1
     * @param addtion
     * @return
     */
    public static Tensor add (Tensor2D t1, double addtion) {
        Tensor2D res = new Tensor2D(t1);
        CommonOps_DDRM.add( res.darray, addtion );
        return res;
    }

    /**
     * Res_{i, j} = this_{i, j} + addtion
     * @param addtion
     * @return
     */
    public Tensor add (double addtion) {
        Tensor2D res = new Tensor2D(this);
        CommonOps_DDRM.add(res.darray, addtion );
        return res;
    }

    /**
     * Res_{i, j} = t1_{i, j} - minuend
     * @param t1
     * @param minuend
     * @return
     */
    public static Tensor subtract (Tensor2D t1, double minuend) {
        return Tensor2D.add( t1, (-1)*minuend );
    }

    /**
     * this_{i, j} = 0
     * @return
     */
    public Tensor set_zero () {
        Tensor2D res = new Tensor2D( this );
        res.darray.zero();
        return res;
    }

    /**
     * Deep clone: Res = this
     * @return
     */
    public Tensor clone () {
        return new Tensor2D( this );
    }

    /**
     * Res: rownum, colnum, N: new_size.Tensor_length[0], [1], [2]
     * @param new_size
     * @return
     */
    public Tensor reshape (Tensor_size new_size) {
        Tensor2D res = new Tensor2D( this );
        res.darray.reshape( new_size.getTensor_length()[0],new_size.getTensor_length()[1],true );
        return res;
    }

    /**
     * Res = {rownum of this, colnum}
     * @return
     */
    public Tensor_size size () {
        Tensor_size res = new Tensor_size( this.darray.getNumRows(), this.darray.getNumCols() );
        return res;
    }

    /**
     * Res = this^T
     * @return
     */
    public Tensor transpose () {
        DMatrixRMaj d1 = new DMatrixRMaj(this.darray );
        CommonOps_DDRM.transpose( d1 );
        Tensor2D res = new Tensor2D( d1 );
        return res;
    }

    /**
     * Res_{i, j} = this_{i, j} + t2_{i, j}
     * @param t2
     * @return
     */
    public Tensor add (Tensor t2) {
        if (t2 instanceof Tensor1D) {
            DMatrixRMaj d1 = new DMatrixRMaj(this.darray );
            CommonOps_DDRM.add(d1, ((Tensor1D) t2).darray, d1);
            return new Tensor2D( d1 );
        }
        else if(t2 instanceof Tensor2D) {
            DMatrixRMaj d1 = new DMatrixRMaj(this.darray );
            CommonOps_DDRM.add(d1, ((Tensor1D) t2).darray, d1);
            return new Tensor2D( d1 );
        }
        else if(t2 instanceof Tensor3D) {
            ArrayList<DMatrixRMaj> d1 = ((Tensor3D)t2).darray;
            for (int i = 0; i < ((Tensor3D)t2).darray.size() ; i++){
                CommonOps_DDRM.add(d1.get( i ), darray, d1.get( i ));
            }
            return new Tensor3D( d1 );
        }
        else{
            throw new MatrixDimensionException("Tensor size error.");
        }
    }

    /**
     * Res_{i, j} = this_{i, j}-t2_{i, j}
     * @param t2
     * @return
     */
    public Tensor subtract (Tensor t2) {
        if (t2 instanceof Tensor2D){
            Tensor2D res = new Tensor2D( this );
            CommonOps_DDRM.subtract(res.darray, ((Tensor1D) t2).darray, res.darray);
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor sizes differ.");
        }
    }

    /**
     * Res_{i, j} = this_{i, j}-minuend
     * @param minuend
     * @return
     */
    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    /**
     * Res_{i, j} = substract - this_{i, j}
     * @param substract
     * @return
     */
    public Tensor subtracted (double substract) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.add( res.darray, substract );
        return res;
    }

    /**
     * Res_{i, j} = this_{i, j}*t2_{i, j}
     * @param t2
     * @return
     */
    public Tensor dot_mul (Tensor t2) {
        if(t2 instanceof Tensor2D){
            Tensor2D res = new Tensor2D( this );
            CommonOps_DDRM.elementMult( res.darray, ((Tensor1D) t2).darray, res.darray );
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor sizes differ.");
        }
    }

    /**
     * Res_{i, j} = t1_{i, j}*times
     * @param t1
     * @param times
     * @return
     */
    public static Tensor dot_mul (Tensor2D t1, double times) {
        Tensor2D res = new Tensor2D( t1 );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = this_{i, j}*times
     * @param times
     * @return
     */
    public Tensor dot_mul (double times) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = dividend/this_{i, j}
     * @param dividend
     * @return
     */
    public Tensor divide(double dividend) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.divide( dividend, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = dividend/t2_{i, j}
     * @param dividend
     * @param t1
     * @return
     */
    public static Tensor divide(double dividend, Tensor2D t1) {
        Tensor2D res = new Tensor2D( t1);
        CommonOps_DDRM.divide( dividend, t1.darray, res.darray );
        return res;
    }

    // 实际上，scale函数只有double 类型传入，所以...单独拿出int并不能优化 除非 调用ejml底层的代码

    /**
     * Res_{i, j} = t1_{i, j}*int_times
     * @param t1
     * @param int_times
     * @return
     */
    public static Tensor dot_mul(Tensor2D t1, int int_times) {
        Tensor2D res = new Tensor2D( t1 );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = this_{i, j}*int_times
     * @param int_times
     * @return
     */
    public Tensor dot_mul(int int_times) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    /**
     * Res_{rownum1 x colnum2} = t1_{rownum1 x colnum1}*t2_{colnum1 x colnum2}
     * @param t1
     * @param t2
     * @return
     */
    public static Tensor cross_mul(Tensor2D t1, Tensor2D t2) {
        Tensor2D res = new Tensor2D( t1.darray.getNumRows(), t2.darray.getNumCols() );
        CommonOps_DDRM.mult( t1.darray, t2.darray, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = this_{i, j}*t2_{i, j}
     * @param t2
     * @return
     */
    public Tensor cross_mul(Tensor t2) {
        if(t2 instanceof Tensor2D){
            Tensor2D res = new Tensor2D( this );
            CommonOps_DDRM.mult( res.darray, ((Tensor2D) t2).darray, res.darray );
            return res;
        }else if(t2 instanceof Tensor1D){
            Tensor2D res = new Tensor2D( this );
            CommonOps_DDRM.mult( res.darray, ((Tensor2D) t2).darray, res.darray );
            return res;
        }else if(t2 instanceof Tensor3D){
            Tensor3D res = new Tensor3D( (Tensor3D) t2 );
            for (int i = 0; i < res.darray.size(); i++) {
                CommonOps_DDRM.mult(this.darray, res.darray.get( i ), res.darray.get( i ));
            }
            return res;
        }else {
            throw new MatrixDimensionException("Tensor size error.");
        }
    }

    /**
     * Res_{i, j} = this_{i, j}/t2_{i, j}
     * @param t2
     * @return
     */
    public Tensor divided (Tensor t2) {
        Tensor2D res = new Tensor2D(this);
        CommonOps_DDRM.elementDiv( res.darray, ((Tensor2D)t2).darray );
        return res;
    }

    /**
     * Res_{i, j} = this_{i, j}^pow
     * @param pow
     * @return
     */
    public Tensor pow(double pow) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = this_{i, j}^pow
     * @param pow
     * @return
     */
    public Tensor pow(int pow) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = ln(this_{i, j})
     * @return
     */
    public Tensor ln () {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementLog( res.darray, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = t1_{i, j}^pow
     * @param t1
     * @param pow
     * @return
     */
    public static Tensor pow(Tensor2D t1 , double pow) {
        Tensor2D res = new Tensor2D( t1.darray.getNumRows(), t1.darray.getNumCols() );
        CommonOps_DDRM.elementPower( pow, t1.darray, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = sigmoid(this_{i, j})
     * @return
     */
    public Tensor sigmoid() {
        CommonOps_DDRM.scale( -1, this.darray );
        CommonOps_DDRM.elementPower( Math.E, this.darray, this.darray );
        CommonOps_DDRM.add(this.darray, 1);
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }

    /**
     * Res_{i, j} = sigmoid(t1_{i, j})
     * @param t1
     * @return
     */
    public static Tensor sigmoid(Tensor2D t1) {
        CommonOps_DDRM.scale( -1, t1.darray );
        CommonOps_DDRM.elementPower( Math.E, t1.darray, t1.darray );
        CommonOps_DDRM.add(t1.darray, 1);
        Tensor2D res = new Tensor2D( t1 );
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }

    /**
     * Res{rownum x 1}(i = 1)
     * Res{1 x colnum}(i = 2)
     * @param t1 , input matrix
     * @param i , i = 1 means sum by row, i = 2 means sum by column.
     * @return
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
                throw new MatrixDimensionException("Input error, i = 1 or i = 2.");
            }
        }
        return res;
    }

    /**
     * Res{rownum x 1}(i = 1)
     * Res{1 x colnum}(i = 2)
     * @param i , i = 1 means sum by row, i = 2 means sum by column.
     */
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
                throw new MatrixDimensionException("Input error, i = 1 or i = 2.");
            }
        }
        return res;
    }

    public Tensor sum (int axis, int... _axis) {
        return null;
    }

    /**
     * Res = \sum{this_{i, j}}
     * @return
     */
    public double sum() {
        return CommonOps_DDRM.elementSum( this.darray );
    }
    public static double sum(Tensor2D t1){
        return CommonOps_DDRM.elementSum( t1.darray );
    }

    /**
     * Res = this{rownum x colnum}
     * @param rownum
     * @param colnum
     * @return
     */
    public Tensor reshape (int rownum, int colnum) {
        Tensor2D res = new Tensor2D( this );
        res.darray.reshape( rownum, colnum, true );
        return res;
    }

    /**
     * Res = (DMatrixRMaj) this
     * @return
     */
    public DMatrixRMaj getData() {
        return this.darray;
    }

    /**
     * Res_{i, j} = tanh(this_{i, j})
     * @return
     */
    public Tensor tanh() {
        Tensor2D res1 = new Tensor2D ( this );
        CommonOps_DDRM.scale(2, res1.darray );
        CommonOps_DDRM.elementPower( Math.E, res1.darray, res1.darray );
        DMatrixRMaj d2 = new DMatrixRMaj(res1.darray);
        CommonOps_DDRM.add(res1.darray, 1);
        CommonOps_DDRM.add(d2, -1);
        CommonOps_DDRM.elementDiv( d2, res1.darray, res1.darray );
        return res1;
    }

    /**
     * Res_{i} = t*this_{i} (this_{i} > 0)
     * Res_{i} = 0 (this_{i} <= 0)
     * @param t
     * @return
     */
    public Tensor relu(double t) {
        Tensor2D res = new Tensor2D( this );
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
        Tensor2D res = new Tensor2D( this );
        for (int i = 0; i < res.darray.getData().length; i++) {
            if (res.darray.getData()[i] > 0) {
                res.darray.getData()[i] = t;
            }else{
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
        Tensor2D res = new Tensor2D( this );
        for (int i = 0; i < res.darray.getData().length; i++) {
            if ( res.darray.getData()[i] > 0) {
                res.darray.getData()[i] = 1;
            }else if ( res.darray.getData()[i] < 0) {
                res.darray.getData()[i] = -1;
            }
        }
        return  res;
    }

    public Tensor softmax() {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementExp( this.darray, res.darray);
        double sum = CommonOps_DDRM.elementSum( res.darray );
        CommonOps_DDRM.scale( 1.0/sum, res.darray );
        return res;
    }

}