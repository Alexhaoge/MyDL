package mydl.tensor;
import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;


public class Tensor2D extends Tensor {
    DMatrixRMaj darray = new DMatrixRMaj();

    /**
     * The default construction method gives a rownum*colnum matrix.
     * (rownum always rank first).
     */
    public Tensor2D(double[][] data) {
        int[] length = new int[2];
        length[0] = data.length/data[0].length;
        length[1] = data[0].length;
        this.size = new Tensor_size(length[0], length[1]);
        this.darray = new DMatrixRMaj(data);
    }

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

    public Tensor2D(int rownum, int colnum) {
        this.size = new Tensor_size( rownum, colnum );
        this.darray = new DMatrixRMaj(rownum, colnum);
    }

    public Tensor2D(Tensor2D t1) {
        this.size = new Tensor_size( t1.size.getTensor_length() );
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

    public Tensor set_zero () {
        Tensor2D res = new Tensor2D( this );
        res.darray.zero();
        return res;
    }

    public Tensor clone () {
        return new Tensor2D( this );
    }

    public Tensor reshape (Tensor_size new_size) {
        Tensor2D res = new Tensor2D( this );
        res.darray.reshape( new_size.getTensor_length()[0],new_size.getTensor_length()[1],true );
        return res;
    }

    public Tensor_size size () {
        Tensor_size res = new Tensor_size( this.darray.getNumRows(), this.darray.getNumCols() );
        return res;
    }

    public Tensor transpose () {
        DMatrixRMaj d1 = new DMatrixRMaj(this.darray );
        CommonOps_DDRM.transpose( d1 );
        Tensor2D res = new Tensor2D( d1 );
        return res;
    }

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

    public Tensor subtract (Tensor x) {
        return null;
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

    @Override
    public Tensor dot_mul (Tensor x) {
        return null;
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

    public Tensor divide(double dividend) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.divide( dividend, res.darray );
        return res;
    }

    public static Tensor divide(double dividend, Tensor2D t1) {
        Tensor2D res = new Tensor2D( t1);
        CommonOps_DDRM.divide( dividend, t1.darray, res.darray );
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
                throw new MatrixDimensionException("Input error, i = 1 or i = 2.");
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
                throw new MatrixDimensionException("Input error, i = 1 or i = 2.");
            }
        }
        return res;
    }

    @Override
    public Tensor sum (int axis, int... _axis) {
        return null;
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
    public Tensor ReLU() {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.abs(this.darray, res.darray);
        CommonOps_DDRM.add( 0.5, this.darray, 0.5, res.darray, res.darray );
        return res;
    }
    public Tensor DiffReLU() {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.abs( this.darray, res.darray );
        CommonOps_DDRM.add( this.darray, res.darray, res.darray );
        CommonOps_DDRM.elementDiv( this.darray, res.darray, res.darray );
        for (int i = 0; i < res.darray.getData().length; i++) {
            if (Double.isNaN( res.darray.getData()[i] )) {
                res.darray.getData()[i] = 0;
            }
        }
        return res;
    }
    public Tensor sgn() {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.abs(this.darray, res.darray);
        CommonOps_DDRM.elementDiv( this.darray, res.darray, res.darray );
        for (int i = 0; i < res.darray.getData().length; i++) {
            if (Double.isNaN( res.darray.getData()[i] )) {
                res.darray.getData()[i] = 0;
            }
        }
        return  res;
    }
}