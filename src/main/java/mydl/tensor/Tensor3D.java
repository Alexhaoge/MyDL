package mydl.tensor;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;


public class Tensor3D extends Tensor {
    ArrayList<DMatrixRMaj> darray = new ArrayList<DMatrixRMaj>();
    // The default construction method gives a one-colums matrix(an array).
//    public Tensor3D(double[][] data) {
//        this.darray = new DMatrixRMaj(data);
//    }

    public Tensor3D(ArrayList<double[][]> a, double N) {
        for (int i = 0; i < N; i++) {
            DMatrixRMaj temp = new DMatrixRMaj(a.get( i ));
            this.darray.add( temp );
        }
    }

    //Cations! If the input is not an array, the result may be strange.
    /**
     * The origin code of DMatrixRMaj.copy() which means deep clone.
     * public DMatrixRMaj copy() {
     * return new DMatrixRMaj(this);
     *    }
     */
    public Tensor3D(ArrayList<DMatrixRMaj> d) {
        for (int i = 0; i < d.size(); i++) {
            this.darray.add( d.get( i ).copy() );
        }
    }

    /**
     * dim means the length of Square matrix(len*len)
     * @param rowNum
     * @param colNum
     * @param N
     */
    public Tensor3D(int rowNum, int colNum, int N) {
        for (int i = 0; i < N; i++) {
            this.darray.add(new DMatrixRMaj(rowNum, colNum));
        }
    }


    public Tensor3D(Tensor3D t1) {
        for (int i = 0; i < t1.darray.size(); i++) {
            this.darray.add(t1.darray.get( i ).copy());
        }
    }

    public static Tensor add (Tensor3D t1, double addtion) {
        Tensor3D res = new Tensor3D(t1);
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.add( res.darray.get( i ), addtion );
        }
        return res;
    }

    public Tensor add (double addtion) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.add( res.darray.get( i ), addtion );
        }
        return res;
    }

    public static Tensor subtract (Tensor3D t1, double minuend) {
        return Tensor3D.add( t1, (-1)*minuend );
    }

    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    public static Tensor substracted (Tensor3D t1, double substract) {
        return t1.subtracted( substract );
    }

    public Tensor subtracted (double substract) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.scale( -1, res.darray.get( i ) );
            CommonOps_DDRM.add( res.darray.get( i ), substract );
        }
        return res;
    }

    public static Tensor dot_mul (Tensor3D t1, double times) {
        Tensor3D res = new Tensor3D( t1 );
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.scale( times, res.darray.get( i ) );
        }
        return res;
    }

    public Tensor dot_mul (double times) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.scale( times, res.darray.get( i ) );
        }
        return res;
    }


    // 实际上，scale函数只有double 类型传入，所以...单独拿出int并不能优化 除非 调用ejml底层的代码

    public static Tensor dot_mul(Tensor3D t1, int int_times) {
        Tensor3D res = new Tensor3D( t1 );
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.scale( int_times, res.darray.get( i ) );
        }
        return res;
    }

    public Tensor dot_mul(int int_times) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.scale( int_times, res.darray.get( i ) );
        }
        return res;
    }



    public static Tensor cross_mul(Tensor3D t1, Tensor3D t2) throws MatrixDimensionException {
        if (t1.darray.size() != t2.darray.size()) {
            System.err.println("The N of 3D tensors differ.");
            return null;
        }
        else {
            Tensor3D res = new Tensor3D( t1 );
            for (int i = 0; i < res.darray.size(); i++) {
                CommonOps_DDRM.mult( t1.darray.get( i ), t2.darray.get( i ), res.darray.get( i ) );
            }
            return res;
        }
    }

    public Tensor cross_mul(Tensor3D t2) {
        if (this.darray.size() != t2.darray.size()) {
            System.err.println("The N of 3D tensors differ.");
            return null;
        }
        else {
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i < this.darray.size(); i++) {
                CommonOps_DDRM.mult( res.darray.get( i ), t2.darray.get( i ), res.darray.get( i ) );
            }
            return res;
        }

    }

    public Tensor pow(double pow) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, res.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }

    public Tensor pow(int pow) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, res.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }

    public static Tensor pow(Tensor3D t1 , double pow) {
        Tensor3D res = new Tensor3D( t1 );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, t1.darray.get( i ), res.darray.get(i) );
        }
        return res;
    }
    public Tensor sigmoid() {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.scale( -1, res.darray.get( i ) );
            CommonOps_DDRM.elementPower( Math.E, res.darray.get( i ), res.darray.get( i ) );
            CommonOps_DDRM.add(res.darray.get( i ), 1);
            CommonOps_DDRM.divide( 1, res.darray.get( i ) );
        }
        return res;
    }

    public static Tensor sigmoid(Tensor3D t1) {
        return t1.sigmoid();
    }

    public Tensor2D sum(int i) {
        switch (i){
            case 1:{
                DMatrixRMaj res = new DMatrixRMaj( this.darray.get( 0 ).getNumRows(), this.darray.size() );
                res.zero();
                for (int j = 0; j < this.darray.size(); j++){
                    DMatrixRMaj temp = new DMatrixRMaj(this.darray.get( i ).getNumRows(), 1);
                    CommonOps_DDRM.sumRows( this.darray.get( i ), temp );
                    for (int k = 0; k < temp.data.length; k++){
                        res.data[j*this.darray.get( 0 ).getNumRows()+k] = temp.data[k];
                    }
                }
                Tensor2D res = new Tensor2D( res );
                return res;
                break;
            }
            case 2:{
                DMatrixRMaj res = new DMatrixRMaj( this.darray.get( 0 ).getNumCols(), this.darray.size() );
                res.zero();
                for (int j = 0; j < this.darray.size(); j++){
                    DMatrixRMaj temp = new DMatrixRMaj(this.darray.get( i ).getNumCols(), 1);
                    CommonOps_DDRM.sumCols( this.darray.get( i ), temp );
                    for (int k = 0; k < temp.data.length; k++){
                        res.data[j*this.darray.get( 0 ).getNumCols()+k] = temp.data[k];
                    }
                }
                Tensor2D res = new Tensor2D( res );
                return res;
                break;
            }
            case 3:{
                DMatrixRMaj res = new DMatrixRMaj(this.darray.get( 0 ).getNumRows(), this.darray.get( 0 ).getNumCols());
                res.zero();
                for (int j = 0; j < this.darray.size(); j++ ){
                    res.data[j] += this.darray.get( i ).data[j];
                }
                Tensor2D res = new Tensor2D( res );
                return res;
                break;
            }
            default:{
                System.err.println("input errors. i = 1, 2, 3");
                return null;
            }
        }

    }

    public static double sum(Tensor3D t1) {
        double tempsum = 0;
        for (int i = 0; i < t1.darray.size(); i++) {
            tempsum += CommonOps_DDRM.elementSum( t1.darray.get( i ) );
        }
        return tempsum;
    }

    public double sum() {
        double tempsum = 0;
        for (int i = 0; i < this.darray.size(); i++) {
            tempsum += CommonOps_DDRM.elementSum( this.darray.get( i ) );
        }
        return tempsum;
    }

    public Tensor reshape (int x, int y, int N) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < N; i++) {
            this.darray.get( i ).reshape( x, y, true );
            res.darray.add(this.darray.get( i ).copy());
        }
        return res;
    }

    public Tensor reshape (int x, int y) {
        return null;
    }

    public Tensor reshape (int x) {
        return null;
    }
    public ArrayList<DMatrixRMaj> getData() {
        return this.darray;
    }

    public Tensor devide (double devidend) {
        Tensor3D res = new Tensor3D(this);
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.divide( devidend, res.darray.get( i ));
        }
        return res;
    }

    public static Tensor devide(double devidend, Tensor3D t1) {
        Tensor3D res = new Tensor3D( t1);
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.divide( devidend, t1.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }

    public Tensor tanh() {
        Tensor3D res1 = new Tensor3D( this );
        for (int i = 0; i < res1.darray.size(); i++){
            CommonOps_DDRM.scale( 2, res1.darray.get( i ) );
            CommonOps_DDRM.scale( -1, res1.darray.get( i ) );
            CommonOps_DDRM.elementPower( Math.E, res1.darray.get( i ), res1.darray.get( i ) );
            CommonOps_DDRM.add(res1.darray.get( i ), 1);
            CommonOps_DDRM.divide( 1, res1.darray.get( i ) );
        }
        Tensor3D res2 = new Tensor3D( this );
        for (int i = 0; i < res2.darray.size(); i++){
            CommonOps_DDRM.scale( 2, res2.darray.get( i ) );
            CommonOps_DDRM.scale( -1, res2.darray.get( i ) );
            CommonOps_DDRM.elementPower( Math.E, res2.darray.get( i ), res2.darray.get( i ) );
            CommonOps_DDRM.add(res2.darray.get( i ), 1);
            CommonOps_DDRM.divide( 1, res2.darray.get( i ) );
            CommonOps_DDRM.scale( -1, res2.darray.get( i ) );
            CommonOps_DDRM.add( res1.darray.get( i ), res2.darray.get( i ), res2.darray.get(i) );
        }

        return res2;
    }
    public Tensor tensor_mul(Tensor3D t2) {
        if (t2.darray.size() != this.darray.size()) {
            System.err.println("Length of two 3D-Tensor differs.");
            return null;
        }else {
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.mult(res.darray.get( i ), t2.darray.get(i), res.darray.get( i ));
            }
            return res;
        }
    }
    public static Tensor tensor_mul(Tensor3D t1, Tensor3D t2) {
        if (t2.darray.size() != t1.darray.size()) {
            System.err.println("Length of two 3D-Tensor differs.");
            return null;
        }else {
            Tensor3D res = new Tensor3D( t1 );
            for (int i = 0; i<t1.darray.size(); i++) {
                CommonOps_DDRM.mult(t1.darray.get( i ), t2.darray.get(i), res.darray.get( i ));
            }
            return res;
        }
    }

}