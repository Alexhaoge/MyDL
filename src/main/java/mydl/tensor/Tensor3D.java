package mydl.tensor;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;


public class Tensor3D extends Tensor{
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
     * @param dim
     * @param N
     */
    public Tensor3D(int dim, int N) {
        for (int i = 0; i < N; i++) {
            this.darray.add(new DMatrixRMaj(dim, dim));
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
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.add( this.darray.get( i ), addtion );
        }
        return this;
    }

    public static Tensor subtract (Tensor3D t1, double minuend) {
        return Tensor3D.add( t1, (-1)*minuend );
    }

    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    public static Tensor substracted (Tensor3D t1, double substract) {
        Tensor3D res = new Tensor3D( t1 );
        res.dot_mul( -1 );
        return (Tensor2D) Tensor3D.add( res , substract );
    }

    public Tensor subtracted (double substract) {
        this.dot_mul( -1 );
        return this.add(substract);
    }

    public static Tensor dot_mul (Tensor3D t1, double times) {
        Tensor3D res = new Tensor3D( t1 );
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.scale( times, res.darray.get( i ) );
        }
        return res;
    }

    public Tensor dot_mul (double times) {
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.scale( times, this.darray.get( i ) );
        }
        return this;
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
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.scale( int_times, this.darray.get( i ) );
        }
        return this;
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
            for (int i = 0; i < this.darray.size(); i++) {
                CommonOps_DDRM.mult( this.darray.get( i ), t2.darray.get( i ), this.darray.get( i ) );
            }
            return this;
        }

    }

    public Tensor pow(double pow) {
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, this.darray.get( i ), this.darray.get( i ) );
        }
        return this;
    }

    public Tensor pow(int pow) {
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, this.darray.get( i ), this.darray.get( i ) );
        }
        return this;
    }

    public static Tensor pow(Tensor3D t1 ,double pow) {
        Tensor3D res = new Tensor3D( t1 );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, t1.darray.get( i ), res.darray.get(i) );
        }
        return res;
    }
    public Tensor sigmoid() {
        this.dot_mul( -1 );
        this.pow(Math.E);
        this.add(1);
        return this.devide( 1 );
    }

    public static Tensor sigmoid(Tensor3D t1) {
        Tensor3D res = new Tensor3D( (Tensor3D) Tensor3D.pow( t1, Math.E ) );
        return res;
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
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.divide( devidend, this.darray.get( i ));
        }
        return this;
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
        res1.dot_mul( 2 );
        res1.sigmoid();
        Tensor3D res2 = new Tensor3D( this );
        res2.dot_mul( -2 );
        res2.sigmoid();
        res2.dot_mul( -1 );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.add( res1.darray.get( i ), res2.darray.get( i ), this.darray.get(i) );
        }
        return this;
    }

}