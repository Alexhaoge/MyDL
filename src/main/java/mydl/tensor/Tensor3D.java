package mydl.tensor;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;


public class Tensor3D extends Tensor {

    private static final long serialVersionUID = -4968352852257997738L;

    /**
     * ArrayList of DMatrixRMaj for storing data.
     * @see {@link org.ejml.data.DMatrixRMaj}, {@link ArrayList}
     */
    ArrayList<DMatrixRMaj> darray = new ArrayList<DMatrixRMaj>();
    // The default construction method gives a one-colums matrix(an array).
//    public Tensor3D(double[][] data) {
//        this.darray = new DMatrixRMaj(data);
//    }

    public Tensor3D(ArrayList<double[][]> a, int N) {
        for (int i = 0; i < N; i++) {
            DMatrixRMaj temp = new DMatrixRMaj(a.get( i ));
            this.darray.add( temp );
        }
        this.size = new Tensor_size( a.get( 0 ).length,1, N);

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

    public Tensor set_zero () {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++){
            res.darray.get( i ).zero();
        }
        return res;
    }

    public Tensor clone () {
        return new Tensor3D( this );
    }

    public Tensor reshape (Tensor_size new_size) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < new_size.getTensor_length()[2]; i++){
            res.darray.get( i ).reshape( new_size.getTensor_length()[0], new_size.getTensor_length()[1], true );
        }
        return res;
    }

    public Tensor_size size () {
        Tensor_size res = new Tensor_size( this.darray.get( 0 ).getNumRows(), this.darray.get( 0 ).getNumCols(), this.darray.size() );
        return res;
    }

    public Tensor transpose () {
        Tensor3D res = new Tensor3D( this.darray.get( 0 ).getNumCols(), this.darray.get( 0 ).getNumRows(), this.darray.size());
        for (int i = 0; i < this.darray.size(); i++) {
            DMatrixRMaj d1;
            d1 = new DMatrixRMaj(this.darray.get( i ));
            res.darray.add( d1 );
        }
        return res;
    }

    public Tensor add (Tensor t2) {
        if (t2 instanceof Tensor3D) {
            if (((Tensor3D) t2).darray.size() != this.darray.size()) {
                throw new MatrixDimensionException("Tensor sizes differ.");
            }
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i < this.darray.size(); i++){
                CommonOps_DDRM.add( res.darray.get( i ), ((Tensor3D) t2).darray.get( i ), res.darray.get( i ) );
            }
            return res;
        }
        else if (t2 instanceof Tensor2D) {
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i < this.darray.size(); i++){
                CommonOps_DDRM.add( res.darray.get( i ), ((Tensor2D) t2).darray, res.darray.get( i ) );
            }
            return res;
        }
        else if (t2 instanceof Tensor1D) {
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i < this.darray.size(); i++){
                CommonOps_DDRM.add( res.darray.get( i ), ((Tensor1D) t2).darray, res.darray.get( i ) );
            }
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor input error.");
        }
    }

    public Tensor subtract (Tensor x) {
        return this.add( x );
    }

    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    public static Tensor subtracted (Tensor3D t1, double subtract) {
        return t1.subtracted( subtract );
    }

    public Tensor subtracted (double subtract) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.scale( -1, res.darray.get( i ) );
            CommonOps_DDRM.add( res.darray.get( i ), subtract );
        }
        return res;
    }

    public Tensor dot_mul (Tensor t2) {
        if (t2 instanceof Tensor3D){
            if (((Tensor3D) t2).darray.size() != this.darray.size()) {
                throw new MatrixDimensionException("Tensor sizes differ.");
            }else {
                Tensor3D res = new Tensor3D( this );
                for (int i = 0; i<this.darray.size(); i++) {
                    CommonOps_DDRM.elementMult(res.darray.get( i ), ((Tensor3D) t2).darray.get(i), res.darray.get( i ));
                }
                return res;
            }
        }
        else if(t2 instanceof Tensor2D){
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.elementMult(res.darray.get( i ), ((Tensor2D) t2).darray, res.darray.get( i ));
            }
            return res;
        }
        else if(t2 instanceof Tensor1D){
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.elementMult(res.darray.get( i ), ((Tensor1D) t2).darray, res.darray.get( i ));
            }
            return res;
        }
        else{
            throw new MatrixDimensionException("dot_mul Tensor input error.");
        }

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
                DMatrixRMaj d1 = new DMatrixRMaj( this.darray.get( 0 ).getNumRows(), this.darray.size() );
                d1.zero();
                for (int j = 0; j < this.darray.size(); j++){
                    DMatrixRMaj temp = new DMatrixRMaj(this.darray.get( i ).getNumRows(), 1);
                    CommonOps_DDRM.sumRows( this.darray.get( i ), temp );
                    for (int k = 0; k < temp.data.length; k++){
                        d1.data[j*this.darray.get( 0 ).getNumRows()+k] = temp.data[k];
                    }
                }
                Tensor2D res = new Tensor2D( d1 );
                return res;
            }
            case 2:{
                DMatrixRMaj d1 = new DMatrixRMaj( this.darray.get( 0 ).getNumCols(), this.darray.size() );
                d1.zero();
                for (int j = 0; j < this.darray.size(); j++){
                    DMatrixRMaj temp = new DMatrixRMaj(this.darray.get( i ).getNumCols(), 1);
                    CommonOps_DDRM.sumCols( this.darray.get( i ), temp );
                    for (int k = 0; k < temp.data.length; k++){
                        d1.data[j*this.darray.get( 0 ).getNumCols()+k] = temp.data[k];
                    }
                }
                Tensor2D res = new Tensor2D( d1 );
                return res;
            }
            case 3:{
                DMatrixRMaj d1 = new DMatrixRMaj(this.darray.get( 0 ).getNumRows(), this.darray.get( 0 ).getNumCols());
                d1.zero();
                for (int j = 0; j < this.darray.size(); j++ ){
                    d1.data[j] += this.darray.get( i ).data[j];
                }
                Tensor2D res = new Tensor2D( d1 );
                return res;
            }
            default:{
                throw new MatrixDimensionException("Tensor size input error.");
            }
        }

    }

    @Override
    public Tensor sum (int axis, int... _axis) {
        return null;
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
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            this.darray.get( i ).reshape( x, y, true );
            res.darray.add(this.darray.get( i ).copy());
        }
        return res;
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
    public Tensor cross_mul(Tensor t2) {
        if (t2 instanceof Tensor3D){
            if (((Tensor3D) t2).darray.size() != this.darray.size()) {
                throw new MatrixDimensionException("Tensor sizes differ.");
            }else {
                Tensor3D res = new Tensor3D( this );
                for (int i = 0; i<this.darray.size(); i++) {
                    CommonOps_DDRM.mult(res.darray.get( i ), ((Tensor3D) t2).darray.get(i), res.darray.get( i ));
                }
                return res;
            }
        }
        else if(t2 instanceof Tensor2D){
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.mult(res.darray.get( i ), ((Tensor2D) t2).darray, res.darray.get( i ));
            }
            return res;
        }
        else if(t2 instanceof Tensor1D){
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.mult(res.darray.get( i ), ((Tensor1D) t2).darray, res.darray.get( i ));
            }
            return res;
        }
        else{
            throw new MatrixDimensionException("cross_mul Tensor input error.");
        }
    }

    public Tensor divide (double x) {
        return this.dot_mul( 1.0/x );
    }

    public static Tensor cross_mul(Tensor3D t1, Tensor3D t2) {
        if (t2.darray.size() != t1.darray.size()) {
            throw new MatrixDimensionException("Tensor sizes differ.");
        }else {
            Tensor3D res = new Tensor3D( t1 );
            for (int i = 0; i<t1.darray.size(); i++) {
                CommonOps_DDRM.mult(t1.darray.get( i ), t2.darray.get(i), res.darray.get( i ));
            }
            return res;
        }
    }
    public Tensor relu(double t) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.abs(this.darray.get( i ), res.darray.get( i ));
            CommonOps_DDRM.add( 0.5*t, this.darray.get( i ), 0.5*t, res.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }
    public Tensor DiffReLU(double t) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.abs( this.darray.get( i ), res.darray.get( i ) );
            CommonOps_DDRM.add( this.darray.get( i ), res.darray.get( i ), res.darray.get( i ) );
            CommonOps_DDRM.elementDiv( this.darray.get( i ), res.darray.get( i ), res.darray.get( i ) );
        }
        for (int i = 0; i < res.darray.size(); i++) {
            for (int j = 0; j < res.darray.get(i).getData().length; j++) {
                if (Double.isNaN( res.darray.get(i).getData()[j] )) {
                    res.darray.get(i).getData()[j] = 0;
                }
            }
        }
        return res;
    }
    public Tensor sgn() {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.abs(this.darray.get( i ), res.darray.get( i ));
            CommonOps_DDRM.elementDiv( this.darray.get( i ), res.darray.get( i ), res.darray.get( i ) );
            for (int j = 0; j < res.darray.get( i ).getData().length; j++) {
                if (Double.isNaN( res.darray.get( i ).getData()[j] )) {
                    res.darray.get( i ).getData()[j] = 0;
                }
            }
        }
        return  res;
    }

}