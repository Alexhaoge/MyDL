package mydl.tensor;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.lang.Character.UnicodeBlock;
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

    /**
     * Res_{i, j, k} = t1_{i, j, k} + addtion
     * @param t1
     * @param addtion
     * @return
     */
    public static Tensor add (Tensor3D t1, double addtion) {
        Tensor3D res = new Tensor3D(t1);
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.add( res.darray.get( i ), addtion );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = this_{i, j, k} + addtion
     * @param addtion
     * @return
     */
    public Tensor add (double addtion) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.add( res.darray.get( i ), addtion );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = t1_{i, j, k} - minuend
     * @param t1
     * @param minuend
     * @return
     */
    public static Tensor subtract (Tensor3D t1, double minuend) {
        return Tensor3D.add( t1, (-1)*minuend );
    }

    /**
     * Res_{i, j, k} = 0
     * @return
     */
    public Tensor set_zero () {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++){
            res.darray.get( i ).zero();
        }
        return res;
    }

    /**
     * Deep clone: Res = this
     * @return
     */
    public Tensor clone () {
        return new Tensor3D( this );
    }

    /**
     * Res: rownum, colnum, N: new_size.Tensor_length[0], [1], [2]
     * @param new_size
     * @return
     */
    public Tensor reshape (Tensor_size new_size) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < new_size.getTensor_length()[2]; i++){
            res.darray.get( i ).reshape( new_size.getTensor_length()[0], new_size.getTensor_length()[1], true );
        }
        return res;
    }

    /**
     * Res = [rownum of this, colnum, N]
     * @return
     */
    public Tensor_size size () {
        Tensor_size res = new Tensor_size( this.darray.get( 0 ).getNumRows(), this.darray.get( 0 ).getNumCols(), this.darray.size() );
        return res;
    }

    /**
     * Res{, , k} = this{, , k}^T
     * @return Transpose by the first two dimensions
     */
    public Tensor transpose () {
        Tensor3D res = new Tensor3D( this.darray.get( 0 ).getNumCols(), this.darray.get( 0 ).getNumRows(), this.darray.size());
        for (int i = 0; i < this.darray.size(); i++) {
            DMatrixRMaj d1;
            d1 = new DMatrixRMaj(this.darray.get( i ));
            res.darray.add( d1 );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = this_{i, j, k}+t2_{i}(dim(t2) = 1)
     * Res_{i, j, k} = this_{i, j, k}+t2_{i, j}(dim(t2) = 2)
     * Res_{i, j, k} = this_{i, j, k}+t2_{i, j, k}(dim(t2) = 3)
     * @param t2
     * @return
     */
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

    /**
     * Res_{i, j, k} = this_{i, j, k}-t2_{i}(dim(t2) = 1)
     * Res_{i, j, k} = this_{i, j, k}-t2_{i, j}(dim(t2) = 2)
     * Res_{i, j, k} = this_{i, j, k}-t2_{i, j, k}(dim(t2) = 3)
     * @param t2
     * @return
     */
    public Tensor subtract (Tensor t2) {
        if (t2 instanceof Tensor3D) {
            if (((Tensor3D) t2).darray.size() != this.darray.size()) {
                throw new MatrixDimensionException("Tensor sizes differ.");
            }
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i < this.darray.size(); i++){
                DMatrixRMaj minuend = new DMatrixRMaj( ((Tensor3D) t2).darray.get( i ));
                CommonOps_DDRM.scale( -1, minuend );
                CommonOps_DDRM.add( res.darray.get( i ),  minuend, res.darray.get( i ));
            }
            return res;
        }
        else if (t2 instanceof Tensor2D) {
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i < this.darray.size(); i++){
                DMatrixRMaj minuend = new DMatrixRMaj( ((Tensor2D) t2).darray);
                CommonOps_DDRM.scale( -1, minuend );
                CommonOps_DDRM.add( res.darray.get( i ), minuend, res.darray.get( i ) );
            }
            return res;
        }
        else if (t2 instanceof Tensor1D) {
            Tensor3D res = new Tensor3D( this );
            DMatrixRMaj minuend = new DMatrixRMaj(res.getData().get( 0 ).getNumRows(), res.getData().get( 0 ).getNumCols());
            DMatrixRMaj Unitvector = new DMatrixRMaj(1, res.getData().get( 0 ).getNumCols());
            for( int j = 0; j < Unitvector.getData().length; j++) {
                Unitvector.getData()[j] = 1;
            }
            for (int i = 0; i < this.darray.size(); i++){
                CommonOps_DDRM.mult(minuend, Unitvector, minuend);
                CommonOps_DDRM.add( res.darray.get( i ), ((Tensor1D) t2).darray, res.darray.get( i ) );
            }
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor input error.");
        }
    }

    /**
     * Res_{i, j, k} = this_{i, j ,k} - minuend
     * @param minuend
     * @return
     */
    public Tensor subtract (double minuend) {
        return this.add( (-1)*minuend );
    }

    /**
     * Res_{i, j, k} = subtract - t1_{i, j ,k}
     * @param t1
     * @param subtract
     * @return
     */
    public static Tensor subtracted (Tensor3D t1, double subtract) {
        return t1.subtracted( subtract );
    }

    /**
     * Res_{i, j, k} = subtract - this_{i, j ,k}
     * @param subtract
     * @return
     */
    public Tensor subtracted (double subtract) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.scale( -1, res.darray.get( i ) );
            CommonOps_DDRM.add( res.darray.get( i ), subtract );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = this_{i, j, k}*t2_{i}(dim(t2) = 1)
     * Res_{i, j, k} = this_{i, j, k}*t2_{i, j}(dim(t2) = 2)
     * Res_{i, j, k} = this_{i, j, k}*t2_{i, j, k}(dim(t2) = 3)
     * @param t2
     * @return
     */
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
            DMatrixRMaj mul = new DMatrixRMaj(res.getData().get( 0 ).getNumRows(), res.getData().get( 0 ).getNumCols());
            DMatrixRMaj Unitvector = new DMatrixRMaj(1, res.getData().get( 0 ).getNumCols());
            for( int j = 0; j < Unitvector.getData().length; j++) {
                Unitvector.getData()[j] = 1;
            }
            CommonOps_DDRM.mult(mul, Unitvector, mul);
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.elementMult(res.darray.get( i ), mul, res.darray.get( i ));
            }
            return res;
        }
        else{
            throw new MatrixDimensionException("dot_mul Tensor input error.");
        }

    }

    /**
     * Res_{i, j, k} = t1_{i, j, k}*times
     * @param t1
     * @param times
     * @return
     */
    public static Tensor dot_mul (Tensor3D t1, double times) {
        Tensor3D res = new Tensor3D( t1 );
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.scale( times, res.darray.get( i ) );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = t1_{i, j, k}*times
     * @param t1
     * @param times
     * @return
     */
    public Tensor dot_mul (double times) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.scale( times, res.darray.get( i ) );
        }
        return res;
    }


    // 实际上，scale函数只有double 类型传入，所以...单独拿出int并不能优化 除非 调用ejml底层的代码
    /**
     * Res_{i, j, k} = t1_{i, j, k}*int_times
     * @param t1
     * @param int_times
     * @return
     */
    public static Tensor dot_mul(Tensor3D t1, int int_times) {
        Tensor3D res = new Tensor3D( t1 );
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.scale( int_times, res.darray.get( i ) );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = this_{i, j, k}*int_times
     * @param int_times
     * @return
     */
    public Tensor dot_mul(int int_times) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.scale( int_times, res.darray.get( i ) );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = this_{i, j, k}^pow
     * @param pow
     * @return
     */
    public Tensor pow(double pow) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, res.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = this_{i, j, k}^pow
     * @param pow
     * @return
     */
    public Tensor pow(int pow) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, res.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = log(this_{i, j, k})
     * @return
     */
    public Tensor ln () {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.elementLog( res.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = this_{i, j, k}^pow
     * @param t1
     * @param pow
     * @return
     */
    public static Tensor pow(Tensor3D t1 , double pow) {
        Tensor3D res = new Tensor3D( t1 );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.elementPower( pow, t1.darray.get( i ), res.darray.get(i) );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = sigmoid(this_{i, j, k})
     * @return
     */
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

    /**
     * Res_{i, j, k} = sigmoid(t1_{i, j, k})
     * @param t1
     * @return
     */
    public static Tensor sigmoid(Tensor3D t1) {
        return t1.sigmoid();
    }

    /**
     * Res{rownum x 1 x N}(i = 1)
     * Res{1 x colnum x N}(i = 2)
     * Res{rownum x colnum x 1}(i = 3)
     * @param i , i = 1 means sum by row, i = 2 means sum by column, i = 3 means by N.
     */
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

    /**
     * Res = \sum{t1_{i, j}}
     * @param t1
     * @return
     */
    public static double sum(Tensor3D t1) {
        double tempsum = 0;
        for (int i = 0; i < t1.darray.size(); i++) {
            tempsum += CommonOps_DDRM.elementSum( t1.darray.get( i ) );
        }
        return tempsum;
    }

    /**
     * Res = \sum{this_{i, j}}
     * @return
     */
    public double sum() {
        double tempsum = 0;
        for (int i = 0; i < this.darray.size(); i++) {
            tempsum += CommonOps_DDRM.elementSum( this.darray.get( i ) );
        }
        return tempsum;
    }

    /**
     * Res = this{rownum x colnum x N}
     * @param rownum
     * @param colnum
     * @param N
     * @return
     */
    public Tensor reshape (int rownum, int colnum, int N) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < N; i++) {
            this.darray.get( i ).reshape( rownum, colnum, true );
            res.darray.add(this.darray.get( i ).copy());
        }
        return res;
    }

    /**
     * Res = this{rownum x colnum x Nofthis}
     * @param rownum
     * @param colnum
     * @return
     */
    public Tensor reshape (int rownum, int colnum) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            this.darray.get( i ).reshape( rownum, colnum, true );
            res.darray.add(this.darray.get( i ).copy());
        }
        return res;
    }

    public Tensor reshape (int x) {
        return null;
    }

    /**
     * Res = (DMatrixRMaj) this
     * @return
     */
    public ArrayList<DMatrixRMaj> getData() {
        return this.darray;
    }

    /**
     * Res_{i, j, k} = dividend/this_{i, j, k}
     * @param dividend
     * @return
     */
    public Tensor devide (double dividend) {
        Tensor3D res = new Tensor3D(this);
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.divide( dividend, res.darray.get( i ));
        }
        return res;
    }

    /**
     * Res_{i, j, k} = dividend/t1_{i, j, k}
     * @param dividend
     * @return
     */
    public static Tensor devide(double dividend, Tensor3D t1) {
        Tensor3D res = new Tensor3D( t1);
        for (int i = 0; i < t1.darray.size(); i++) {
            CommonOps_DDRM.divide( dividend, t1.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }

    /**
     * Res_{i, j, k} = tanh(this_{i, j, k})
     * @return
     */
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

    /**
     * Res_{rownum1 x colnum2, k} = this_{rownum1 x colnum1, k}*t2_{colnum1 x colnum2, k}(dim(t2) = 3)
     * Res_{rownum1 x colnum2, k} = this_{rownum1 x colnum1, k}*t2_{colnum1 x colnum2}(dim(t2) = 2)
     * Res_{rownum1 x 1, k} = this_{rownum1 x colnum1, k}*t2_{colnum1 x 1}(dim(t2) = 1)
     * @param t2
     * @return
     */
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
            DMatrixRMaj mul = new DMatrixRMaj(res.getData().get( 0 ).getNumRows(), res.getData().get( 0 ).getNumCols());
            DMatrixRMaj Unitvector = new DMatrixRMaj(1, res.getData().get( 0 ).getNumCols());
            for( int j = 0; j < Unitvector.getData().length; j++) {
                Unitvector.getData()[j] = 1;
            }
            CommonOps_DDRM.mult(mul, Unitvector, mul);
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.mult(res.darray.get( i ), mul, res.darray.get( i ));
            }
            return res;
        }
        else{
            throw new MatrixDimensionException("cross_mul Tensor input error.");
        }
    }

    /**
     * Res_{i, j, k} = t2_{i}/this_{i, j, k}(dim(t2) = 1)
     * Res_{i, j, k} = t2_{i, j}/this_{i, j, k}(dim(t2) = 2)
     * Res_{i, j, k} = t2_{i, j, k}/this_{i, j , k}(dim(t2) = 3)
     * @param t2
     * @return
     */
    public Tensor divided (Tensor t2) {
        if (t2 instanceof Tensor3D){
            if (((Tensor3D) t2).darray.size() != this.darray.size()) {
                throw new MatrixDimensionException("Tensor sizes differ.");
            }else {
                Tensor3D res = new Tensor3D(this);
                for (int i = 0; i < res.darray.size(); i++)  {
                    CommonOps_DDRM.elementDiv( res.darray.get(i), ((Tensor3D)t2).darray.get(i) );
                }
                return res;
            }
        }
        else if(t2 instanceof Tensor2D){
            Tensor3D res = new Tensor3D( this );
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.elementDiv( res.darray.get(i), ((Tensor2D)t2).darray );
            }
            return res;
        }
        else if(t2 instanceof Tensor1D){
            Tensor3D res = new Tensor3D( this );
            DMatrixRMaj divide = new DMatrixRMaj(res.getData().get( 0 ).getNumRows(), res.getData().get( 0 ).getNumCols());
            DMatrixRMaj Unitvector = new DMatrixRMaj(1, res.getData().get( 0 ).getNumCols());
            for( int j = 0; j < Unitvector.getData().length; j++) {
                Unitvector.getData()[j] = 1;
            }
            CommonOps_DDRM.mult(divide, Unitvector, divide);
            for (int i = 0; i<this.darray.size(); i++) {
                CommonOps_DDRM.elementDiv( res.darray.get(i), ((Tensor1D)t2).darray );
            }
            return res;
        }
        else{
            throw new MatrixDimensionException("cross_mul Tensor input error.");
        }
    }

    /**
     * Res_{i, j, k} = 1.0/this_{i, j, k}
     * @param x
     * @return
     */
    public Tensor divide (double x) {
        return this.dot_mul( 1.0/x );
    }

    /**
     * Res_{rownum1 x colnum2, k} = t1_{rownum1 x colnum1, k}*t2_{colnum1 x colnum2, k}(dim(t2) = 3)
     * Res_{rownum1 x colnum2, k} = t1_{rownum1 x colnum1, k}*t2_{colnum1 x colnum2}(dim(t2) = 2)
     * Res_{rownum1 x 1, k} = t1_{rownum1 x colnum1, k}*t2_{colnum1 x 1}(dim(t2) = 1)
     * @param t1
     * @param t2
     * @return
     */
    public static Tensor cross_mul(Tensor3D t1, Tensor3D t2) {
        return t1.dot_mul( t2 );
    }

    /**
     * Res_{i} = t*this_{i} (this_{i} > 0)
     * Res_{i} = 0 (this_{i} <= 0)
     * @param t
     * @return
     */
    public Tensor relu(double t) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.abs(this.darray.get( i ), res.darray.get( i ));
            CommonOps_DDRM.add( 0.5*t, this.darray.get( i ), 0.5*t, res.darray.get( i ), res.darray.get( i ) );
        }
        return res;
    }

    /**
     * Res_{i} = t (this_{i} > 0)
     * Res_{i} = 0 (this_{i} <= 0)
     * @param t
     * @return
     */
    public Tensor DiffReLU(double t) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            for (int j = 0; j < res.darray.get(i).getData().length; j++) {
                if (res.darray.get(i).getData()[i] > 0) {
                    res.darray.get(i).getData()[i] = t;
                }else{
                    res.darray.get(i).getData()[i] = 0;
                }
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
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            for (int j = 0; j < res.darray.get( i ).getData().length; j++) {
                if (res.darray.get( i ).getData()[i] > 0) {
                    res.darray.get( i ).getData()[i] = 1;
                }else if ( res.darray.get( i ).getData()[i] < 0) {
                    res.darray.get( i ).getData()[i] = -1;
                }
            }
        }
        return  res;
    }

}