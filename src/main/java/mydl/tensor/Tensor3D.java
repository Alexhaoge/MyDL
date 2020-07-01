package mydl.tensor;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;

/**
 * 3D tensor, based on {@code ArrayList<DMatrixRMaj>}.
 */
public class Tensor3D extends Tensor {

    private static final long serialVersionUID = -4968352852257997738L;

    /**
     * ArrayList of DMatrixRMaj for storing data.
     * @see {@link org.ejml.data.DMatrixRMaj}, {@link ArrayList}
     */
    public ArrayList<DMatrixRMaj> darray = new ArrayList<DMatrixRMaj>();

    /**
     * 
     * @param a
     * @param N
     */
    public Tensor3D(ArrayList<double[][]> a, int N) {
        for (int i = 0; i < N; i++) {
            DMatrixRMaj temp = new DMatrixRMaj(a.get( i ));
            this.darray.add( temp );
        }
        this.size = new Tensor_size(N, a.get(0).length, a.get(0)[0].length);

    }

    //Cations! If the input is not an array, the result may be strange.
    public Tensor3D(ArrayList<DMatrixRMaj> d) {
        for (int i = 0; i < d.size(); i++) {
            this.darray.add( d.get( i ).copy() );
        }
        this.size = new Tensor_size(d.size(), d.get(0).getNumRows(), d.get(0).getNumCols());
    }

    /**
     * Constructor with shape in three-integer form
     * @param rowNum
     * @param colNum
     * @param N
     */
    public Tensor3D(int N, int rowNum, int colNum) {
        for (int i = 0; i < N; i++) {
            this.darray.add(new DMatrixRMaj(rowNum, colNum));
        }
        this.size = new Tensor_size(N, rowNum, colNum);
    }

    /**
     * Constructor with a Tensor_size.
     * @param _size Shape of this Tensor3D.
     * @throws MatrixDimensionException if {@code Tensor_size.size != 3}.
     */
    public Tensor3D(Tensor_size _size) throws MatrixDimensionException{
        if (_size.size != 3)
            throw new MatrixDimensionException("Tensor_size must be 3 for Tensor3D");
        this.size = new Tensor_size(_size);
        for (int i = 0; i < this.size.Tensor_length[0]; i++) {
            this.darray.add(new DMatrixRMaj(this.size.Tensor_length[1], this.size.Tensor_length[2]));
        }
    }

    /**
     * 
     * @param data
     */
    public Tensor3D(double[][][]data) {
        int N = data.length;
        int colnum = data[0].length;
        int rownum = data[0][0].length;
        for (int i = 0; i < N; i++) {
            DMatrixRMaj d1 = new DMatrixRMaj(data[i]);
            this.darray.add( d1 );
        }
        this.size = new Tensor_size( N, colnum, rownum );
    }    

    public Tensor3D(Tensor3D t3) {
        for (int i = 0; i < t3.darray.size(); i++) {
            this.darray.add(t3.darray.get(i).copy());
        }
        this.size = new Tensor_size(t3.size);
    }

    public Tensor3D(Tensor2D t2){
        this.size = new Tensor_size(1, t2.size.Tensor_length[0], t2.size.Tensor_length[1]);
        this.darray.add(new DMatrixRMaj(t2.darray));
    }

    public Tensor add (double x) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.add( res.darray.get( i ), x );
        }
        return res;
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
        if (new_size.total_size() != this.size().total_size()) {
            throw new MatrixDimensionException( "Reshape tensor size error." );
        }
        switch (new_size.size){
            case 1: {
                DMatrixRMaj d1 = new DMatrixRMaj(new_size.total_size(), 1);
                for (int i = 0; i < this.darray.size(); i++) {
                    for (int j = 0; j < this.darray.get( 0 ).getNumElements(); j++) {
                        d1.data[ i * this.darray.get(0).getNumElements() + j ] = this.darray.get( i ).data[j];
                    }
                }
                Tensor1D res = new Tensor1D( d1 );
                return res;
            }
            case 2:{
                DMatrixRMaj d1 = new DMatrixRMaj(new_size.Tensor_length[0], new_size.Tensor_length[1]);
                for (int i = 0; i < this.darray.size(); i++) {
                    for (int j = 0; j < this.darray.get( 0 ).getNumElements(); j++) {
                        d1.data[i*this.darray.get( 0 ).getNumElements()+j] = this.darray.get( i ).data[j];
                    }
                }
                Tensor2D res = new Tensor2D( d1 );
                return res;
            }
            case 3:{
                int temp_num1 = 0, temp_num2 = 0;
                double[][][] data = new double[new_size.Tensor_length[0]][new_size.Tensor_length[1]][new_size.Tensor_length[2]];
                double[] temp_data = new double[this.darray.size()*this.darray.get( 0 ).getNumElements()];
                for (int i = 0; i < this.darray.size(); i++) {
                    for(int j = 0; j < this.darray.get( 0 ).getNumElements(); j++) {
                        temp_data[temp_num1] = this.darray.get( i ).data[j];
                        temp_num1 ++;
                    }
                }
                for (int i = 0; i < new_size.getTensor_length()[0]; i++) {
                    for (int j = 0; j < new_size.getTensor_length()[1]; j++) {
                        for (int k = 0; k < new_size.getTensor_length()[2]; k++) {
                            data[i][j][k] = temp_data[temp_num2];
                            temp_num2 ++;
                        }
                    }
                }

                Tensor3D res = new Tensor3D( data );
                return res;
            }
            default:{
                throw new MatrixDimensionException("Dimension errors");
            }
        }
    }

    public Tensor_size size () {
        Tensor_size res = new Tensor_size(  this.darray.size(), this.darray.get( 0 ).getNumRows(), this.darray.get( 0 ).getNumCols() );
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
            DMatrixRMaj minuend = new DMatrixRMaj( ((Tensor2D) t2).darray);
            for (int i = 0; i < this.darray.size(); i++){
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

    public Tensor subtract (double x) {
        return this.add( (-1)*x );
    }
    
    public Tensor subtracted (double x) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.scale( -1, res.darray.get( i ) );
            CommonOps_DDRM.add( res.darray.get( i ), x );
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

    public Tensor dot_mul (double times) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.scale( times, res.darray.get( i ) );
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
            CommonOps_DDRM.elementPower( res.darray.get( i ), pow, res.darray.get( i ) );
        }
        return res;
    }

    public Tensor pow(int pow) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.elementPower( res.darray.get( i ), pow, res.darray.get( i ) );
        }
        return res;
    }

    public Tensor ln () {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            CommonOps_DDRM.elementLog( res.darray.get( i ), res.darray.get( i ) );
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

    public Tensor reshape (int rownum, int colnum, int N) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < N; i++) {
            this.darray.get( i ).reshape( rownum, colnum, true );
            res.darray.add(this.darray.get( i ).copy());
        }
        return res;
    }

    public Tensor reshape (int rownum, int colnum) {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < this.darray.size(); i++) {
            this.darray.get( i ).reshape( rownum, colnum, true );
            res.darray.add(this.darray.get( i ).copy());
        }
        return res;
    }

   public Tensor reshape (int N, int rownum, int colnum, int dimselect) {
        switch (dimselect) {
            case 1:{
                Tensor1D res = new Tensor1D( this.darray.get( 0 ).getNumRows() );
                for (int i = 0; i < this.darray.get( 0 ).getNumRows(); i++) {
                    res.getData().data[i] =  this.darray.get( N ).get( i, colnum );
                }
                return res;
            }
            case 2:{
                Tensor2D res = new Tensor2D( this.darray.get( N ));
                res.reshape( rownum, colnum );
                return res;
            }
            case -1:{
                Tensor1D res = new Tensor1D( this.darray.get( 0 ).getNumCols() );
                for (int i = 0; i < this.darray.get( 0 ).getNumCols(); i++) {
                    res.getData().data[i] =  this.darray.get( N ).get( rownum, i );
                }
                return res;
            }
            default:{
                throw new MatrixDimensionException("Tensor size error.");
            }
        }
    }

    public ArrayList<DMatrixRMaj> getData() {
        return this.darray;
    }

    public Tensor devide (double dividend) {
        Tensor3D res = new Tensor3D(this);
        for (int i = 0; i < this.darray.size(); i++) {
            CommonOps_DDRM.divide( dividend, res.darray.get( i ));
        }
        return res;
    }

    public Tensor tanh() {
        Tensor3D res1 = new Tensor3D( this );
        for (int i = 0; i < res1.darray.size(); i++) {
            CommonOps_DDRM.scale(2, res1.darray.get( i ) );
            CommonOps_DDRM.elementPower( Math.E, res1.darray.get( i ), res1.darray.get( i ) );
            DMatrixRMaj d2 = new DMatrixRMaj(res1.darray.get( i ));
            CommonOps_DDRM.add(res1.darray.get( i ), 1);
            CommonOps_DDRM.add(d2, -1);
            CommonOps_DDRM.elementDiv( d2, res1.darray.get( i ), res1.darray.get( i ) );
        }
        return res1;

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

    public Tensor divide (double x) {
        return this.dot_mul( 1.0/x );
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
    
    public Tensor softmax() {
        Tensor3D res = new Tensor3D( this );
        for (int i = 0; i < res.darray.size(); i++) {
            double elementMax = CommonOps_DDRM.elementMax( res.darray.get( i ) );;
            CommonOps_DDRM.subtract( res.darray.get( i ), elementMax, res.darray.get( i ) );
            CommonOps_DDRM.elementExp( res.darray.get( i ), res.darray.get( i ));
            double sum = CommonOps_DDRM.elementSum( res.darray.get( i ) );
            CommonOps_DDRM.scale( 1.0/sum, res.darray.get( i ) );
            return res;
        }
        return res;
    }

    public double elementMax () {
        double max = CommonOps_DDRM.elementMax( this.darray.get( 0 ) );
        for (int i = 0; i < this.darray.size(); i++) {
            double temp_max = CommonOps_DDRM.elementMax( this.darray.get( i ));
            if ( temp_max  > max) {
                max = temp_max;
            }
        }
        return max;
    }
}