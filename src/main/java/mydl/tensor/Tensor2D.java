package mydl.tensor;
import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import java.util.ArrayList;

public class Tensor2D extends Tensor {

    private static final long serialVersionUID = 2602694190691785623L;

    /**
     * DMatrixRMaj for storing data.
     * @see org.ejml.data.DMatrixRMaj
     */
    public DMatrixRMaj darray = new DMatrixRMaj();

    /**
     * The default construction method gives a rownum*colnum matrix.
     * (rownum always rank first).
     * @param data Data of this tensor.
     */
    public Tensor2D(double[][] data) {
        this.size = new Tensor_size(data[0].length, data.length);
        // this.darray = new DMatrixRMaj(data[0].length, data.length);
        // for (int i = 0; i < data.length; i++) {
        //     for (int j = 0; j < data[0].length; j++) {
        //         this.darray.set( j, i, data[i][j]);
        //     }
        // }
        this.darray = new DMatrixRMaj(data);
    }

    /**
     * Construct a Tensor2D, rownum = data.length and colnum = 1.
     * @param data Data of this tensor.
     */
    public Tensor2D(double[] data) {
        this.size = new Tensor_size(1, data.length);
        this.darray = new DMatrixRMaj(1, data.length, false, data);
    }

    // /**
    //  * The origin code of DMatrixRMaj.copy() which means deep clone.
    //  * public DMatrixRMaj copy() {
    //  * return new DMatrixRMaj(this);
    //  *    }
    //  */

    /**
     * Construct a Tensor2D from DmatrixRmaj.
     * @param darray The DMatrixRmaj of this tensor2D.
     */
    public Tensor2D(DMatrixRMaj darray) {
        this.size = new Tensor_size( darray.getNumRows(), darray.getNumCols() );
        this.darray = darray.copy();
    }

    /**
     * The default construction method gives a rownum*colnum matrix.
     * @param rownum Row number.
     * @param colnum Column number.
     */
    public Tensor2D(int rownum, int colnum) {
        this.size = new Tensor_size( rownum, colnum );
        this.darray = new DMatrixRMaj(rownum, colnum);
    }

    /**
     * Constructor with a Tensor_size.
     * @param _size Shape of this Tensor2D.
     * @throws MatrixDimensionException if {@code Tensor_size.size != 2}.
     */
    public Tensor2D(Tensor_size _size) throws MatrixDimensionException{
        if(_size.size != 2)
            throw new MatrixDimensionException("Tensor_size must be 2 for Tensor2D");
        this.size = new Tensor_size(_size);
        this.darray = new DMatrixRMaj(this.size.Tensor_length[0], this.size.Tensor_length[1]);
    }

    /**
     * Construct a same Tensor2D(Deep clone).
     * @param t2 Tensor2D to copy.
     */
    public Tensor2D(Tensor2D t2) {
        this.size = new Tensor_size( t2.size.getTensor_length() );
        this.darray = new DMatrixRMaj(t2.darray);
    }

    /**
     * Constructor that convert a Tensor1D to Tensor2D.
     * @param t1 Tensor1D to be converted.
     */
    public Tensor2D(Tensor1D t1){
        this.size = new Tensor_size(t1.darray.numRows, t1.darray.numCols);
        this.darray = new DMatrixRMaj(t1.darray);
    }

    /**
     * Constructor that convert a Tensor3D to Tensor2D. Note this Tensor3D
     * must have a shape of {@code 1 x N x M}.
     * @param t3 Tensor3D to be converted.
     * @throws MatrixDimensionException If the shape of Tensor3D is not
     * {@code 1 x N x M} 
     */
    public Tensor2D(Tensor3D t3) throws MatrixDimensionException{
        if (t3.darray.size() == 1) {
            this.size = new Tensor_size(t3.size.Tensor_length[1], t3.size.Tensor_length[2]);
            this.darray = new DMatrixRMaj(t3.darray.get(0));
        } else {
            throw new MatrixDimensionException(
                "This 3D tensor cannot be converted to 2D Tensor");
        }
    }

    @Override
    public Tensor add (double x) {
        Tensor2D res = new Tensor2D(this);
        CommonOps_DDRM.add(res.darray, x );
        return res;
    }


    @Override
    public Tensor set_zero () {
        this.darray.zero();
        return this;
    }

    /**
     * Deep clone: Res = this
     * @return A new Tensor2D.
     */
    public Tensor clone () {
        return new Tensor2D( this );
    }

    @Override
    public Tensor reshape (Tensor_size new_size) {
        if (new_size.total_size() != this.size().total_size()) {
            throw new MatrixDimensionException( "Reshape tensor size error." );
        }
        switch (new_size.getSize()) {
            case 1:{
                Tensor1D res = new Tensor1D( this.darray.data );
                return res;
            }
            case 2:{
                return this.reshape( new_size.Tensor_length[0], new_size.Tensor_length[1] );
            }
            case 3:{
                double[][][]data = new double[new_size.Tensor_length[0]][new_size.Tensor_length[1]][new_size.Tensor_length[2]];
                int temp_num = 0;
                DMatrixRMaj d2 = new DMatrixRMaj(this.darray.numCols, this.darray.numRows);
                CommonOps_DDRM.transpose( this.darray, d2 );
                for (int i = 0; i < new_size.Tensor_length[0]; i++) {
                    for (int j = 0; j < new_size.Tensor_length[1]; j++) {
                        for (int k = 0; k < new_size.Tensor_length[2]; k++) {
                            data[i][j][k] = d2.data[temp_num];
                            temp_num ++;
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

    @Override
    public Tensor transpose () {
        DMatrixRMaj d1 = new DMatrixRMaj(this.darray );
        CommonOps_DDRM.transpose( d1 );
        Tensor2D res = new Tensor2D( d1 );
        return res;
    }

    @Override
    public Tensor add (Tensor t2) {
        if (t2 instanceof Tensor1D) {
            DMatrixRMaj d1 = new DMatrixRMaj( this.darray );
            CommonOps_DDRM.add(d1, ((Tensor1D) t2).darray, d1);
            return new Tensor2D( d1 );
        }
        else if(t2 instanceof Tensor2D) {
            DMatrixRMaj d1 = new DMatrixRMaj( this.darray );
            CommonOps_DDRM.add(d1, ((Tensor2D) t2).darray, d1);
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

    @Override
    public Tensor subtract (Tensor t2) {
        if (t2 instanceof Tensor2D){
            Tensor2D res = new Tensor2D( this );
            CommonOps_DDRM.subtract(res.darray, ((Tensor2D) t2).darray, res.darray);
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor sizes differ.");
        }
    }

    @Override
    public Tensor subtract (double x) {
        return this.add( (-1)*x );
    }

    @Override
    public Tensor subtracted (double x) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.add( res.darray, x );
        return res;
    }

    @Override
    public Tensor dot_mul (Tensor t2) {
        if(t2 instanceof Tensor2D){
            Tensor2D res = new Tensor2D( this );
            CommonOps_DDRM.elementMult( res.darray, ((Tensor2D) t2).darray, res.darray );
            return res;
        }
        else {
            throw new MatrixDimensionException("Tensor sizes differ.");
        }
    }

    @Override
    public Tensor dot_mul (double times) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( times, res.darray );
        return res;
    }

    @Override
    public Tensor divide(double dividend) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.divide( dividend, res.darray );
        return res;
    }

    // 实际上，scale函数只有double 类型传入，所以...单独拿出int并不能优化 除非 调用ejml底层的代码
    @Override
    public Tensor dot_mul(int int_times) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( int_times, res.darray );
        return res;
    }

    @Override
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

    @Override
    public Tensor divided (Tensor t2) {
        Tensor2D res = new Tensor2D(this);
        CommonOps_DDRM.elementDiv( res.darray, ((Tensor2D)t2).darray );
        return res;
    }

    @Override
    public Tensor pow(double pow) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    @Override
    public Tensor pow(int pow) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementPower( pow, res.darray, res.darray );
        return res;
    }

    @Override
    public Tensor ln () {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.elementLog( res.darray, res.darray );
        return res;
    }

    @Override
    public Tensor sigmoid() {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.elementPower( Math.E, res.darray, res.darray );
        CommonOps_DDRM.add(res.darray, 1);
        CommonOps_DDRM.divide( 1, res.darray );
        return res;
    }

    /**
     * Res{rownum x 1}(i = 1)
     * Res{1 x colnum}(i = 2)
     * @param t1 , input matrix
     * @param i , i = 1 means sum by row, i = 2 means sum by column.
     * @return A new Tensor
     * @deprecated experimental
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
     * @return A new tensor
     * @deprecated experimental
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

    @Override
    public double sum() {
        return CommonOps_DDRM.elementSum( this.darray );
    }
    public static double sum(Tensor2D t1){
        return CommonOps_DDRM.elementSum( t1.darray );
    }

    @Deprecated
    public Tensor reshape (int rownum, int colnum) {
        if (rownum*colnum != this.size().total_size()) {
            throw new MatrixDimensionException( "Reshape tensor size error." );
        }
        Tensor2D res = new Tensor2D( this );
        res.darray.reshape( rownum, colnum, true );
        res.size().Tensor_length[0] = rownum;
        res.size().Tensor_length[1] = colnum;
        return res;
    }


    /**
     * Res = (DMatrixRMaj) this
     * @return DMatrixRMaj.
     */
    public DMatrixRMaj getData() {
        return this.darray;
    }

    @Override
    public Tensor tanh() {
        Tensor2D res1 = new Tensor2D ( this );
        CommonOps_DDRM.scale(-2, res1.darray );
        CommonOps_DDRM.elementPower( Math.E, res1.darray, res1.darray );
        CommonOps_DDRM.add(  res1.darray, 1 );
        CommonOps_DDRM.divide(2, res1.darray);
        CommonOps_DDRM.add(  res1.darray, -1 );
        return res1;
    }

    @Override
    public Tensor relu(double t) {
        Tensor2D res = new Tensor2D( this );
        CommonOps_DDRM.abs(this.darray, res.darray);
        CommonOps_DDRM.add( 0.5*t, this.darray, 0.5*t, res.darray, res.darray );
        return res;
    }

    @Override
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

    @Override
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

    @Override
    public Tensor softmax() {
        Tensor2D res = new Tensor2D( this );
        double elementMax = CommonOps_DDRM.elementMax( res.darray );;
        CommonOps_DDRM.subtract( res.darray, elementMax, res.darray );
        CommonOps_DDRM.elementExp( res.darray, res.darray);
        double sum = CommonOps_DDRM.elementSum( res.darray );
        CommonOps_DDRM.scale( 1.0/sum, res.darray );
        return res;
    }

    @Override
    public double elementMax () {
        return (CommonOps_DDRM.elementMax( this.darray ));
    }
}