package mydl.tensor;
import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

/**
 * One-dimension Tensor. Defaut Tensor1D is {@code 1xN} row vector.
 * <p>If you want to create a {@code Nx1} column vector, please use 
 * {@link Tensor1D#transpose} after calling constructor. If you want
 * to know a whether a Tensor1D is a column vector or row vector, please
 * check {@code Tensor1D.darray.numCols} or {@code Tensor1D.darray.numRows}.
 */
public class Tensor1D extends Tensor {

    private static final long serialVersionUID = 743000385145097795L;

    /**
     * DMatrixRMaj for storing data.
     * @see {@link org.ejml.data.DMatrixRMaj}
     */
    public DMatrixRMaj darray = new DMatrixRMaj();
    /**
     * The default construction method gives a one-colums matrix(an array).
     * @apiNote If the input is not an array, the result may be strange!
     */
    public Tensor1D(double[] data) {
        this.size = new Tensor_size( data.length );
        this.darray = new DMatrixRMaj(1, data.length, false, data);
    }

    /**
     * 
     * @see {@link Tensor}
     * @throws MatrixDimensionException 
     */

    /**
     * Constructor with the {@link org.ejml.data.DMatrixRMaj} data form.
     * @param darray
     * @throws MatrixDimensionException if {@darray}
     */
    public Tensor1D(DMatrixRMaj darray) throws MatrixDimensionException{
        if(darray.numCols != 1 && darray.numCols != 1)
            throw new MatrixDimensionException("DMatrixRMaj given for new Tensor1D is not one-dimension");
        this.size = new Tensor_size( darray.data.length );
        this.darray = new DMatrixRMaj(darray);
    }

    /**
     * Contructor with the length of Tensor1D.
     * @param length Positive integer.
     * @apiNote This contructor produce a Tensor1D with no initial values.
     */
    public Tensor1D(int length) {
        this.size = new Tensor_size( length );
        this.darray = new DMatrixRMaj(1, length);
    }

    /**
     * Constructor with a Tensor_size.
     * @param _size Shape of this Tensor1D.
     * @throws MatrixDimensionException if {@code Tensor_size.size != 1}.
     */
    public Tensor1D(Tensor_size _size) throws MatrixDimensionException{
        if(_size.size != 1)
            throw new MatrixDimensionException("Tensor_size must be 1 for Tensor1D");
        this.size = new Tensor_size(_size);
        this.darray = new DMatrixRMaj(1, this.size.Tensor_length[0]);
    }

    /**
     * Copy constructor.
     * @param t1
     */
    public Tensor1D(Tensor1D t1) {
        this.size = new Tensor_size(t1.size);
        this.darray = new DMatrixRMaj(t1.darray);
    }

    /**
     * Constructor that convert a Tensor2D to Tensor1D. Note this Tensor2D
     * must have a shape of {@code 1xN} or {@code Nx1}. 
     * @param t2
     * @throws MatrixDimensionException If the shape of Tensor2D is not
     * {@code 1xN} or {@code Nx1}. 
     */
    public Tensor1D(Tensor2D t2) throws MatrixDimensionException{
        if (t2.darray.numRows == 1) {
            this.size = new Tensor_size(t2.darray.numCols);
            this.darray = new DMatrixRMaj(t2.darray);
        } else if (t2.darray.numCols == 1) {
            this.size = new Tensor_size(t2.darray.numRows);
            this.darray = new DMatrixRMaj(t2.darray);
        } else {
            throw new MatrixDimensionException(
                "This 2D tensor cannot be converted to 1D Tensor");
        }
    }

    /**
     * Res_{i} = this_{i} + x
     * @param x
     * @return
     */
    public Tensor add (double x) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.add(res.darray, x );
        return res;
    }

    /**
     * Res_{i} = this_{i} - x
     * @param x
     * @return
     */
    public Tensor subtract (double x) {
        return this.add( (-1)*x );
    }

    /**
     * Res_{i} = x - this_{i}
     * @param x
     * @return
     */
    public Tensor subtracted (double x) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.scale( -1, res.darray );
        CommonOps_DDRM.add( res.darray, x );
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
    @Override
    public Tensor subtract (int x) {
        return super.subtract( x );
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

    public Tensor cross_mul(Tensor t2) {
        try {
            if (t2 instanceof Tensor1D){
                if(this.darray.numRows == 1){
                    Tensor1D res = new Tensor1D(t2.size.Tensor_length[0]);
                    CommonOps_DDRM.mult(this.darray, ((Tensor1D) t2).darray, res.darray);
                    return res;
                } else {
                    Tensor2D res = new Tensor2D(this.size.Tensor_length[0], t2.size.Tensor_length[0]);
                    CommonOps_DDRM.mult(this.darray, ((Tensor1D) t2).darray, res.darray);
                    return res;
                }
            } else if (t2 instanceof Tensor2D){
                Tensor1D res = new Tensor1D( this.darray.getNumElements());
                CommonOps_DDRM.mult(this.darray, ((Tensor2D) t2).darray, res.darray);
                return res;
            } else if (t2 instanceof Tensor3D){
                Tensor3D _t2 = (Tensor3D) t2;
                Tensor3D res = new Tensor3D(_t2.size.Tensor_length[0], 
                    this.size.Tensor_length[0], _t2.size.Tensor_length[2]);
                for(int i=0; i<_t2.size.Tensor_length[0]; i++)
                    CommonOps_DDRM.mult(this.darray, _t2.darray.get(i), res.darray.get(i));
                return res;
            } else {
                throw new MatrixDimensionException("Invalid tensor shape");
            }
        } catch (MatrixDimensionException e) {
            throw new MatrixDimensionException("Tensor shapes not compatible");
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
     * Res_{i} = this_{i}^pow
     * @param pow
     * @return
     */
    public Tensor pow(double pow) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementPower( res.darray, pow, res.darray );
        return res;
    }

    /**
     * Res_{i} = this_{i}^pow
     * @param pow
     * @return
     */
    public Tensor pow(int pow) {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementPower( res.darray, pow, res.darray );
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
        if (new_size.total_size() != this.size().total_size()) {
            throw new MatrixDimensionException( "Reshape tensor size error." );
        }
        switch (new_size.size){
            case 1: {
                Tensor1D res = new Tensor1D( this );
                res.darray.reshape( new_size.Tensor_length[0], 1 );
                res.size = new_size;
                return res;
            }
            case 2:{
                DMatrixRMaj d1 = new DMatrixRMaj(this.darray.data);
                d1.reshape( new_size.Tensor_length[0], new_size.Tensor_length[1] );
                Tensor2D res = new Tensor2D( d1 );
                return res;
            }
            case 3:{
                int temp_num = 0;
                double[][][] data = new double[new_size.Tensor_length[0]][new_size.Tensor_length[1]][new_size.Tensor_length[2]];
                for (int i = 0; i < new_size.Tensor_length[0]; i++) {
                    for (int j = 0; j < new_size.Tensor_length[1]; j++) {
                        for (int k = 0; k < new_size.Tensor_length[2]; k++) {
                            data[i][j][k] = this.darray.data[temp_num];
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
     * Res_{i} = tanh(this_{i})
     * @return
     */
    public Tensor tanh() {
        Tensor1D res1 = new Tensor1D( this );
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

    public Tensor softmax() {
        Tensor1D res = new Tensor1D( this );
        CommonOps_DDRM.elementExp( this.darray, res.darray);
        double sum = CommonOps_DDRM.elementSum( res.darray );
        CommonOps_DDRM.divide( 1.0/sum, res.darray );
        return res;
    }

}