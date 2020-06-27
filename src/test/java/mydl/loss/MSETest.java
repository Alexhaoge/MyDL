package mydl.loss;

import static org.junit.Assert.assertEquals;

import org.ejml.EjmlUnitTests;
import org.junit.Test;

import mydl.tensor.Tensor1D;

public class MSETest {
    @Test
    public void calTest(){
        double[] xx = new double[3];
        xx[0]=1;
        xx[1]=2;
        xx[2]=3;
        double[] yy = new double[3];
        yy[0]=3;
        yy[1]=2;
        yy[2]=1;
        Tensor1D x = new Tensor1D(xx);
        Tensor1D y = new Tensor1D(yy);
        MSE _mse = new MSE();
        assertEquals(_mse.loss(x, y), 8.0/3.0, 1e-5);
        double[] zz = new double[3];
        zz[0]=-4.0/3;
        zz[1]=0;
        zz[2]=4.0/3;
        Tensor1D z = new Tensor1D(zz);
        EjmlUnitTests.assertEquals(((Tensor1D)_mse.grad(x, y)).getData(), z.getData());
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void exceptionTest(){
        double[] xx = new double[3];
        xx[0]=1;
        xx[1]=2;
        xx[2]=3;
        double[] yy = new double[4];
        yy[0]=3;
        yy[1]=2;
        yy[2]=1;
        Tensor1D x = new Tensor1D(xx);
        Tensor1D y = new Tensor1D(yy);
        MSE _mse = new MSE();
        System.out.println(_mse.loss(x, y));
    }
}