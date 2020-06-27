package mydl.loss;

import static org.junit.Assert.assertEquals;

import org.ejml.EjmlUnitTests;
import org.junit.Test;

import mydl.tensor.Tensor1D;

public class CrossentropyTest {

    @Test
    public void calTest(){
        double[] xx = new double[3];
        xx[0]=0.1;
        xx[1]=0.4;
        xx[2]=0.5;
        double[] yy = new double[3];
        yy[0]=0;
        yy[1]=0;
        yy[2]=1;
        Tensor1D x = new Tensor1D(xx);
        Tensor1D y = new Tensor1D(yy);
        CategoricalCrossentropy cce = new CategoricalCrossentropy();
        assertEquals(cce.loss(x, y), -Math.log(0.5), 1e-5);
        double[] zz = new double[3];
        zz[0]=0;
        zz[1]=0;
        zz[2]=-2;
        Tensor1D z = new Tensor1D(zz);
        EjmlUnitTests.assertEquals(((Tensor1D)cce.grad(x, y)).getData(), z.getData());
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void unmatchexceptionTest1(){
        double[] xx = new double[3];

        double[] yy = new double[4];
        Tensor1D x = new Tensor1D(xx);
        Tensor1D y = new Tensor1D(yy);
        CategoricalCrossentropy cce = new CategoricalCrossentropy();
        System.out.println(cce.loss(x, y));
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void output1exceptionTest1(){
        double[] xx = new double[1];
        double[] yy = new double[1];
        Tensor1D x = new Tensor1D(xx);
        Tensor1D y = new Tensor1D(yy);
        CategoricalCrossentropy cce = new CategoricalCrossentropy();
        System.out.println(cce.loss(x, y));
    }
}