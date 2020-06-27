package mydl.layer;

import org.ejml.EjmlUnitTests;
import org.junit.Test;

import mydl.tensor.Tensor1D;

public class ActivationTest {
    @Test
    public void Tanhtest(){
        Tanh x = new Tanh();
        double[] yt = new double[3];
        yt[0]=1;
        yt[1]=0;
        yt[2]=-1;
        double[] zt = new double[3];
        zt[0]=Math.tanh(1);
        zt[1]=Math.tanh(0);
        zt[2]=Math.tanh(-1);
        double[] gt = new double[3];
        gt[0]=(1-Math.pow(Math.tanh(1),2))*1;
        gt[1]=(1-Math.pow(Math.tanh(0),2))*0;
        gt[2]=(1-Math.pow(Math.tanh(-1),2))*(-1);
        Tensor1D y = new Tensor1D(yt);
        Tensor1D z = new Tensor1D(zt);
        Tensor1D g = new Tensor1D(gt);
        EjmlUnitTests.assertEquals(((Tensor1D)x.forward(y)).getData(), z.getData());
        EjmlUnitTests.assertEquals(((Tensor1D)x.backward(y)).getData(), g.getData());
    }

    @Test
    public void ReLU(){
        ReLU x = new ReLU(3);
        double[] yt = new double[3];
        yt[0]=1;
        yt[1]=0;
        yt[2]=-1;
        double[] zt = new double[3];
        zt[0]=3;
        zt[1]=0;
        zt[2]=0;
        double[] gt = new double[3];
        gt[0]=3;
        gt[1]=0;
        gt[2]=0;
        Tensor1D y = new Tensor1D(yt);
        Tensor1D z = new Tensor1D(zt);
        Tensor1D g = new Tensor1D(gt);
        EjmlUnitTests.assertEquals(((Tensor1D)x.forward(y)).getData(), z.getData());
        EjmlUnitTests.assertEquals(((Tensor1D)x.backward(y)).getData(), g.getData());
    }

    @Test
    public void SimoidTest(){
        Sigmoid x = new Sigmoid();
        double[] yt = new double[3];
        yt[0]=1;
        yt[1]=0;
        yt[2]=-1;
        double[] zt = new double[3];
        zt[0]=0.7310585786300049;
        zt[1]=0.5;
        zt[2]=0.2689414213699951;
        double[] gt = new double[3];
        gt[0]=0.19661193324148185*1;
        gt[1]=0.25*0;
        gt[2]=0.19661193324148185*(-1);
        Tensor1D y = new Tensor1D(yt);
        Tensor1D z = new Tensor1D(zt);
        Tensor1D g = new Tensor1D(gt);
        EjmlUnitTests.assertEquals(((Tensor1D)x.forward(y)).getData(), z.getData());
        EjmlUnitTests.assertEquals(((Tensor1D)x.backward(y)).getData(), g.getData());
    }
}