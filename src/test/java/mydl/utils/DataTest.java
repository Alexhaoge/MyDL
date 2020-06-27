package mydl.utils;

import java.util.ArrayList;

import org.ejml.EjmlUnitTests;
import org.junit.Test;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor2D;
import mydl.tensor.Tensor_size;

public class DataTest {
    
    @Test
    public void to_dataTest(){
        ArrayList<Tensor> inputs = new ArrayList<Tensor>();
        ArrayList<Tensor> targets = new ArrayList<Tensor>();
        for(int i=0;i<10;i++){
            inputs.add(Tensor.random(new Tensor_size(4,2)));
            targets.add(Tensor.random(new Tensor_size(4,2)));
        }
        ArrayList<Data> d1 = Data.to_data(inputs, targets, false);
        ArrayList<Data> d2 = Data.to_data(inputs, targets, false);
        for(int i=0;i<10;i++)
            EjmlUnitTests.assertEquals(((Tensor2D)d1.get(i).input).getData(), ((Tensor2D)d2.get(i).input).getData());
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void to_dataTest_exception(){
        ArrayList<Tensor> inputs = new ArrayList<Tensor>();
        ArrayList<Tensor> targets = new ArrayList<Tensor>();
        for(int i=0;i<10;i++){
            inputs.add(Tensor.random(new Tensor_size(4,2)));
            if(i<9)
            targets.add(Tensor.random(new Tensor_size(4,2)));
        }
        ArrayList<Data> d1 = Data.to_data(inputs, targets, false);
        System.out.println(d1.size());
    }
}