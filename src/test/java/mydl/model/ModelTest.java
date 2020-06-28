package mydl.model;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import mydl.layer.Dense;
import mydl.layer.Linear1D;
import mydl.layer.ReLU;
import mydl.layer.Reshape;
import mydl.layer.Sigmoid;
import mydl.layer.Softmax;
import mydl.layer.Tanh;
import mydl.loss.CategoricalCrossentropy;
import mydl.optimizer.SGD;
import mydl.tensor.Tensor_size;

public class ModelTest {
    Sequential model = new Sequential();

    @Test(expected = IndexOutOfBoundsException.class)
    public void AddDropLayerTest(){
        while(!model.isEmptyModel()) model.pop();
        model.add(new Dense(new Tensor_size(10), 2));
        model.add(new Softmax());
        assertEquals(model.layers.size(), 2);
        model.pop();
        model.pop();
        assertEquals(model.layers.size(), 0);
        model.pop();
    }

    @Test
    public void compileSuccessTest(){
        while(!model.isEmptyModel()) model.pop();
        model.add(new ReLU());
        model.add(new Dense(new Tensor_size(10,6), 4));
        model.add(new Softmax());
        model.add(new Reshape(new Tensor_size(10,4), new Tensor_size(40)));
        model.add(new Linear1D(40, 5));
        model.compile(new SGD(), new CategoricalCrossentropy());
    }

    @Test (expected = RuntimeException.class)
    public void compileExceptionTest(){
        while(!model.isEmptyModel()) model.pop();
        model.add(new Sigmoid());
        model.add(new Dense(new Tensor_size(10,6), 1));
        model.add(new Tanh());
        model.add(new Reshape(new Tensor_size(4,10), new Tensor_size(40)));
        model.compile(new SGD(), new CategoricalCrossentropy());
    }

    @Test
    public void forwardTest(){

    }

}