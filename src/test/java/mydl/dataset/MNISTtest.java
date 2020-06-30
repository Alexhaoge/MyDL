package mydl.dataset;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;

import org.junit.Test;

import mydl.tensor.Tensor;

public class MNISTtest {

    @Test
    public void readtest(){
        ArrayList<Tensor> train_image = MNIST.readTrainImage2D();
        assertEquals(train_image.size(), 60000);
        ArrayList<Tensor> train_label = MNIST.readTrainLabel();
        assertEquals(train_label.size(), 60000);
        ArrayList<Tensor> test_image = MNIST.readTestImage2D();
        assertEquals(test_image.size(), 10000);
        ArrayList<Tensor> test_label = MNIST.readTestLabel();
        assertEquals(test_label.size(), 10000);
    }
}