package mydl.utils;

import java.util.ArrayList;

import mydl.tensor.Tensor;

/**
 * A data class contains the input tensor and target tensor.
 * @deprecated need amendment
 */
public class Data {
    public Tensor input;
    public Tensor target;
    public Data(Tensor _input, Tensor _target){
        input = _input.clone();
        target = _target.clone();
    }
    public static ArrayList<Data> to_data(
    ArrayList<Tensor> inputs, ArrayList<Tensor> targets, int batch_size)
    throws IndexOutOfBoundsException{
        if(inputs.size()!=targets.size())
            throw new IndexOutOfBoundsException("sample size does not match");
        //here
            ArrayList<Data> data = new ArrayList<Data>();
        for(int i = 0; i < inputs.size(); i++)
            data.add(new Data(inputs.get(i), targets.get(i)));
        return data;
    }
}