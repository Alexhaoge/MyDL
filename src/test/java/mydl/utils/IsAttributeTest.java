package mydl.utils;



import static org.junit.Assert.assertEquals;

import org.junit.Test;

import mydl.tensor.Tensor;
import mydl.tensor.Tensor1D;

public class IsAttributeTest {
    
    @Test
    public void attributeTest(){
        Tensor1D x = new Tensor1D(2);
        Tensor z = new Tensor1D(4);
        Tensor c = (Tensor) x;
        assertEquals(IsAttribute.isAttribute(x, "darray"), true);
        assertEquals(IsAttribute.isAttribute(z, "darray"), true);
        assertEquals(IsAttribute.isAttribute(c, "darray"), true);
    }
}