package mydl.utils;

import java.lang.reflect.Field;

/**
 * Determin whether a attribute in the given object.
 */
public class IsAttribute {
    /**
     * Determin whether a attribute in the given object.
     * @param A the given object.
     * @param name A string which is the name of the attribute.
     * @return boolean, true if the object have such attribute.
     */
    public static boolean isAttribute(Object A, String name){
        Field[] fields = A.getClass().getDeclaredFields();
        for(int i=0;i<fields.length;i++)
            if(fields[i].getName().equals(name))
                return true;
        return false;
    }
}