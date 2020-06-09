package test.java;

public class accesstest {
    public static void main(String[] args) {
        A a = new A();
        a.setx(2);
        a.b.output();
    }
}

class A{
    B b = new B();
    public void setx(int t){
        b.x = t;
    }
}

class B{
    protected int x=1;
    public void output(){
        System.out.println(x);
    }
}