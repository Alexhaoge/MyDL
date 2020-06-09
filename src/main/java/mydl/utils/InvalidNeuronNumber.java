package mydl.utils;

/**
 * @Deprecated This is a exception never used. Will be removed soon.
 */
public class InvalidNeuronNumber extends IllegalArgumentException{
    /**
     *Incorrect cell number like 0 or negative integer
     */
    private static final long serialVersionUID = 1L;

    /**
     * Create exception
     * @param message string of error message
     */
    public InvalidNeuronNumber(String message) {
        super(message);
    }
}