from scipy import sparse
from .base_conv import num2state, state2num
def beauty_array(H_tmp, path = "array.txt"):
    try:
        H_tmp = H_tmp.toarray()
    except:
        pass
    with open(path, 'w') as f:
            f.write("{:>6} ".format(""))
            for j in range(H_tmp.shape[1]):
                    f.write("{:>6}    ".format(str(num2state(j, 2))))

            f.write("\n")
            for i in range(H_tmp.shape[0]):
                    f.write("{:>6}".format(str(num2state(i, 2))))
                    for j in range(H_tmp.shape[1]):        
                            f.write("{:>6.3f}, ".format(H_tmp[i,j]))
                    f.write("\n")