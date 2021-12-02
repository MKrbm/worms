#include <uftree.hpp>
#include "../include/model.hpp"



std::vector<std::vector<int>> heisenberg1D::return_bonds(int L, bool PBC){
    std::vector<std::vector<int>> bonds(L, std::vector<int>(2));

    for(int i=0; i<L; i++){
        bonds[i][0] = i;
        bonds[i][1] = (i+1)%L;
    }
    // std::vector<std::vector<int>> vtr {{34,55},{45},{53,62}};
    return bonds;
}

const int heisenberg1D::N_op = 3;

heisenberg1D::heisenberg1D(int L, double Jz, double Jxy, double h, bool PBC)
    :L(L), PBC(PBC), Jz(Jz), Jxy(Jxy), bonds(return_bonds(L,PBC)),
    Nb(PBC ? L : L-1), h(h), base_model_spin_1D(L, PBC ? L : L-1, 3)
{
    std::cout << "model output" << std::endl;
    std::cout << "L : " << L << std::endl;
    std::cout << "Nb : " << Nb << std::endl;
    std::cout << "Jz : " << Jz << std::endl;
    std::cout << "h : " << h << std::endl;
    std::cout << "end \n" << std::endl;

    //H_1
    std::cout << "define H_1" << std::endl;
    operator_list[0]= {-1, 1, 2, -1};
    weigths[0] = ((1+h)/2);
    worm_dir[0] = {0, 2, 1};
    


    //H_2
    std::cout << "define H_2" << std::endl;
    operator_list[1] = {0, -1, -1, 3};
    weigths[1] = (h);
    worm_dir[1] = {2, 0, 3};



    //H_3
    std::cout << "define H_3" << std::endl;
    operator_list[2] = {-1, 2, 1, -1};
    weigths[2] = (1/2.0);
    worm_dir[2] = {1, 3, 0};

    // if (N_op != weigths.size()){
    //     std::cout << "number of operators is inconsistent" << std::endl;
    // }
    
    double S = weigths[0] + weigths[1]; // 
    prob[0] = weigths[0]/S;
    prob[1] = weigths[1]/S;


    rho = h*Nb + (1+h)/2 * Nb;
}

/*
pick diagonal operator type at random for given r ~ uniform(0,1)
*/
int heisenberg1D::DopAtRand(double r){
    double sum = 0;
    int i;
    for(i=0; i<NDop-1; i++){
        sum += prob[i];
        if (sum >= r) break;
    }

    return i;
}