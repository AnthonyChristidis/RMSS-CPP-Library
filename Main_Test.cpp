// Libraries included 
#include <iostream>
#include <armadillo>
#include <vector>

int main()
{
    //std::vector<std::vector<std::vector<arma::mat>>> ensembles;
    //arma::uvec h, t, u = { 1, 2, 3 };

    //// Dynamic allocation of the vectors
    //for (arma::uword h_ind = 0; h_ind < h.size(); h_ind++) {

    //    // 2D vector for a fixed trimming value
    //    std::vector<std::vector<arma::mat>> ensembles_h;

    //    for (arma::uword t_ind = 0; t_ind < t.size(); t_ind++) {

    //        // 1D vector for a fixed sparsity value
    //        std::vector<arma::mat> ensembles_t;

    //        for (arma::uword u_ind = 0; u_ind < u.size(); u_ind++)
    //            ensembles_t.push_back(arma::randn(3, 3));

    //        // Adding the 1D vector to the 2D vector of ensembles for a fixed trimming value
    //        ensembles_h.push_back(ensembles_t);
    //    }

    //    // Adding the 2D vector to the 3D vector of ensembles
    //    ensembles.push_back(ensembles_h);
    //}

    arma::mat some_mat = arma::randn(5, 5);
    arma::uvec rows = arma::uvec(3);
    rows(0) = 0; rows(1) = 2; rows(2) = 4;
    arma::uvec columns = arma::uvec(2);
    columns(0) = 2; columns(1) = 3;

    std::cout << some_mat << std::endl;

    some_mat.submat(rows, columns) = arma::ones(3, 2);

    std::cout << some_mat;

    return 0;
}



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
