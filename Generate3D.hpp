/*
* ===========================================================
* File Type: HPP
* File Name: Generate3D.hpp
* Package Name: RMSS
*
* Created by Anthony-A. Christidis.
* Copyright (c) Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

// Header files included
#include "EnsembleModel.hpp"

std::vector<std::vector<std::vector<arma::vec>>> Generate3D_Intercepts(std::vector<std::vector<std::vector<EnsembleModel>>>& ensembles,
    arma::uvec& h, arma::uvec& t, arma::uvec& u,
    arma::uword& n_models) {

    // Matrix for coefficients
    arma::vec intercepts_vector = arma::vec(n_models);

    // Creation of 3D vector for coefficients
    std::vector<std::vector<std::vector<arma::vec>>> intercept3D;

    // Allocation of the coefficients
    for (arma::uword h_ind = 0; h_ind < h.size(); h_ind++) {

        // 2D vector for a fixed trimming value
        std::vector<std::vector<arma::vec>> intercept3D_h;

        for (arma::uword t_ind = 0; t_ind < t.size(); t_ind++) {

            // 1D vector for a fixed sparsity value
            std::vector<arma::vec> intercept3D_t;

            for (arma::uword u_ind = 0; u_ind < u.size(); u_ind++) {

                intercepts_vector = ensembles[h_ind][t_ind][u_ind].Get_Final_Intercepts();
                intercept3D_t.push_back(intercepts_vector);
            }

            // Adding the 1D vector to the 2D vector of ensembles for a fixed trimming value
            intercept3D_h.push_back(intercept3D_t);
        }

        // Adding the 2D vector to the 3D vector of ensembles
        intercept3D.push_back(intercept3D_h);
    }

    return intercept3D;
}

std::vector<std::vector<std::vector<arma::mat>>> Generate3D_Coefficients(std::vector<std::vector<std::vector<EnsembleModel>>>& ensembles,
    arma::uvec& h, arma::uvec& t, arma::uvec& u,
    arma::uword& p, arma::uword& n_models){

    // Matrix for coefficients
    arma::mat coef_matrix = arma::mat(p, n_models);

    // Creation of 3D vector for coefficients
    std::vector<std::vector<std::vector<arma::mat>>> coef3D;

    // Allocation of the coefficients
    for (arma::uword h_ind = 0; h_ind < h.size(); h_ind++) {

        // 2D vector for a fixed trimming value
        std::vector<std::vector<arma::mat>> coef3D_h;

        for (arma::uword t_ind = 0; t_ind < t.size(); t_ind++) {

            // 1D vector for a fixed sparsity value
            std::vector<arma::mat> coef3D_t;

            for (arma::uword u_ind = 0; u_ind < u.size(); u_ind++) {

                coef_matrix = ensembles[h_ind][t_ind][u_ind].Get_Final_Coef();
                coef3D_t.push_back(coef_matrix);
            }

            // Adding the 1D vector to the 2D vector of ensembles for a fixed trimming value
            coef3D_h.push_back(coef3D_t);
        }

        // Adding the 2D vector to the 3D vector of ensembles
        coef3D.push_back(coef3D_h);
    }

    return coef3D;
}