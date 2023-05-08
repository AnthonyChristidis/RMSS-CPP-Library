/*
* ===========================================================
* File Type: CPP
* File Name: RInterface.cpp
* Package Name: RMSS
*
* Created by Anthony-A. Christidis.
* Copyright (c) Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

// Header files included
#include "EnsembleModel.hpp"
#include "InitializeEnsembleModel.hpp"
#include "Generate3D.hpp"

Rcpp::List RInterface(arma::mat& x, arma::vec& y,
    arma::mat& med_x, arma::mat& mad_x,
    arma::mat& med_x_ensemble, arma::mat& mad_x_ensemble,
    double& med_y, double& mad_y,
    arma::uword& n_models,
    arma::uvec& h, arma::uvec& t, arma::uvec& u,
    double& tolerance,
    arma::uword& max_iter,
    arma::umat& initial_split) {

    // Creation of 3D vector to store ensembles
    std::vector<std::vector<std::vector<EnsembleModel>>> ensembles;

    // Initialization of ensembles
    InitializeEnsembleModel(ensembles,
        x, y,
        med_x, mad_x,
        med_x_ensemble, mad_x_ensemble,
        med_y, mad_y,
        n_models,
        h, t, u,
        tolerance,
        max_iter,
        initial_split);

    // Creating list for output
    Rcpp::List output;
    arma::uword p = x.n_cols;
    output["intercepts"] = Generate3D_Intercepts(ensembles, h, t, u, n_models);
    output["coef"] = Generate3D_Coefficients(ensembles, h, t, u, p, n_models);

    // Return final output
    return output;
}