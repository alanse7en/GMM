//
//  GMM.cpp
//  GMM
//
//  Created by deng on 14-8-18.
//  Copyright (c) 2014年 deng. All rights reserved.
//

#include "GMM.h"

int myRandom(int i) {
    return rand()%i;
}

VectorXd randperm(unsigned long n) {
    vector<unsigned int> randPerm(n);
    auto ite = randPerm.begin();
    for (int i = 0; i < n; ++i) {
        *ite = i;
        ++ite;
    }
    srand( unsigned( time( 0)));
    random_shuffle(randPerm.begin(), randPerm.end(), myRandom);
    VectorXd randVec(n, 1);
    for (int i = 0; i < n; ++i) {
        randVec(i) = randPerm.at(i);
    }
    return randVec;
}

GMM::GMM(long nComponents, long nDimensions, fitOption option) : nComponents(nComponents), nDimensions(nDimensions), option(option)
{
    checkOption(this->option);
    mu = MatrixXd::Zero(nComponents, nDimensions);
    MatrixXd covTmp = MatrixXd::Zero(nDimensions, nDimensions);
    Sigma = vector<MatrixXd>(nComponents, covTmp);
    p = VectorXd::Zero(nComponents);
}

GMM::GMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option)
     : mu(mu), Sigma(Sigma), p(p), option(option)
{
    checkModel();
    checkOption(this->option);
    nComponents = mu.rows();
    nDimensions = mu.cols();
}

GMM& GMM::operator=(const GMM &gmm) {
    mu = gmm.mu;
    Sigma = gmm.Sigma;
    p = gmm.p;
    nComponents = gmm.nComponents;
    nDimensions = gmm.nDimensions;
    option = gmm.option;
    return *this;
}

GMM::GMM(GMM& gmm) {
    mu = gmm.mu;
    Sigma = gmm.Sigma;
    p = gmm.p;
    nComponents = gmm.nComponents;
    nDimensions = gmm.nDimensions;
    option = gmm.option;
}

void GMM::checkData(MatrixXd data) {
    if (data.cols() != nDimensions) {
        throw runtime_error( "Dimension of Data must agree with the model.");
    }
    if (data.rows() <= nDimensions) {
        throw runtime_error( "Data must have more rows than columns.");
    }
    if (data.rows() <= nComponents) {
        throw runtime_error
            ( "Data must have more rows than the number of components.");
    }
}

void GMM::checkOption(fitOption option) {
    if (option.regularize < 0) {
        throw runtime_error( "The regularize must be a non-negative scalar");
    }
    if ( (option.covType != "spherical") 
            && (option.covType != "diagonal") && (option.covType != "full") ) {
        throw runtime_error( "Invalid covType option.");
    }
    if ( (option.display != "iter") && (option.display != "final") ) {
        throw runtime_error( "Invalid display option.");
    }
    if ( (option.start != "random") && (option.start != "kmeans") ) {
        throw runtime_error( "Invalid start option.");
    }
}

void GMM::checkModel() {
    if ( (mu.rows() != p.rows()) 
        || (mu.rows() != Sigma.size()) || (p.rows() != Sigma.size()) )
    {
        throw runtime_error
            ( "The dimension of mean, p and cov must agree with each other");
    }
}

void GMM::initParam(MatrixXd data) {
    if (option.start == "random") {
        // Generate random permutation
        VectorXd randPerm = randperm(nComponents);
        for (int i = 0; i < randPerm.size(); ++i)
        {
            mu.block(i,1,1,mu.cols()) = data.block(randPerm(i),1,1,mu.cols());
            p(i) = 1/nComponents;
            MatrixXd sigma();
        }
    }
}

void GMM::fit(MatrixXd data) {
    checkData(data);
    initParam(data);
}