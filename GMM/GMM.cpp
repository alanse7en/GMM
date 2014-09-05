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
        VectorXd randPerm = randperm(data.rows()-1);
        MatrixXd centered = data.rowwise() - data.colwise().mean();
        MatrixXd sigma    = (centered.transpose() * centered) / double(data.rows() -1);
        for (int i = 0; i < nComponents; ++i)
        {
            mu.block(i,0,1,mu.cols()) = data.block(randPerm(i),0,1,mu.cols());
            p(i) = 1/double(nComponents);
            Sigma.at(i) = sigma;
        }
    }
}

void GMM::likelihood(MatrixXd data, MatrixXd &lh) {
    for (int i = 0; i < lh.rows(); ++i)
    {
        for (int j = 0; j < lh.cols(); ++j)
        {
            auto det  = Sigma.at(j).determinant();
            auto para = sqrt(pow(2*M_PI, nDimensions) * det);
            MatrixXd centered = data.block(i, 0, 1, data.cols()) - mu.block(j, 0, 1, mu.cols());
            MatrixXd tmpv = centered * (Sigma.at(j).inverse()) * (centered.transpose());
            lh(i, j) = exp( -0.5*tmpv(0,0) )/para;
        }
    }
}

double GMM::posterior(MatrixXd data, MatrixXd &post) {
    MatrixXd lh = MatrixXd::Zero(data.rows(), nComponents);
    likelihood(data, lh);
    MatrixXd P = (p.transpose() ).replicate(data.rows(), 1);
    MatrixXd lh_m_P = lh.cwiseProduct(P);
    post = lh_m_P.cwiseQuotient( lh_m_P.rowwise().sum().replicate(1, nComponents));
    double loss = lh_m_P.array().rowwise().sum().log().sum();
    return loss;
}

void GMM::fit(MatrixXd data) {
    checkData(data);
    initParam(data);
    double oldLoss = 0.0;
    for (auto ite = 0; ite < option.maxIter; ++ite) {
        // E step
        option.iters = ite +1;
        MatrixXd post = MatrixXd::Zero(data.rows(), nComponents);
        auto loss = posterior(data, post);
        if (option.display == "iter")
        {
            printf("The loss is %f\n", loss);
        }
        if (abs(oldLoss - loss) <= option.tolFun)
        {
            option.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        for (int i = 0; i < nComponents; ++i)
        {
            p(i) = post.colwise().sum()(i)/double(data.rows());
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                MatrixXd centered = data.block(j, 0, 1, data.cols()) 
                    - mu.block(i,0,1,mu.cols());
                sigmaTmp = sigmaTmp + post(j, i) * centered.transpose() * centered;
            }
            double normPara = post.colwise().sum()(i);
            mu.block(i,0,1,mu.cols()) = muTmp/normPara;
            Sigma.at(i) = sigmaTmp/normPara;
        }
    }
    printf("The final loss is %f, takes %d steps\n", oldLoss, option.iters);
}