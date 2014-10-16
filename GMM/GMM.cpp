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

void kMeans(const MatrixXd &data, long nComponents, VectorXd &idx, MatrixXd &mu)
{
    // Randomly initialize mu
    VectorXd randPerm = randperm(data.rows()-1);
    for (int i = 0; i < mu.rows(); ++i)
    {
        mu.block(i,0,1,mu.cols()) = data.block(randPerm(i),0,1,mu.cols());
    }
    VectorXd idxTmp = idx;
    MatrixXd distances = MatrixXd::Zero(data.rows(), nComponents);
    while(1) {
        for (int i = 0; i < data.rows(); ++i)
        {
            for (int j = 0; j < nComponents; ++j)
            {
                distances(i,j) = (data.block(i, 0, 1, data.cols()) -
                                  mu.block(j, 0, 1, mu.cols())).norm();
            }
        }
        // Reassign indexes
        for (int i = 0; i < data.rows(); ++i) {
            distances.row(i).minCoeff(&idxTmp(i));
        }
        // Update mu
        for (int i = 0; i < mu.rows(); ++i) {
            int total = 0;
            for (int k = 0; k < data.rows(); ++k) {
                if (idxTmp(k) == i) {
                    total++;
                    mu.block(i, 0, 1, mu.cols()) += data.block(k, 0, 1, data.cols());
                }
            }
            mu.block(i, 0, 1, mu.cols()) = mu.block(i, 0, 1, mu.cols())/double(total);
        }
        // convergence
        if (idxTmp == idx) {
            break;
        }
        idx = idxTmp;
    }
}

ifstream & operator>>(ifstream &in, BaseGMM &BaseGMM)
{
    // Read nComponents
    string nComponentsStr;
    getline(in, nComponentsStr);
    BaseGMM.nComponents = atol(nComponentsStr.c_str());
    // Read nDimensions
    string nDimensionsStr;
    getline(in, nDimensionsStr);
    BaseGMM.nDimensions = atol(nDimensionsStr.c_str());
    // Make sure the file is a valid one.
    BaseGMM.checkModel();
    //Initialize p, mu, Sigma
    BaseGMM.p = VectorXd::Zero(BaseGMM.nComponents);
    BaseGMM.mu = MatrixXd::Zero(BaseGMM.nComponents, BaseGMM.nDimensions);
    MatrixXd sigma = MatrixXd::Zero(BaseGMM.nDimensions, BaseGMM.nDimensions);
    BaseGMM.Sigma = vector<MatrixXd>(BaseGMM.nComponents, sigma);
    // Read p
    for (long i = 0; i < BaseGMM.nComponents; ++i)
    {
        string pStr;
        getline(in, pStr);
        BaseGMM.p(i) = atof(pStr.c_str());
    }
    // Read mu
    for (int i = 0; i < BaseGMM.nComponents; ++i)
    {
        string muStr;
        getline(in, muStr);
        istringstream muStream(muStr);
        for (int j = 0; j < BaseGMM.nDimensions; ++j)
        {
            string mustr;
            muStream >> mustr;
            BaseGMM.mu(i, j) = atof(mustr.c_str());
        }
    }
    // Read Sigma
    for (auto ite = BaseGMM.Sigma.begin(); ite != BaseGMM.Sigma.cend(); ++ite)
    {
        for (long i = 0; i < BaseGMM.nDimensions; ++i)
        {
            string SigmaStr;
            getline(in, SigmaStr);
            istringstream SigmaStream(SigmaStr);
            for (long j = 0; j < BaseGMM.nDimensions; ++j)
            {
                string sigmastr;
                SigmaStream >> sigmastr;
                (*ite)(i, j) = atof(sigmastr.c_str());
            }
        }
    }
    // Read Options
    string tmpStr;
    getline(in, BaseGMM.option.start);
    getline(in ,tmpStr);
    BaseGMM.option.regularize = atof(tmpStr.c_str());;
    getline(in, BaseGMM.option.display);
    getline(in, tmpStr);
    BaseGMM.option.maxIter = atoi(tmpStr.c_str());;
    getline(in, tmpStr);
    BaseGMM.option.tolFun = atof(tmpStr.c_str());;
    getline(in, tmpStr);
    BaseGMM.result.converged = atoi(tmpStr.c_str());;
    getline(in ,tmpStr);
    BaseGMM.result.iters = atoi(tmpStr.c_str());
    BaseGMM.checkOption();

    if (!in)
    {
        throw runtime_error
            ( "Can not read from the file.\n");
    }
    return in;
}

ostream & operator<<(ostream &os, const BaseGMM &BaseGMM)
{
    os << "nComponents: " << BaseGMM.nComponents << "\tnDimensions: " 
        << BaseGMM.nDimensions << endl;
    cout << "The p of the model:\n" << BaseGMM.p.transpose() << endl;
    cout << "The mean of the model:\n " << BaseGMM.mu << endl;
    cout << "The sigma of the model:";
    for (auto ite = BaseGMM.Sigma.begin(); ite != BaseGMM.Sigma.cend(); ++ite) {
        cout << endl;
        cout << "Sigma " <<(ite - BaseGMM.Sigma.cbegin()) + 1 << ":" << endl;
        cout << *ite;
    }
    return os;
}

ofstream & operator<<(ofstream &out, const BaseGMM & BaseGMM)
{
    // write nComponents and nDimensions
    
    out << BaseGMM.nComponents << "\n";
    out << BaseGMM.nDimensions << "\n";
    // write p
    out << BaseGMM.p << "\n";
    // write mu
    for (int i = 0; i < BaseGMM.mu.rows(); ++i)
    {
        for (int j = 0; j < BaseGMM.mu.cols(); ++j)
        {
            out << BaseGMM.mu(i,j) << " ";
        }
        out << "\n";
    }
    // write Sigma
    for (auto ite = BaseGMM.Sigma.begin(); ite != BaseGMM.Sigma.cend(); ++ite)
    {
        MatrixXd sigma = *ite;
        for (int i = 0; i < sigma.rows(); ++i)
        {
            for (int j = 0; j < sigma.cols(); ++j)
            {
                out << sigma(i,j) << " ";
            }
            out << "\n";
        }
    }
    out << BaseGMM.option.start << "\n";
    out << BaseGMM.option.regularize << "\n";
    out << BaseGMM.option.display << "\n";
    out << BaseGMM.option.maxIter << "\n";
    out << BaseGMM.option.tolFun << "\n";
    out << BaseGMM.result.converged << "\n";
    out << BaseGMM.result.iters << "\n";
    if (!out)
    {
        throw runtime_error
            ( "Can not write into the file.\n");
    }
    return out;
}

BaseGMM::BaseGMM(long nComponents, long nDimensions, const fitOption &option,
                 const fitResult &result) :
    nComponents(nComponents), nDimensions(nDimensions), option(option), result(result),
    mu(MatrixXd::Zero(nComponents, nDimensions)),
    Sigma(vector<MatrixXd>(nComponents, MatrixXd::Zero(nDimensions, nDimensions))),
    p(VectorXd::Zero(nComponents))
{
    checkOption();

}

BaseGMM::BaseGMM(const MatrixXd &mu, const vector<MatrixXd> &Sigma, const VectorXd &p,
                 const fitOption &option, const fitResult &result) :
    mu(mu), Sigma(Sigma), p(p), option(option), result(result),
    nComponents(mu.rows()), nDimensions(mu.cols())
{
    checkModel();
    checkOption();
}

void BaseGMM::checkData(const MatrixXd &data) {
    checkNaN(data);
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

void BaseGMM::checkOption() {
    if (option.regularize < 0) {
        throw runtime_error( "The regularize must be a non-negative scalar.\n");
    }
    if ( (option.display != "iter") && (option.display != "final") ) {
        throw runtime_error( "Invalid display option.\n");
    }
    if ( (option.start != "random") && (option.start != "kmeans") ) {
        throw runtime_error( "Invalid start option.\n");
    }
}

void BaseGMM::showResult() {
    if (result.converged == true)
    {
        printf("Fit process converged in %i steps.\n", result.iters);
    }
    else
    {
        printf("Fit process failed to converge in %i steps.\n", result.iters);
    }
}

void BaseGMM::checkModel() {
    if ( (mu.rows() != p.rows()) 
        || (mu.rows() != Sigma.size()) || (p.rows() != Sigma.size()) )
    {
        throw runtime_error
            ( "The dimension of mean, p and cov must agree with each other\n");
    }
    if (nComponents == 0 || nDimensions == 0)
    {
        throw runtime_error
            ( "nComponents or nDimensions should not be 0. Bad initialization.\n");
    }
}

void BaseGMM::checkNaN(const MatrixXd &mat) {
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            stringstream tmp;
            tmp << mat(i, j);
            if (tmp.str() == "nan")
            {
                throw runtime_error
                    ("NaN found in a matrix.");
            }
        }
    }
}

void BaseGMM::initParam(const MatrixXd &data) {
    // Initial with random setting
    if (option.start == "random") {
        // Generate random permutation
        VectorXd randPerm = randperm(data.rows()-1);
        MatrixXd centered = data.rowwise() - data.colwise().mean();
        MatrixXd sigma    = 
            (centered.transpose() * centered) / double(data.rows() -1);
        for (int i = 0; i < nComponents; ++i)
        {
            mu.block(i,0,1,mu.cols()) = data.block(randPerm(i),0,1,mu.cols());
            p(i) = 1/double(nComponents);
            Sigma.at(i) = sigma;
        }
    }
    // Initial with kmeans
    else
    {
        VectorXd idx = VectorXd::Zero(data.rows());
        // get idx and mu with kmeans
        kMeans(data, nComponents, idx, mu);
        for (int i = 0; i < nComponents; ++i) {
            // total is the num of datapoints belong to i-th component
            int total = 0;
            for (int k = 0; k < data.rows(); ++k) {
                if (idx(k) == i) {
                    total++;
                }
            }
            p(i) = double(total)/double(data.rows());
            // Extract data belong to i-th component
            MatrixXd dataTmp = MatrixXd::Zero(total, data.cols());
            int k = 0;
            for (int j = 0; j < data.rows(); ++j) {
                if (idx(j) == i) {
                    dataTmp.block(k, 0, 1, dataTmp.cols()) =
                        data.block(j, 0, 1, data.cols());
                    k++;
                }
            }
            MatrixXd centered = dataTmp.rowwise() - data.colwise().mean();
            MatrixXd sigma =
                (centered.transpose() * centered) / double(total -1);
            Sigma.at(i) = sigma;
        }
    }
}

void BaseGMM::likelihood(const MatrixXd &data, MatrixXd &lh) {
    for (int i = 0; i < lh.rows(); ++i)
    {
        for (int j = 0; j < lh.cols(); ++j)
        {
            auto det  = Sigma.at(j).determinant();
            auto para = sqrt(pow(2*M_PI, nDimensions) * det);
            MatrixXd centered = 
                data.block(i, 0, 1, data.cols()) - mu.block(j, 0, 1, mu.cols());
            MatrixXd tmpv = 
                centered * (Sigma.at(j).inverse()) * (centered.transpose());
            lh(i, j) = exp( -0.5*tmpv(0,0) )/para;
        }
    }
    checkNaN(lh);
}

double BaseGMM::posterior(const MatrixXd &data, MatrixXd &post) {
    MatrixXd lh = MatrixXd::Zero(data.rows(), nComponents);
    likelihood(data, lh);
    MatrixXd P = (p.transpose() ).replicate(data.rows(), 1);
    MatrixXd lh_m_P = lh.cwiseProduct(P);
    post = lh_m_P.cwiseQuotient( lh_m_P.rowwise().sum().replicate(1, nComponents));
    double loss = lh_m_P.array().rowwise().sum().log().sum();
    checkNaN(post);
    return loss;
}

void DiffFullGMM::fit(const MatrixXd &data) {
    checkData(data);
    initParam(data);
    double oldLoss = 0.0;
    MatrixXd post = MatrixXd::Zero(data.rows(), nComponents);
    double lossDiff = 0.0;
    for (auto ite = 0; ite < option.maxIter; ++ite) {
        // E step
        result.iters = ite +1;
        auto loss = posterior(data, post);
        if (option.display == "iter")
        {
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            result.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // different sigma
        MatrixXd centered = MatrixXd::Zero(1, data.cols());
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                centered = data.block(j, 0, 1, data.cols())
                    - mu.block(i,0,1,mu.cols());
                sigmaTmp = sigmaTmp + post(j, i) * centered.transpose() * centered;
            }
            // double normPara = post.colwise().sum()(i);
            mu.block(i,0,1,mu.cols()) = muTmp/p(i);
            Sigma.at(i) = sigmaTmp/p(i) + option.regularize * 
            MatrixXd::Identity(sigmaTmp.rows(), sigmaTmp.cols());
        }
        p = p/double(data.rows());
}

    // Print the final result: the likelihood loss, mean and sigma.
    if (option.display != "off") {
        printf("The final loss is %f, takes %d steps\n", oldLoss, result.iters);
    }
}

void DiffDiagGMM::fit(const MatrixXd &data) {
    checkData(data);
    initParam(data);
    double oldLoss = 0.0;
    double lossDiff = 0.0;
    double loss = 0.0;
    MatrixXd post = MatrixXd::Zero(data.rows(), nComponents);
    for (auto ite = 0; ite < option.maxIter; ++ite) {
        // E step
        result.iters = ite +1;
        loss = posterior(data, post);
        if (option.display == "iter")
        {
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            result.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // different sigma
        // diagonal sigma
        MatrixXd centered = MatrixXd::Zero(1, data.cols());
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                centered = data.block(j, 0, 1, data.cols())
                    - mu.block(i,0,1,mu.cols());
                sigmaTmp = sigmaTmp + post(j, i) * centered.transpose() * centered;
            }
            mu.block(i,0,1,mu.cols()) = muTmp/p(i);
            Sigma.at(i) = ((sigmaTmp/p(i) + option.regularize *
                           MatrixXd::Identity(sigmaTmp.rows(), sigmaTmp.cols())).
                           diagonal()).asDiagonal();
        }                
        p = p/double(data.rows());
    }

    // Print the final result: the likelihood loss, mean and sigma.
    if (option.display != "off") {
        printf("The final loss is %f, takes %d steps\n", oldLoss, result.iters);
    }
}

void DiffSpheGMM::fit(const MatrixXd &data) {
    checkData(data);
    initParam(data);
    double oldLoss = 0.0;
    double loss = 0.0;
    double lossDiff = 0.0;
    MatrixXd post = MatrixXd::Zero(data.rows(), nComponents);
    for (auto ite = 0; ite < option.maxIter; ++ite) {
        // E step
        result.iters = ite +1;
        loss = posterior(data, post);
        if (option.display == "iter")
        {
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            result.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // different sigma
        // spherical sigma
        MatrixXd centered = MatrixXd::Zero(1, data.cols());
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                centered = data.block(j, 0, 1, data.cols())
                    - mu.block(i,0,1,mu.cols());
                sigmaTmp = sigmaTmp + post(j, i) * centered.transpose() * centered;
            }
            sigmaTmp = sigmaTmp/p(i) + option.regularize * 
                MatrixXd::Identity(sigmaTmp.rows(), sigmaTmp.cols());
            double diagonal = (sigmaTmp.diagonal()).sum()/sigmaTmp.rows();
            VectorXd sigmaDia = VectorXd::Ones(sigmaTmp.rows())*diagonal;
            MatrixXd sigma = sigmaDia.asDiagonal();
            mu.block(i,0,1,mu.cols()) = muTmp/p(i);
            Sigma.at(i) = sigma;
        }                
        p = p/double(data.rows());
    }

    // Print the final result: the likelihood loss, mean and sigma.
    if (option.display != "off") {
        printf("The final loss is %f, takes %d steps\n", oldLoss, result.iters);
    }
}

void ShaFullGMM::fit(const MatrixXd &data) {
    checkData(data);
    initParam(data);
    double oldLoss =  0.0;
    double loss    =  0.0;
    double lossDiff = 0.0;
    MatrixXd post = MatrixXd::Zero(data.rows(), nComponents);
    for (auto ite = 0; ite < option.maxIter; ++ite) {
        // E step
        result.iters = ite +1;
        loss = posterior(data, post);
        if (option.display == "iter")
        {
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            result.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // sharing sigma
        MatrixXd sigma = MatrixXd::Zero(data.cols(), data.cols());
        // full sigma
        MatrixXd centered = MatrixXd::Zero(1, data.cols());
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                centered = data.block(j, 0, 1, data.cols()) 
                    - mu.block(i,0,1,mu.cols());
                sigmaTmp = sigmaTmp + post(j, i) * centered.transpose() * centered;
            }
            mu.block(i,0,1,mu.cols()) = muTmp/p(i);
            sigma = sigma + sigmaTmp/p(i);
        }
        sigma = sigma/double(nComponents) + option.regularize *
                MatrixXd::Identity(sigma.rows(), sigma.cols());
            
        Sigma = vector<MatrixXd>(nComponents, sigma);        
        p = p/double(data.rows());
    }

    // Print the final result: the likelihood loss, mean and sigma.
    if (option.display != "off") {
        printf("The final loss is %f, takes %d steps\n", oldLoss, result.iters);
    }
}

void ShaDiagGMM::fit(const MatrixXd &data) {
    checkData(data);
    initParam(data);
    double oldLoss =  0.0;
    double loss    =  0.0;
    double lossDiff = 0.0;
    MatrixXd post = MatrixXd::Zero(data.rows(), nComponents);
    for (auto ite = 0; ite < option.maxIter; ++ite) {
        // E step
        result.iters = ite +1;
        loss = posterior(data, post);
        if (option.display == "iter")
        {
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            result.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        MatrixXd sigma = MatrixXd::Zero(data.cols(), data.cols());
        // diagonal sigma
        MatrixXd centered = MatrixXd::Zero(1, data.cols());
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                centered = data.block(j, 0, 1, data.cols()) 
                    - mu.block(i,0,1,mu.cols());
                sigmaTmp = sigmaTmp + post(j, i) * centered.transpose() * centered;
            }
            mu.block(i,0,1,mu.cols()) = muTmp/p(i);
            sigma = sigma + sigmaTmp/p(i);
        }
        sigma = sigma/double(nComponents) + option.regularize *
                MatrixXd::Identity(sigma.rows(), sigma.cols());
        VectorXd sigDia = sigma.diagonal();
        sigma = sigDia.asDiagonal();
        Sigma = vector<MatrixXd>(nComponents, sigma);
        p = p/double(data.rows());
    }

    // Print the final result: the likelihood loss, mean and sigma.
    if (option.display != "off") {
        printf("The final loss is %f, takes %d steps\n", oldLoss, result.iters);
    }
}

void ShaSpheGMM::fit(const MatrixXd &data) {
    checkData(data);
    initParam(data);
    double oldLoss =  0.0;
    double loss    =  0.0;
    double lossDiff = 0.0;
    MatrixXd post = MatrixXd::Zero(data.rows(), nComponents);
    for (auto ite = 0; ite < option.maxIter; ++ite) {
        // E step
        result.iters = ite +1;
        loss = posterior(data, post);
        if (option.display == "iter")
        {
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            result.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // sharing sigma
        MatrixXd sigma = MatrixXd::Zero(data.cols(), data.cols());
        MatrixXd centered = MatrixXd::Zero(1, data.cols());
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                centered = data.block(j, 0, 1, data.cols()) 
                    - mu.block(i,0,1,mu.cols());
                sigmaTmp = sigmaTmp + post(j, i) * centered.transpose() * centered;
            }
            mu.block(i,0,1,mu.cols()) = muTmp/p(i);
            sigma = sigma + sigmaTmp/p(i);
        }
        sigma = sigma/double(nComponents) + option.regularize *
                MatrixXd::Identity(sigma.rows(), sigma.cols());
        double diagonal = sigma.diagonal().sum()/sigma.rows();
        sigma = (VectorXd::Ones(sigma.rows())*diagonal).asDiagonal();            
        Sigma = vector<MatrixXd>(nComponents, sigma);
        p = p/double(data.rows());
    }

    // Print the final result: the likelihood loss, mean and sigma.
    if (option.display != "off") {
        printf("The final loss is %f, takes %d steps\n", oldLoss, result.iters);
    }
}

double BaseGMM::cluster(const MatrixXd &data, MatrixXd &post) {
    checkData(data);
    double nLogL = -posterior(data, post);
    return nLogL;
}

double BaseGMM::cluster(const MatrixXd &data, MatrixXd &post, VectorXd &idx) {
    double nLogL = this->cluster(data, post);
    for (int i = 0; i < data.rows(); ++i)
    {
        post.row(i).maxCoeff( &idx(i));
    }
    return nLogL;
}