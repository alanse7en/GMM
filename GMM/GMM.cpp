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

void kMeans(MatrixXd data, long nComponents, VectorXd &idx,
        VectorXd &p, MatrixXd &mu, vector<MatrixXd> Sigma)
{

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
    fitOption option;
    string tmpStr;
    getline(in, option.covType);
    getline(in, tmpStr);
    option.sharedCov = atoi(tmpStr.c_str());;
    getline(in, option.start);
    getline(in ,tmpStr);
    option.regularize = atof(tmpStr.c_str());;
    getline(in, option.display);
    getline(in, tmpStr);
    option.maxIter = atoi(tmpStr.c_str());;
    getline(in, tmpStr);
    option.tolFun = atof(tmpStr.c_str());;
    getline(in, tmpStr);
    option.converged = atoi(tmpStr.c_str());;
    getline(in ,tmpStr);
    option.iters = atoi(tmpStr.c_str());
    BaseGMM.checkOption(option);
    BaseGMM.option = option;

    if (!in)
    {
        throw runtime_error
            ( "Can not read from the file.\n");
    }
    return in;
}

ostream & operator<<(ostream &os, const BaseGMM &BaseGMM)
{
    os << "nComponents: " << BaseGMM.nComponents << "\tnDimensions: " << BaseGMM.nDimensions << endl;
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
    fitOption option = BaseGMM.option;
    out << option.covType << "\n";
    out << option.sharedCov << "\n";
    out << option.start << "\n";
    out << option.regularize << "\n";
    out << option.display << "\n";
    out << option.maxIter << "\n";
    out << option.tolFun << "\n";
    out << option.converged << "\n";
    out << option.iters << "\n";
    if (!out)
    {
        throw runtime_error
            ( "Can not write into the file.\n");
    }
    return out;
}

BaseGMM::BaseGMM(long nComponents, long nDimensions, fitOption option) : 
    nComponents(nComponents), nDimensions(nDimensions), option(option)
{
    checkOption(this->option);
    mu = MatrixXd::Zero(nComponents, nDimensions);
    MatrixXd covTmp = MatrixXd::Zero(nDimensions, nDimensions);
    Sigma = vector<MatrixXd>(nComponents, covTmp);
    p = VectorXd::Zero(nComponents);
}

BaseGMM::BaseGMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option)
     : mu(mu), Sigma(Sigma), p(p), option(option)
{
    checkModel();
    checkOption(this->option);
    nComponents = mu.rows();
    nDimensions = mu.cols();
}

BaseGMM& BaseGMM::operator=(const BaseGMM &BaseGMM) {
    mu = BaseGMM.mu;
    Sigma = BaseGMM.Sigma;
    p = BaseGMM.p;
    nComponents = BaseGMM.nComponents;
    nDimensions = BaseGMM.nDimensions;
    option = BaseGMM.option;
    return *this;
}

BaseGMM::BaseGMM(BaseGMM& BaseGMM) {
    mu = BaseGMM.mu;
    Sigma = BaseGMM.Sigma;
    p = BaseGMM.p;
    nComponents = BaseGMM.nComponents;
    nDimensions = BaseGMM.nDimensions;
    option = BaseGMM.option;
}

void BaseGMM::checkData(MatrixXd data) {
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

void BaseGMM::checkOption(fitOption option) {
    if (option.regularize < 0) {
        throw runtime_error( "The regularize must be a non-negative scalar.\n");
    }
    if ( (option.covType != "spherical") 
            && (option.covType != "diagonal") && (option.covType != "full") ) {
        throw runtime_error( "Invalid covType option.\n");
    }
    if ( (option.display != "iter") && (option.display != "final") ) {
        throw runtime_error( "Invalid display option.\n");
    }
    if ( (option.start != "random") && (option.start != "kmeans") ) {
        throw runtime_error( "Invalid start option.\n");
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

void BaseGMM::checkNaN(MatrixXd mat) {
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

void BaseGMM::initParam(MatrixXd data) {
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
    else
    {
        VectorXd idx = VectorXd::Zero(data.rows());
        kMeans(data, nComponents, idx, p, mu, Sigma);
    }
}

void BaseGMM::likelihood(MatrixXd data, MatrixXd &lh) {
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

double BaseGMM::posterior(MatrixXd data, MatrixXd &post) {
    MatrixXd lh = MatrixXd::Zero(data.rows(), nComponents);
    likelihood(data, lh);
    MatrixXd P = (p.transpose() ).replicate(data.rows(), 1);
    MatrixXd lh_m_P = lh.cwiseProduct(P);
    post = lh_m_P.cwiseQuotient( lh_m_P.rowwise().sum().replicate(1, nComponents));
    double loss = lh_m_P.array().rowwise().sum().log().sum();
    checkNaN(post);
    return loss;
}

void DiffFullGMM::fit(MatrixXd data) {
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
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        double lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            option.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // different sigma
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                MatrixXd centered = data.block(j, 0, 1, data.cols()) 
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
    printf("The final loss is %f, takes %d steps\n", oldLoss, option.iters);
}

void DiffDiagGMM::fit(MatrixXd data) {
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
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        double lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            option.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // different sigma
        // diagonal sigma
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                MatrixXd centered = data.block(j, 0, 1, data.cols()) 
                    - mu.block(i,0,1,mu.cols());
                sigmaTmp = sigmaTmp + post(j, i) * centered.transpose() * centered;
            }
            sigmaTmp = sigmaTmp/p(i) + option.regularize * 
                MatrixXd::Identity(sigmaTmp.rows(), sigmaTmp.cols());
            MatrixXd sigma = (sigmaTmp.diagonal()).asDiagonal();
            mu.block(i,0,1,mu.cols()) = muTmp/p(i);
            Sigma.at(i) = sigma;
        }                
        p = p/double(data.rows());
    }

    // Print the final result: the likelihood loss, mean and sigma.
    printf("The final loss is %f, takes %d steps\n", oldLoss, option.iters);
}

void DiffSpheGMM::fit(MatrixXd data) {
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
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        double lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            option.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // different sigma
        // spherical sigma
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                MatrixXd centered = data.block(j, 0, 1, data.cols()) 
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
    printf("The final loss is %f, takes %d steps\n", oldLoss, option.iters);
}

void ShaFullGMM::fit(MatrixXd data) {
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
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        double lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            option.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // sharing sigma
        MatrixXd sigma = MatrixXd::Zero(data.cols(), data.cols());
        // full sigma                
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                MatrixXd centered = data.block(j, 0, 1, data.cols()) 
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
    printf("The final loss is %f, takes %d steps\n", oldLoss, option.iters);
}

void ShaDiagGMM::fit(MatrixXd data) {
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
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        double lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            option.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        MatrixXd sigma = MatrixXd::Zero(data.cols(), data.cols());
        // diagonal sigma
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                MatrixXd centered = data.block(j, 0, 1, data.cols()) 
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
    printf("The final loss is %f, takes %d steps\n", oldLoss, option.iters);
}

void ShaSpheGMM::fit(MatrixXd data) {
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
            printf("Iteration: %d  Loss: %f\n", ite+1,loss);
        }
        // Check the converge condition.
        double lossDiff = oldLoss -loss;
        if (lossDiff >= 0 && lossDiff <= option.tolFun * abs(loss))
        {
            option.converged = true;
            break;
        }
        oldLoss = loss;
        // M step
        p = post.colwise().sum();
        // sharing sigma
        MatrixXd sigma = MatrixXd::Zero(data.cols(), data.cols());
        for (int i = 0; i < nComponents; ++i)
        {
            MatrixXd muTmp = MatrixXd::Zero(1, data.cols());
            MatrixXd sigmaTmp = MatrixXd::Zero(data.cols(), data.cols());
            for (int j = 0; j < data.rows(); ++j)
            {
                muTmp = muTmp + post(j,i)*data.block(j, 0, 1, data.cols());
                MatrixXd centered = data.block(j, 0, 1, data.cols()) 
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
    printf("The final loss is %f, takes %d steps\n", oldLoss, option.iters);
}

double BaseGMM::cluster(MatrixXd data, MatrixXd &post) {
    checkData(data);
    double nLogL = -posterior(data, post);
    return nLogL;
}

double BaseGMM::cluster(MatrixXd data, MatrixXd &post, VectorXd &idx) {
    double nLogL = this->cluster(data, post);
    for (int i = 0; i < data.rows(); ++i)
    {
        post.row(i).maxCoeff( &idx(i));
    }
    return nLogL;
}