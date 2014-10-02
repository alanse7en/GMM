
#ifndef __BaseGMM__BaseGMM__
#define __BaseGMM__BaseGMM__

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <fstream>

using namespace std;
using namespace Eigen;
/*!Option structure for the fit function of Gaussian moxiture model.*/ 
struct fitOption {
    string covType = "full";
    /*!<
     *'diagonal': covariance matrices are restricted to be diagonal;

     *'spherical': covariance matrices are restricted to be spherical;

     *'full': covariance matrices are restricted to be full( default).
    */
    bool sharedCov = false;
    /*!<
     *True if the component covariance matrices are restricted to be the same, default is false.
    */
    string start = "kmeans";
    /*!<
     *'random': randomly select from train data to initialize model parameters;

     *'kmeans' use kmeans to get initial clustering index of the train data( default).
    */
    double regularize = 0.0;
    /*!<
     *Regularization parameter to avoid covariance being singular, defalut is 0.
     */
    string display = "iter";
    /*!<
     *Show information during training (default); 'final' show final information.
     */
    unsigned int maxIter = 1000;
    /*!<
     *Maximum iterations allowed, default is 1000;
     */
    double tolFun = 1e-6;
    /*!<
     *The termination tolerance for the log-likelihood function, default is 1e-6.
     */
    bool converged = false;
    /*!<
     *True if fit function converged, false for not converged.
     */
    unsigned int iters = 0;
    /*!<
     *The num of iterations when the fit function converged.
     */
};

/*! 
    * \brief Base Gaussian mixture model class.
*/
class BaseGMM {
    friend ostream & operator<<(ostream &os, const BaseGMM &BaseGMM);
    friend ifstream & operator>>(ifstream &in, BaseGMM &BaseGMM);
    friend ofstream & operator<<(ofstream &out, const BaseGMM &BaseGMM);
protected: 
    long nComponents;/*!< The number of mixture components, K; */
    long nDimensions;/*!< The number of dimensions for each Gaussian component, D; */
    MatrixXd mu;/*!< A K-by-D matrix of component means; */
    vector<MatrixXd> Sigma;/*!< A K-by-D-by-D matrix of component covariance matrices; */
    VectorXd p;/*!< A K-by-1 vector of the proportion of each component. */
    fitOption option;/*!< Containing options about the BaseGMM model. */
    /*! 
        * Check the train data to ensure it is valid.
        * @param data The train data matrix to be checked.
    */
    void checkData(MatrixXd data);
    /*!
        * Check the option to ensure it is valid.
        * @param option The option for fit function to be checked.
    */
    void checkOption(fitOption option);
    /*!
        * Check model to ensure it is valid.
    */
    void checkModel();
    /*!
        * Check if there is Not-a-Number in the matrix
    */
    void checkNaN(MatrixXd mat);
    /*!
        * Initialize model parameters.
        * @param data The train data matrix.
    */
    void initParam(MatrixXd data);
    /*!
        * @param data A N-by-D matrix containing the data;
        * @param lh A N-by-K matrix containing the likelihood,
        * lh(i, j) = Pr(datapoint I | component J)
    */
    void likelihood(MatrixXd data, MatrixXd &lh);
    /*! calculate posterior probability of given data and BaseGMM
        * @param data A N-by-D matrix containing the data;
        * @return The Log-likelihood loss
    */
    double posterior(MatrixXd data, MatrixXd &post);
    
public:
    /// Default constructor
    BaseGMM() = default;
    ///  Constructors with nComponents, nDimensions and option
    BaseGMM(long nComponents, long nDimensions, fitOption option = fitOption());
    ///  Constructors with mu, Sigma, p, and option
    BaseGMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption());
    ///  Copy constructors
    BaseGMM(BaseGMM &BaseGMM);
    ///  = operator
    BaseGMM& operator=(const BaseGMM& BaseGMM);
    /// Destructor
    virtual ~BaseGMM() = default;
    /*!
        * \brief Train BaseGMM to fit data;
        * @param data A N-by-D matrix containing the train data.
    */
    virtual void fit(MatrixXd data) = 0;
    /*!
        * \brief Cluster the with trained model.
        * @param data  A N-by-D matrix containing the test data;
        * @param post  A N-by-K matrix containing the posterior peobability of p(component J | dataPoint I);
        * @return The negative Log likelihood
    */
    double cluster(MatrixXd data, MatrixXd &post);
    /*!
        * \brief Cluster the with trained model.
        * @param data  A N-by-D matrix containing the test data;
        * @param idx  A N-by-1 vector representing N data points' correspoding Gaussian components;
        * @param post  A N-by-K matrix containing the posterior peobability of p(component J | dataPoint I);
        * @return The negative Log likelihood
    */
    double cluster(MatrixXd data, MatrixXd &post, VectorXd &idx);
};

class DiffFullGMM : public BaseGMM {
public:
    DiffFullGMM() = default;
    DiffFullGMM(long nComponents, long nDimensions, fitOption option = fitOption()) :
        BaseGMM(nComponents, nDimensions, option) {}
    DiffFullGMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption()) :
        BaseGMM(mu, Sigma, p, option) {}
    DiffFullGMM(DiffFullGMM &dfGMM) : BaseGMM(dfGMM) {}
    DiffFullGMM& operator=(const DiffFullGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    void fit(MatrixXd data) override;
};

class DiffDiagGMM : public BaseGMM {
public:
    DiffDiagGMM() = default;
    DiffDiagGMM(long nComponents, long nDimensions, fitOption option = fitOption()) :
        BaseGMM(nComponents, nDimensions, option) {}
    DiffDiagGMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption()) :
        BaseGMM(mu, Sigma, p, option) {}
    DiffDiagGMM(DiffDiagGMM &ddGMM) : BaseGMM(ddGMM) {}
    DiffDiagGMM& operator=(const DiffDiagGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    void fit(MatrixXd data) override;
};

class DiffSpheGMM : public BaseGMM {
public:
    DiffSpheGMM() = default;
    DiffSpheGMM(long nComponents, long nDimensions, fitOption option = fitOption()) :
        BaseGMM(nComponents, nDimensions, option) {}
    DiffSpheGMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption()) :
        BaseGMM(mu, Sigma, p, option) {}
    DiffSpheGMM(DiffSpheGMM &dsGMM) : BaseGMM(dsGMM) {}
    DiffSpheGMM& operator=(const DiffSpheGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    void fit(MatrixXd data) override;
};

class ShaFullGMM : public BaseGMM {
public:
    ShaFullGMM() = default;
    ShaFullGMM(long nComponents, long nDimensions, fitOption option = fitOption()) :
        BaseGMM(nComponents, nDimensions, option) {}
    ShaFullGMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption()) :
        BaseGMM(mu, Sigma, p, option) {}
    ShaFullGMM(ShaFullGMM &sfGMM) : BaseGMM(sfGMM) {}
    ShaFullGMM& operator=(const ShaFullGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    void fit(MatrixXd data) override;
};

class ShaDiagGMM : public BaseGMM {
public:
    ShaDiagGMM() = default;
    ShaDiagGMM(long nComponents, long nDimensions, fitOption option = fitOption()) :
        BaseGMM(nComponents, nDimensions, option) {}
    ShaDiagGMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption()) :
        BaseGMM(mu, Sigma, p, option) {}
    ShaDiagGMM(ShaDiagGMM &sdGMM) : BaseGMM(sdGMM) {}
    ShaDiagGMM& operator=(const ShaDiagGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    void fit(MatrixXd data) override;
};

class ShaSpheGMM : public BaseGMM {
public:
    ShaSpheGMM() = default;
    ShaSpheGMM(long nComponents, long nDimensions, fitOption option = fitOption()) :
        BaseGMM(nComponents, nDimensions, option) {}
    ShaSpheGMM(MatrixXd mu, vector<MatrixXd> Sigma, VectorXd p, fitOption option = fitOption()) :
        BaseGMM(mu, Sigma, p, option) {}
    ShaSpheGMM(ShaSpheGMM &ssGMM) : BaseGMM(ssGMM) {}
    ShaSpheGMM& operator=(const ShaSpheGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    void fit(MatrixXd data) override;
};

/// Random number generator
int myRandom(int i);
/*! 
    * \brief Random permutation generator.
    * @param n The function permutates 1:n.
*/
VectorXd randperm(unsigned long n);

/*!
    * \brief K-Means class. Provide fit functionality.
*/

void kMeans(MatrixXd data, long nComponents, VectorXd &idx,
        VectorXd &p, MatrixXd &mu, vector<MatrixXd> Sigma);

/*!
    * \brief Read BaseGMM from a file stream.
*/
ifstream & operator>>(ifstream &in, BaseGMM &BaseGMM);
/*!
    * \brief Print BaseGMM.
*/
ostream & operator<<(ostream &os, const BaseGMM &BaseGMM);
/*!
    * \brief write BaseGMM into a file
*/
ofstream & operator<<(ofstream &out, const BaseGMM &BaseGMM);

#endif /* defined(__BaseGMM__BaseGMM__) */
