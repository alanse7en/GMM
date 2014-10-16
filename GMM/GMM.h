
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
     *Show information during training (default); 'final' show final information; 'off' show nothing
     */
    unsigned int maxIter = 1000;
    /*!<
     *Maximum iterations allowed, default is 1000;
     */
    double tolFun = 1e-6;
    /*!<
     *The termination tolerance for the log-likelihood function, default is 1e-6.
     */
};

/*! Training result */
struct fitResult {
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
    fitResult result = fitResult();/*!< Containing training results about the BaseGMM model. */
    /*! 
        * Check the train data to ensure it is valid.
        * @param data The train data matrix to be checked.
    */
    void checkData(const MatrixXd &data);
    /*!
        * Check the option to ensure it is valid.
        * @param option The option for fit function to be checked.
    */
    void checkOption();
    /*!
        * Check model to ensure it is valid.
    */
    void checkModel();
    /*!
        * Check if there is Not-a-Number in the matrix
    */
    void checkNaN(const MatrixXd &mat);
    /*!
        * Initialize model parameters.
        * @param data The train data matrix.
    */
    void initParam(const MatrixXd &data);
    /*!
        * @param data A N-by-D matrix containing the data;
        * @param lh A N-by-K matrix containing the likelihood,
        * lh(i, j) = Pr(datapoint I | component J)
    */
    void likelihood(const MatrixXd &data, MatrixXd &lh);
    /*! calculate posterior probability of given data and BaseGMM
        * @param data A N-by-D matrix containing the data;
        * @return The Log-likelihood loss
    */
    double posterior(const MatrixXd &data, MatrixXd &post);
    
public:
    /// Default constructor
    BaseGMM() = default;
    ///  Constructors with nComponents, nDimensions and option
    BaseGMM(long nComponents, long nDimensions, const fitOption &option = fitOption(),
            const fitResult &result = fitResult());
    ///  Constructors with mu, Sigma, p, and option
    BaseGMM(const MatrixXd &mu, const vector<MatrixXd> &Sigma, const VectorXd &p,
            const fitOption &option = fitOption(), const fitResult &result = fitResult());
    ///  Copy constructors
    BaseGMM(const BaseGMM &) = default;
    ///  = operator
    BaseGMM& operator= (const BaseGMM &) = default;
    /// Destructor
    virtual ~BaseGMM() = default;
    /// Show training result
    void showResult();
    /*!
        * \brief Train BaseGMM to fit data;
        * @param data A N-by-D matrix containing the train data.
    */
    virtual void fit(const MatrixXd &data) = 0;
    /*!
        * \brief Cluster the with trained model.
        * @param data  A N-by-D matrix containing the test data;
        * @param post  A N-by-K matrix containing the posterior peobability of p(component J | dataPoint I);
        * @return The negative Log likelihood
    */
    double cluster(const MatrixXd &data, MatrixXd &post);
    /*!
        * \brief Cluster the with trained model.
        * @param data  A N-by-D matrix containing the test data;
        * @param idx  A N-by-1 vector representing N data points' correspoding Gaussian components;
        * @param post  A N-by-K matrix containing the posterior peobability of p(component J | dataPoint I);
        * @return The negative Log likelihood
    */
    double cluster(const MatrixXd &data, MatrixXd &post, VectorXd &idx);
};

class DiffFullGMM : public BaseGMM {
public:
    /// default constructor
    DiffFullGMM() = default;
    /// constructor with nComponents, nDimensions, fit option and fit result
    DiffFullGMM(long nComponents, long nDimensions, fitOption option = fitOption(),
                fitResult result = fitResult()) :
        BaseGMM(nComponents, nDimensions, option, result) {}
    /// constructor with mu ,Sigma, p, fit option and fit result
    DiffFullGMM(const MatrixXd &mu, const vector<MatrixXd> &Sigma, const VectorXd &p,
                const fitOption &option = fitOption(), const fitResult &result = fitResult()) :
        BaseGMM(mu, Sigma, p, option, result) {}
    /// copy constructor
    DiffFullGMM(const DiffFullGMM &dfGMM) : BaseGMM(dfGMM) {}
    /// copy assignment operator
    DiffFullGMM& operator=(const DiffFullGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    void fit(const MatrixXd &data) override;
};

class DiffDiagGMM : public BaseGMM {
public:
    DiffDiagGMM() = default;
    DiffDiagGMM(long nComponents, long nDimensions, fitOption option = fitOption(),
                fitResult result = fitResult()) :
        BaseGMM(nComponents, nDimensions, option, result) {}
    DiffDiagGMM(const MatrixXd &mu, const vector<MatrixXd> &Sigma, const VectorXd &p,
                const fitOption &option = fitOption(), const fitResult &result = fitResult()) :
        BaseGMM(mu, Sigma, p, option, result) {}
    DiffDiagGMM(const DiffDiagGMM &ddGMM) : BaseGMM(ddGMM) {}
    DiffDiagGMM& operator=(const DiffDiagGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    operator DiffFullGMM () const {
        DiffFullGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    void fit(const MatrixXd &data) override;
};

class DiffSpheGMM : public BaseGMM {
public:
    DiffSpheGMM() = default;
    DiffSpheGMM(long nComponents, long nDimensions, fitOption option = fitOption(),
                fitResult result = fitResult()) :
        BaseGMM(nComponents, nDimensions, option, result) {}
    DiffSpheGMM(const MatrixXd &mu, const vector<MatrixXd> &Sigma, const VectorXd &p,
                const fitOption &option = fitOption(), const fitResult &result = fitResult()) :
        BaseGMM(mu, Sigma, p, option, result) {}
    DiffSpheGMM(const DiffSpheGMM &dsGMM) : BaseGMM(dsGMM) {}
    DiffSpheGMM& operator=(const DiffSpheGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    operator DiffFullGMM () const {
        DiffFullGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    operator DiffDiagGMM () const {
        DiffDiagGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    void fit(const MatrixXd &data) override;
};

class ShaFullGMM : public BaseGMM {
public:
    ShaFullGMM() = default;
    ShaFullGMM(long nComponents, long nDimensions, fitOption option = fitOption(),
               fitResult result = fitResult()) :
        BaseGMM(nComponents, nDimensions, option, result) {}
    ShaFullGMM(const MatrixXd &mu, const vector<MatrixXd> &Sigma, const VectorXd &p,
               const fitOption &option = fitOption(), const fitResult &result = fitResult()) :
        BaseGMM(mu, Sigma, p, option, result) {}
    ShaFullGMM(const ShaFullGMM &sfGMM) : BaseGMM(sfGMM) {}
    ShaFullGMM& operator=(const ShaFullGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    operator DiffFullGMM () const {
        DiffFullGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    void fit(const MatrixXd &data) override;
};

class ShaDiagGMM : public BaseGMM {
public:
    ShaDiagGMM() = default;
    ShaDiagGMM(long nComponents, long nDimensions, fitOption option = fitOption(),
               fitResult result = fitResult()) :
        BaseGMM(nComponents, nDimensions, option, result) {}
    ShaDiagGMM(const MatrixXd &mu, const vector<MatrixXd> &Sigma, const VectorXd &p,
               const fitOption &option = fitOption(), const fitResult &result = fitResult()) :
        BaseGMM(mu, Sigma, p, option,result) {}
    ShaDiagGMM(const ShaDiagGMM &sdGMM) : BaseGMM(sdGMM) {}
    ShaDiagGMM& operator=(const ShaDiagGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    operator DiffDiagGMM () const {
        DiffDiagGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    operator ShaFullGMM () const {
        ShaFullGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    operator DiffFullGMM () const {
        DiffFullGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    void fit(const MatrixXd &data) override;
};

class ShaSpheGMM : public BaseGMM {
public:
    ShaSpheGMM() = default;
    ShaSpheGMM(long nComponents, long nDimensions, fitOption option = fitOption(),
               fitResult result = fitResult()) :
        BaseGMM(nComponents, nDimensions, option, result) {}
    ShaSpheGMM(const MatrixXd &mu, const vector<MatrixXd> &Sigma, const VectorXd &p,
               const fitOption &option = fitOption(), const fitResult &result = fitResult()) :
        BaseGMM(mu, Sigma, p, option, result) {}
    ShaSpheGMM(const ShaSpheGMM &ssGMM) : BaseGMM(ssGMM) {}
    ShaSpheGMM& operator=(const ShaSpheGMM& rhs) {
        BaseGMM::operator=(rhs);
        return *this;
    }
    operator DiffFullGMM () const {
        DiffFullGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    operator DiffSpheGMM () const {
        DiffSpheGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    operator DiffDiagGMM () const {
        DiffDiagGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    operator ShaFullGMM () const {
        ShaFullGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    operator ShaDiagGMM () const {
        ShaDiagGMM ret(mu, Sigma, p, option, result);
        return ret;
    }
    void fit(const MatrixXd &data) override;
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

void kMeans(const MatrixXd &data, long nComponents, VectorXd &idx, MatrixXd &mu);

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
