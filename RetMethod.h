/*==========================================================================
 *
 *  Original source copyright (c) 2001, Carnegie Mellon University.
 *  See copyright.cmu for details.
 *  Modifications copyright (c) 2002, University of Massachusetts.
 *  See copyright.umass for details.
 *
 *==========================================================================
 */


//#ifndef _SIMPLEKLRETMETHOD_HPP
//#define _SIMPLEKLRETMETHOD_HPP

#ifndef RetttMethod_H_
#define RetttMethod_H_

#include <cmath>
#include "UnigramLM.hpp"
#include "ScoreFunction.hpp"
#include "DocModel.h"
#include "TextQueryRep.hpp"
#include "TextQueryRetMethod.h"
#include "Counter.hpp"
#include "DocUnigramCounter.hpp"

#include "Parameters.h"


extern double negGenMUHM;
extern int RSMethodHM;

namespace lemur 
{
namespace retrieval
{

/// Query model representation for the simple KL divergence model

class QueryModel : public ArrayQueryRep {
public:
    /// construct a query model based on query text
    QueryModel(const lemur::api::TermQuery &qry,
               const lemur::api::Index &dbIndex) :
        ArrayQueryRep(dbIndex.termCountUnique()+1, qry, dbIndex), qm(NULL),
        ind(dbIndex), colKLComputed(false) {

        DNsize = 0;

        colQLikelihood = 0;
        colQueryLikelihood();
    }

    /// construct an empty query model
    QueryModel(const lemur::api::Index &dbIndex) :
        ArrayQueryRep(dbIndex.termCountUnique()+1), qm(NULL), ind(dbIndex),
        colKLComputed(false) {
        colQLikelihood = 0;

        DNsize = 0;
        startIteration();
        while (hasMore()) {
            lemur::api::QueryTerm *qt = nextTerm();
            countInNonRel[qt->id()] = 0;
            setCount(qt->id(), 0);
            delete qt;
        }
    }


    virtual ~QueryModel(){ if (qm) delete qm;}


    /// interpolate the model with any (truncated) unigram LM, default parameter  to control the truncation is the number of words
    /*!
        The interpolated model is defined as <tt> origModCoeff</tt>*p(w|original_model)+(1-<tt>origModCoeff</tt>*p(w|new_truncated_model).
        <p> The "new truncated model" gives a positive probability to all words that "survive" in the truncating process, but gives a zero probability to all others.
        So, the sum of all word probabilities according to the truncated model does not
        have to sum to 1. The assumption is that if a word has an extrememly small probability, adding it to the query model will not affect scoring that much. <p> The truncation procedure is as follows:  First, we sort the probabilities in <tt> qModel</tt> passed in, and then iterate over all the entries. For each entry, we check the stopping condition and add the entry to the existing query model if none of the following stopping conditions is satisfied. If, however, any of the conditions is satisfied, the process will terminate. The three stopping conditions are: (1) We already added <tt>howManyWord</tt> words. (2) The total sum of probabilities added exceeds the threshold <tt>prSumThresh</tt>. (3) The probability of the current word is below <tt>prThresh</tt>.
      */

    virtual void interpolateWith(const lemur::langmod::UnigramLM &qModel,
                                 double origModCoeff,
                                 int howManyWord, double prSumThresh=1,
                                 double prThresh=0);
    virtual double scoreConstant() const {
        return totalCount();
    }

    /// load a query model/rep from input stream is
    virtual void load(istream &is);

    /// save a query model/rep to output stream os
    virtual void save(ostream &os);

    /// save a query clarity to output stream os
    virtual void clarity(ostream &os);
    /// compute query clarity score
    virtual double clarity() const;

    /// get and compute if necessary query-collection KL-div (useful for recovering the true divergence value from a score)
    double colDivergence() const {
        if (colKLComputed) {
            return colKL;
        } else {
            colKLComputed = true;
            double d=0;
            startIteration();
            while (hasMore()) {
                lemur::api::QueryTerm *qt=nextTerm();
                double pr = qt->weight()/(double)totalCount();
                double colPr = ((double)ind.termCount(qt->id()) /
                                (double)(ind.termCount())); // ML smoothing
                d += pr*log(pr/colPr);
                delete qt;
            }
            colKL=d;
            return d;
        }
    }

    ///compute the KL-div of the query model and any unigram LM, i.e.,D(Mq|Mref)
    double KLDivergence(const lemur::langmod::UnigramLM &refMod) {
        double d=0;
        startIteration();
        while (hasMore()) {
            lemur::api::QueryTerm *qt=nextTerm();
            double pr = qt->weight()/(double)totalCount();
            d += pr*log(pr/refMod.prob(qt->id()));
            delete qt;
        }
        return d;
    }

    double colQueryLikelihood() const {
        if (colQLikelihood == 0) {
            //Sum w in Q qtf * log(qtcf/termcount);
            lemur::api::COUNT_T tc = ind.termCount();
            startIteration();
            while (hasMore()) {
                lemur::api::QueryTerm *qt = nextTerm();
                lemur::api::TERMID_T id = qt->id();
                double qtf = qt->weight();
                lemur::api::COUNT_T qtcf = ind.termCount(id);
                double s = qtf * log((double)qtcf/(double)tc);
                colQLikelihood += s;
                delete qt;
            }
        }
        return colQLikelihood;
    }

    /*
     * wichMethod: 0 --> baseline collection
     *             1 --> baseline nonRel
    */
    double negativeQueryGeneration( const lemur::api::DocumentRep *dRep, vector<int> JudgDocs  , int whichMethod , bool newNonRel, double negMu , double delta) const
    {

        if(whichMethod == 0)//baseline(collection)
        {
            double readedDelta=delta;//0.007;

            double mu= negMu;//ind.docLengthAvg();//negGenMUHM;//2500;
            negQueryGen =0;


            lemur::api::COUNT_T tc = ind.termCount();
            startIteration();
            lemur::utility::HashFreqVector hfv(ind,dRep->getID());
            //cout<<"Did: "<<ind.document(dRep->getID())<<endl;
            while (hasMore())
            {

                lemur::api::QueryTerm *qt = nextTerm();
                double pwq = qt->weight()/totalCount();
                //cout<<qt->id()<<" "<<pwq<<endl;

                double delta =readedDelta;

                int freq=0;
                hfv.find(qt->id(),freq);
                if(freq>0)
                    delta =0.0;

                lemur::api::TERMID_T id = qt->id();

                lemur::api::COUNT_T qtcf = ind.termCount(id);

                double pwc = (double)qtcf/(double)tc;
                double pwdbar = (delta/(delta*ind.termCountUnique()+mu))+((mu*pwc)/(delta*ind.termCountUnique()+mu));
                negQueryGen+= pwq *log(pwq/pwdbar);
                //cout<<ind.termCount(qt->id())<<" "<<ind.term(qt->id())<<": "<<delta<<","<<pwdbar<<endl;
                delete qt;
            }
            return negQueryGen;

        }else if (whichMethod == 1)// using DN instead of collection
        {
            if (newNonRel)
                DNsize += ind.docLength(JudgDocs[JudgDocs.size()-1]);
            //cout<<"Did: "<<ind.document(dRep->getID())<<" DNsize: "<<DNsize<<endl;
            double mu= negMu;//ind.docLengthAvg();//negGenMUHM;//2500;
            negQueryGen =0;
            //if(negQueryGen == 0)
            //{
            lemur::api::COUNT_T tc = ind.termCount();
            startIteration();
            double surat = 0, makhraj = 0, cwdbar = 0;
            lemur::utility::HashFreqVector hfv(ind,dRep->getID()), *hfv2;
            /*
            while(hasMore()){
                lemur::api::QueryTerm *qt = nextTerm();
                int freq=0 ;
                hfv.find(qt->id(),freq);
                if(freq>0)
                    cwdbar = 0;
                else
                {
                    cwdbar = countInNonRel[qt->id()];
                }
                if (cwdbar != 0)
                    surat+= cwdbar/DNsize;
                lemur::api::TERMID_T id = qt->id();
                lemur::api::COUNT_T qtcf = ind.termCount(id);
                double pwc = (double)qtcf/(double)tc;

                makhraj+= pwc;
                //cout<<ind.term(qt->id())<<": "<<cwdbar<<","<<DNsize<<"-> "<<cwdbar/DNsize<<endl;
            }
            double alpha_d = (1.0-surat)/makhraj;
            */
            if (newNonRel)
                hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);
            startIteration();
            while (hasMore())
            {
                //cout<<"hala "<<DNsize<<endl;
                lemur::api::QueryTerm *qt = nextTerm();
                double pwq = qt->weight()/totalCount();
                if (newNonRel)
                {
                    int freq;
                    hfv2->find(qt->id(),freq);
                    countInNonRel[qt->id()] += freq;
                }
                double cwdbar = 0;

                int freq=0 ;
                hfv.find(qt->id(),freq);
                if(freq>0){
                    cwdbar = 0;
                    //delete qt;
                    //cout<<"boogh"<<endl;
                    //continue;
                }
                else
                {
                   // cout<<"booooooooooooogh"<<endl;
                    /*for (int i = 0 ; i<JudgDocs.size() ; i++){
                            lemur::utility::HashFreqVector hfv(ind,JudgDocs[i]);
                            hfv.find(qt->id(),freq);
                            cwdbar += freq;
                            DNsize += ind.docLength(JudgDocs[i]);
                        }*/
                    cwdbar = countInNonRel[qt->id()];
                }
                lemur::api::TERMID_T id = qt->id();

                lemur::api::COUNT_T qtcf = ind.termCount(id);

                //DNsize = countInNonRel.size();//????????????????????????????????????

                double pwc = (double)qtcf/(double)tc;
                double pwdbar;
                //if (cwdbar != 0)
                    pwdbar = (cwdbar/((DNsize+mu)))+((mu*pwc)/(DNsize+mu));
                //else
                  //  pwdbar = (delta/(delta*ind.termCountUnique()+mu))+((mu*pwc)/(delta*ind.termCountUnique()+mu));
                /*if (cwdbar == 0)
                    pwdbar = pwc;
                else
                    pwdbar = (alpha_d*cwdbar)/DNsize;*/
                    //if (freq==0)
                        cout<<freq<<" "<< pwq *log(pwq/pwdbar)<<endl;
                        negQueryGen+= pwq *log(pwq/pwdbar);
                //cout<<ind.term(qt->id())<<" freq: "<<freq<<" (cwdbar/(DNsize+mu)): "<<(cwdbar/(DNsize+mu))<<" pwdbar: "<<pwdbar<<endl;
                //cout<<"na hala DNsize: "<<DNsize<<"\nnegQueryGen: "<<negQueryGen<<endl<<endl;
             //   cout<<"cwdbar: "<<cwdbar<<"\npwc: "<<pwc<<"\npwdbar: "<<pwdbar<<endl;
               // cout<<ind.termCount(qt->id())<<" "<<ind.term(qt->id())<<": "<<cwdbar<<","<<pwdbar<<endl;
                delete qt;


            }
            if (newNonRel)
                delete hfv2;
            //}
//            cout<<"Did: "<<dRep->getID()<<endl;
            //cout<<"DNsize: "<<DNsize<<"\nnegQueryGen: "<<negQueryGen<<endl<<endl;
            return negQueryGen;
            //cout<<negQueryGen<<"dddddddddddd\n";
        }

    }

    double negativeKL(const lemur::api::DocumentRep *dRep, vector<int> JudgDocs, bool newNonRel, double negMu, double beta = 1) const
    {
       if (newNonRel)
            DNsize += ind.docLength(JudgDocs[JudgDocs.size()-1]);

        double mu= negMu;//ind.docLengthAvg();//negGenMUHM;//2500;
        negQueryGen =0;
        //if(negQueryGen == 0)
        //{
        lemur::api::COUNT_T tc = ind.termCount();
        startIteration();
        lemur::utility::HashFreqVector hfv(ind,dRep->getID()), *hfv2;
        if (newNonRel)
            hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);
        while (hasMore())
        {

            lemur::api::QueryTerm *qt = nextTerm();
            double pwq = qt->weight()/totalCount();//????????????????????????????????????????
            if (newNonRel)
            {
                int freq;
                hfv2->find(qt->id(),freq);
                countInNonRel[qt->id()] += freq;
            }
            double cwdbar = 0;
            int freq=0 ;
            hfv.find(qt->id(),freq);
            cwdbar = countInNonRel[qt->id()];
            //Query Term Elimination
            if (RSMethodHM==2 && freq > 0){
                //cout<<"miad!"<<endl;
                cwdbar = 0;
            }
            lemur::api::TERMID_T id = qt->id();

            lemur::api::COUNT_T qtcf = ind.termCount(id);

            double pwc = (double)qtcf/(double)tc;
            double pwd =  ((double)freq/((double)ind.docLength(dRep->getID())+mu))+((mu*pwc)/((double)ind.docLength(dRep->getID())+mu)) ;
            double pwdbar = (cwdbar/(DNsize+mu))+((mu*pwc)/(DNsize+mu));
            negQueryGen+= pwdbar *log(pwdbar/pwd);


         //   cout<<"cwdbar: "<<cwdbar<<"\npwc: "<<pwc<<"\npwdbar: "<<pwdbar<<endl;

            delete qt;


        }
        if (newNonRel)
            delete hfv2;
        //}
        //    cout<<"Did: "<<dRep->getID()<<endl;
        //  cout<<"DNsize: "<<DNsize<<"\nnegQueryGen: "<<negQueryGen<<endl<<endl;
        return negQueryGen;


    }
    double interpolateSimsScore(lemur::api::TextQueryRep *textQR,int docID ,
                                                         vector<int> relJudgDoc ,vector<int> nonReljudgDoc , bool newNonRel)
    {
        return 0.0;
#if 0
        double relSim =0.0, nonRelSim = 0.0;
        double mu = 2500;

        //***********nonRelSim*******************//
        if (newNonRel)
            DNsize += ind.docLength(JudgDocs[JudgDocs.size()-1]);

        lemur::api::COUNT_T tc = ind.termCount();
        startIteration();
        lemur::utility::HashFreqVector hfv(ind,dRep->getID()), *hfv2;
        if (newNonRel)
            hfv2 = new lemur::utility::HashFreqVector(ind,JudgDocs[JudgDocs.size()-1]);
        while (hasMore())
        {
            lemur::api::QueryTerm *qt = nextTerm();
            double pwq = qt->weight()/totalCount();
            if (newNonRel)
            {
                int freq;
                hfv2->find(qt->id(),freq);
                countInNonRel[qt->id()] += freq;
            }
            double cwdbar = 0;
            int freq=0 ;
            hfv.find(qt->id(),freq);
            /*
            if(freq>0)
                cwdbar = 0;
            else
            {
                cwdbar = countInNonRel[qt->id()];
            }
            */

            lemur::api::TERMID_T id = qt->id();

            lemur::api::COUNT_T qtcf = ind.termCount(id);

            double pwc = (double)qtcf/(double)tc;

            cwdbar = countInNonRel[qt->id()];

            double pwdbar = (cwdbar/(DNsize+mu))+((mu*pwc)/(DNsize+mu));
            negQueryGen+= pwq *log(pwq/pwdbar);

            delete qt;
        }

        if (newNonRel)
            delete hfv2;
#endif
    }



protected:
    // For Query likelihood adjusted score
    mutable double negQueryGen;
    mutable double colQLikelihood;
    mutable double colKL;
    mutable bool colKLComputed;

    mutable map <int, int> countInNonRel;
    mutable int DNsize ;



    lemur::api::IndexedRealVector *qm;
    const lemur::api::Index &ind;
};



/// Simple KL-divergence scoring function
/*!
      The KL-divergence formula D(model_q || model_d), when used for ranking
      documents, can be computed
      efficiently by re-writing the formula as a sum over all matched
      terms in a query and a document. The details of such rewriting are
      described in the following two papers:
      <ul>
      <li>C. Zhai and J. Lafferty. A study of smoothing methods for language models applied to ad hoc
      information retrieval, In 24th ACM SIGIR Conference on Research and Development in Information
      Retrieval (SIGIR'01), 2001.
      <li>P. Ogilvie and J. Callan. Experiments using the Lemur toolkit. In Proceedings of the Tenth Text
      Retrieval Conference (TREC-10).
      </ul>
    */

class ScoreFunc : public lemur::api::ScoreFunction {
public:
    enum RetParameter::adjustedScoreMethods adjScoreMethod;
    void setScoreMethod(enum RetParameter::adjustedScoreMethods adj) {
        adjScoreMethod = adj;
    }
    virtual double matchedTermWeight(const lemur::api::QueryTerm *qTerm,
                                     const lemur::api::TextQueryRep *qRep,
                                     const lemur::api::DocInfo *info,
                                     const lemur::api::DocumentRep *dRep) const {
        double w = qTerm->weight();
        double d = dRep->termWeight(qTerm->id(),info);//d = p_seen(w|d)/(a(d)*p(w|C)) [slide7-11]
        double l = log(d);
        double score = w*l;
        //cout<<info->docID()<<endl;
        /*
          cerr << "M:" << qTerm->id() <<" d:" << info->docID() << " w:" << w
          << " d:" << d << " l:" << l << " s:" << score << endl;
        */
        return score;
        //    return (qTerm->weight()*log(dRep->termWeight(qTerm->id(),info)));
    }
    /// score adjustment (e.g., appropriate length normalization)
    virtual double adjustedScore(double origScore,
                                 const lemur::api::TextQueryRep *qRep,
                                 const lemur::api::DocumentRep *dRep) const {
        const QueryModel *qm = dynamic_cast<const QueryModel *>(qRep);
        // this cast is unnecessary
        //SimpleKLDocModel *dm = (SimpleKLDocModel *)dRep;
        // dynamic_cast<SimpleKLDocModel *>dRep;

        double qsc = qm->scoreConstant();//|q|
        double dsc = log(dRep->scoreConstant());//log(a(d))
        double cql = qm->colQueryLikelihood();//sigma(c(w,q)*P(w|C))
        // real query likelihood
        double s = dsc * qsc + origScore + cql;
        double qsNorm = origScore/qsc;
        double qmD = qm->colDivergence();
        /*
          cerr << "A:"<< origScore << " dsc:" << dsc  << " qsc:" << qsc
          << " cql:" << cql << " s:"  << s << endl;
        */
        /// The following are three different options for scoring
        switch (adjScoreMethod) {
        case RetParameter::QUERYLIKELIHOOD:
            /// ==== Option 1: query likelihood ==============
            // this is the original query likelihood scoring formula
            return s;
            //      return (origScore+log(dm->scoreConstant())*qm->scoreConstant());
        case RetParameter::CROSSENTROPY:
            /// ==== Option 2: cross-entropy (normalized query likelihood) ====
            // This is the normalized query-likelihood, i.e., cross-entropy
            assert(qm->scoreConstant()!=0);
            // return (origScore/qm->scoreConstant() + log(dm->scoreConstant()));
            // add the term colQueryLikelihood/qm->scoreConstant
            s = qsNorm + dsc + cql/qsc;
            return (s);
        case RetParameter::NEGATIVEKLD:
            /// ==== Option 3: negative KL-divergence ====
            // This is the exact (negative) KL-divergence value, i.e., -D(Mq||Md)
            assert(qm->scoreConstant()!=0);
            s = qsNorm + dsc - qmD;
            /*
            cerr << origScore << ":" << qsNorm << ":" << dsc  << ":" << qmD  << ":" << s << endl;
          */
            return s;
            //      return (origScore/qm->scoreConstant() + log(dm->scoreConstant())
            //          - qm->colDivergence());
        default:
            cerr << "unknown adjusted score method" << endl;
            return origScore;
        }
    }

};

/// KL Divergence retrieval model with simple document model smoothing
class RetMethod : public lemur::api::TextQueryRetMethod {
public:

    /// Construction of SimpleKLRetMethod requires a smoothing support file, which can be generated by the application GenerateSmoothSupport. The use of this smoothing support file is to store some pre-computed quantities so that the scoring procedure can be speeded up.
    RetMethod(const lemur::api::Index &dbIndex,
              const string &supportFileName,
              lemur::api::ScoreAccumulator &accumulator);
    virtual ~RetMethod();

    virtual lemur::api::TextQueryRep *computeTextQueryRep(const lemur::api::TermQuery &qry) {
        return (new QueryModel(qry, ind));
    }

    virtual lemur::api::DocumentRep *computeDocRep(lemur::api::DOCID_T docID);


    virtual lemur::api::ScoreFunction *scoreFunc() {
        return (scFunc);
    }

    virtual void updateTextQuery(lemur::api::TextQueryRep &origRep,
                                 const lemur::api::DocIDSet &relDocs, const lemur::api::DocIDSet &nonRelDocs);

    virtual void updateProfile(lemur::api::TextQueryRep &origRep,
                               vector<int> relJudglDoc ,vector<int> nonReljudgDoc);
    virtual void updateThreshold(lemur::api::TextQueryRep &origRep,
                                 vector<int> relJudglDoc ,vector<int> nonReljudgDoc ,int mode,double relSumScores , double nonRelSumScores);
    virtual float computeProfDocSim(lemur::api::TextQueryRep *origRep,int docID ,vector<int>relDocs ,vector<int>nonRelDocs , bool newNonRel);


    void setDocSmoothParam(RetParameter::DocSmoothParam &docSmthParam);
    void setQueryModelParam(RetParameter::QueryModelParam &queryModParam);


    double getThreshold(){return mozhdehHosseinThreshold;}
    void setThreshold(double thr)
    {
        mozhdehHosseinThreshold = thr;
    }
    double getNegWeight(){return mozhdehHosseinNegWeight;}
    void setNegWeight(double negw)
    {
        mozhdehHosseinNegWeight =negw;
    }

    //for linear thr updating method
    double getC1(){return C1;}
    void setC1(double val){C1=val;}
    double getC2(){return C2;}
    void setC2(double val){C2=val;}

    //for diff thr updating method
    void setDiffThrUpdatingParam(double alpha){diffThrUpdatingParam=alpha;}
    double getDiffThrUpdatingParam(){return diffThrUpdatingParam;}


    double getNegMu(){return NegMu;}
    void setNegMu(double val){NegMu=val;}

    double getDelta(){return delta;}
    void setDelta(double val){delta = val;}

protected:
    double mozhdehHosseinThreshold;
    double mozhdehHosseinNegWeight;
    mutable int thresholdUpdatingMethod;/* 0->no updating
                                           1->linear
                                           2->diff
                                        */
    double C1,C2;//for linear
    double diffThrUpdatingParam;//for diff

    double NegMu;
    double delta;

    //Matrix Factorization method for query expansion
    bool MF;

    /// needed for fast one-step Markov chain
    double *mcNorm;

    /// needed for fast alpha computing
    double *docProbMass;
    /// needed for supporting fast absolute discounting
    lemur::api::COUNT_T *uniqueTermCount;
    /// a little faster if pre-computed
    lemur::langmod::UnigramLM *collectLM;
    /// support the construction of collectLM
    lemur::langmod::DocUnigramCounter *collectLMCounter;
    /// keep a copy to be used at any time
    ScoreFunc *scFunc;

    /// @name query model updating methods (i.e., feedback methods)
    //@{
    /// Mixture model feedback method
    void computeMixtureFBModel(QueryModel &origRep,
                               const lemur::api::DocIDSet & relDocs, const lemur::api::DocIDSet &nonRelDocs);
    /// Divergence minimization feedback method
    void computeDivMinFBModel(QueryModel &origRep,
                              const lemur::api::DocIDSet &relDocs);
    void computeMEDMMFBModel(QueryModel &origRep,
                             const lemur::api::DocIDSet &relDocs);

    /// Markov chain feedback method
    void computeMarkovChainFBModel(QueryModel &origRep,
                                   const lemur::api::DocIDSet &relDocs) ;
    /// Relevance model1 feedback method
    void computeRM1FBModel(QueryModel &origRep,
                           const lemur::api::DocIDSet & relDocs,const lemur::api::DocIDSet & nonRelDocs);
    /// Relevance model1 feedback method
    void computeRM2FBModel(QueryModel &origRep,
                           const lemur::api::DocIDSet & relDocs);
    void computeRM3FBModel(QueryModel &origRep,
                           const lemur::api::DocIDSet & relDocs);
    void computeRM4FBModel(QueryModel &origRep,
                           const lemur::api::DocIDSet & relDocs);



    //@}

    RetParameter::DocSmoothParam docParam;
    RetParameter::QueryModelParam qryParam;

    /// Load support file support
    void loadSupportFile();
    const string supportFile;
};


inline  void RetMethod::setDocSmoothParam(RetParameter::DocSmoothParam &docSmthParam)
{
    docParam = docSmthParam;
    loadSupportFile();
}

inline  void RetMethod::setQueryModelParam(RetParameter::QueryModelParam &queryModParam)
{
    qryParam = queryModParam;
    // add a parameter to the score function.
    // isn't available in the constructor.
    scFunc->setScoreMethod(qryParam.adjScoreMethod);
    loadSupportFile();
}
}
}

#endif /* _SIMPLEKLRETMETHOD_HPP */
