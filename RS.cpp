#ifndef SMTH
#define SMTH

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <sstream>
#include <DocStream.hpp>
#include <BasicDocStream.hpp>
#include "IndexManager.hpp"
#include "ResultFile.hpp"
//#include "DocUnigramCounter.hpp"
#include "RetMethod.h"
#include "QueryDocument.hpp"
#include <sstream>

#include "Parameters.h"

using namespace lemur::api;
using namespace lemur::langmod;
using namespace lemur::parse;
using namespace lemur::retrieval;
using namespace std;

void loadJudgment();
void computeRSMethods(Index *);
void MonoKLModel(Index* ind);
vector<int> queryDocList(Index* ind,TextQueryRep *textQR);


template <typename T>
string numToStr(T number)
{
    ostringstream s;
    s << number;
    return s.str();
}

extern double startThresholdHM , endThresholdHM , intervalThresholdHM ;
extern int WHO;// 0--> server , 1-->Mozhdeh, 2-->AP, other-->Hossein
extern string outputFileNameHM;
extern string resultFileNameHM;
extern int feedbackMode;
extern double startNegWeight,endNegWeight , negWeightInterval;
extern double startNegMu, endNegMu, NegMuInterval;
extern double startDelta, endDelta, deltaInterval;
extern int RSMethodHM;
extern int negGenModeHM;
extern double smoothJMInterval1,smoothJMInterval2;

extern int updatingThresholdMode;

map<string , vector<string> >queryRelDocsMap;
string judgmentPath,indexPath,queryPath;
string resultPath = "";
//int numberOfProcessedQueries = 0 , numberOfQueries=0;

#define DATASET 0 //0-->infile, 1-->ohsu

int main(int argc, char * argv[])
{
    readParams(string(argv[1]));
    cout<< "reading param file: "<<argv[1]<<endl;
    switch (WHO)
    {
    case 0:
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/qrels_en";
            indexPath= "/home/iis/Desktop/Edu/thesis/index/infile/en/index.key";
            queryPath = "/home/iis/Desktop/Edu/thesis/Data/INFILE/q_en_titleKeyword_en.stemmed.xml";
        }else if(DATASET == 1)//ohsu
        {
            //judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-train/qrels.ohsu.adapt.87";
            judgmentPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-test/qrels.ohsu.88-91";
            //indexPath= "/home/iis/Desktop/Edu/thesis/index/ohsumed/ohsu/index/index.key";
            indexPath= "/home/iis/Desktop/Edu/thesis/index/ohsumed/ohsu/testIndex/index/index.key";
            queryPath = "/home/iis/Desktop/Edu/thesis/Data/ohsumed/trec9-train/trec9-train_output/stemmed_ohsu_query.txt";
        }
        break;
    case 1:
        judgmentPath = "/home/mozhdeh/Desktop/INFILE/hosein-data/qrels_en";
        indexPath = "/home/mozhdeh/Desktop/INFILE/javid-index/index.key";
        queryPath = "/home/mozhdeh/Desktop/INFILE/hosein-data/q_en_titleKeyword_en.stemmed.xml";
        break;
        //case 2:
        //    judgmentPath ="/home/mozhdeh/Desktop/AP/Data/jud-ap.txt";
        //    indexPath = "/home/mozhdeh/Desktop/AP/index/index.key";
        //   queryPath = "/home/mozhdeh/Desktop/AP/Data/topics.stemmed.xml";
        //   break;
    default:
        if(DATASET == 0)//infile
        {
            judgmentPath = "/home/hossein/Desktop/lemur/DataSets/Infile/Data/qrels_en";
            indexPath = "/home/hossein/Desktop/lemur/DataSets/Infile/Index/en/index.key";
            queryPath = "/home/hossein/Desktop/lemur/DataSets/Infile/Data/q_en_titleKeyword_en.stemmed.xml";//????????
        }else if(DATASET == 1)//ohsu
        {
            judgmentPath = "/home/hossein/Desktop/lemur/DataSets/Ohsumed/Data/trec9-train/qrels.ohsu.adapt.87";
            indexPath = "/home/hossein/Desktop/lemur/DataSets/Ohsumed/Index/trec9-train/index.key";
            queryPath = "/home/hossein/Desktop/lemur/DataSets/Ohsumed/Data/trec9-train/stemmed_ohsu_query.txt";

        }

        break;
    }

    Index *ind = IndexManager::openIndex(indexPath);// Your own path to index
    //cerr<<ind->term("the")<<endl;
    //cerr<<ind->term("doping");

    loadJudgment();
    computeRSMethods(ind);

    //MonoKLModel(ind);

}



void computeRSMethods(Index* ind)
{
    DocStream *qs = new BasicDocStream(queryPath); // Your own path to topics
    ArrayAccumulator accumulator(ind->docCount());
    RetMethod *myMethod = new RetMethod(*ind,"",accumulator);


    string outFilename;
    if(DATASET == 0)
    {
        outFilename =outputFileNameHM+"_infile";
    }
    else if (DATASET == 1)
    {
        outFilename =outputFileNameHM+"_ohsu";
    }
    ofstream out(outFilename.c_str());


    /*
#define RETMODE 1//LM(0) ,RS(1)
    //#define NEGMODE 0//coll(0) ,NonRel(1)
#define FBMODE 0//NoFB(0),NonRel(1),Normal(2)
#define UPDTHRMODE 0//No(0),Linear(1) ,Diff(2)
*/

#define RETMODE RSMethodHM//LM(0) ,RS(1), NegKLQTE(2),NegKL(3)
#define NEGMODE negGenModeHM//coll(0) ,NonRel(1)
#define FBMODE feedbackMode//NoFB(0),NonRel(1),Normal(2),Mixture(3)
#define UPDTHRMODE 1//updatingThresholdMode//No(0),Linear(1) ,Diff(2)

    cout<< "RSMethod: "<<RSMethodHM<<" NegGenMode: "<<negGenModeHM<<" feedbackMode: "<<feedbackMode<<" updatingThrMode: "<<updatingThresholdMode<<"\n";
    cout<< "RSMethod: "<<RETMODE<<" NegGenMode: "<<NEGMODE<<" feedbackMode: "<<FBMODE<<" updatingThrMode: "<<UPDTHRMODE<<"\n";
    cout<<"outfile: "<<outFilename<<endl;
    double start_thresh =startThresholdHM, end_thresh= endThresholdHM;
    double start_negMu =startNegMu, end_negMu= endNegMu;
    double start_delta =startDelta, end_delta= endDelta;


    //double global_rel_ret =0, global_ret = 0, global_all_rels ;
    double init_thr = start_thresh;

//#if !FBMODE && !UPDTHRMODE
    for (double thresh = start_thresh ; thresh<=end_thresh ; thresh += intervalThresholdHM)
    {
        //myMethod->setThreshold(thresh);
        myMethod->setThreshold(init_thr);
        for (double delta = start_delta ; delta<=end_delta ; delta += deltaInterval)
        {
            myMethod->setDelta(delta);
            resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_"+numToStr(delta)+".res";
            myMethod->setNegMu(2500);

            //for (double negmu = start_negMu ; negmu<=end_negMu ; negmu += NegMuInterval)
            {
                double negmu = 2500;//ind->docLengthAvg();
                myMethod->setNegMu(negmu);
                resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_"+numToStr(negmu)+".res";

                //for (double lambda_1 = 0 ; lambda_1<=1 ; lambda_1 += smoothJMInterval1){
                double lambda_1 = smoothJMInterval1;
                myMethod->setLambda1(lambda_1);
                for (double lambda_2 = 0 ; lambda_2<=1 ; lambda_2 += smoothJMInterval2)
                {
                    //double lambda_2 =smoothJMInterval2;//FIXME ????????????????????????????????????????????????
                    resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_"+numToStr(negmu)+"_lambda1:"+numToStr( lambda_1)+"_lambda2:"+numToStr( lambda_2)+".res";
                    myMethod->setLambda2(lambda_2);

//#endif


#if RETMODE == 1 && FBMODE == 1
                    double start_negThr = startNegWeight , end_negThr = endNegWeight;
                    for (double neg = start_negThr ; neg<=end_negThr ; neg += negWeightInterval)
                    {
                        myMethod->setNegWeight(neg);
                        out<<"negWeight: "<<myMethod->getNegWeight()<<endl;
                        resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_"+numToStr(neg)+".res";

#endif

#if UPDTHRMODE == 1

                        for(double c1 = 0.1 ; c1<=1 ;c1+=0.2)//inc
                        {
                            myMethod->setC1(c1);
                            for(double c2 = 0.1 ; c2 <= 1 ; c2+=0.2)//dec
                            {
                                myMethod->setThreshold(init_thr);
                                myMethod->setC2(c2);

                                //for(int numOfShownNonRel =1;numOfShownNonRel< 20;numOfShownNonRel+=2 )
                                int numOfShownNonRel = 5;
                                {

                                    //for(int numOfnotShownDoc = 20 ;numOfnotShownDoc <= 800 ; numOfnotShownDoc+=20)
                                    {
                                        int numOfnotShownDoc = 500;

                                        //myMethod->setThreshold(-5.0);
                                        cout<<"c1: "<<c1<<" c2: "<<c2<<" numOfShownNonRel: "<<numOfShownNonRel<<" numOfnotShownDoc: "<<numOfnotShownDoc<<" "<<endl;
                                        resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_c1:"+numToStr(c1)+"_c2:"+numToStr(c2)+"_#showNonRel:"+numToStr(numOfShownNonRel)+"_#notShownDoc:"+numToStr(numOfnotShownDoc)+".res";
#endif

#if UPDTHRMODE == 2
                                        for(double alph = 0.1 ; alph <= 1.0 ; alph+=0.1)
                                        {
                                            myMethod->setDiffThrUpdatingParam(alph);
                                            out<<"diff_alpha: "<<myMethod->getDiffThrUpdatingParam()<<endl;
                                            resultPath = resultFileNameHM.c_str() +numToStr( myMethod->getThreshold() )+"_alpha:"+numToStr(alph)+".res";
#endif


                                            IndexedRealVector results;

                                            out<<"threshold: "<<myMethod->getThreshold()<< " negmu: "<<myMethod->getNegMu();
                                            out<<" delta: "<<myMethod->getDelta()<<" lambda_1: "<<lambda_1<<" lambda_2: "<<lambda_2<<endl;

                                            qs->startDocIteration();
                                            TextQuery *q;


                                            //if(DATASET == 0)
                                            //    resultPath +="_infile.res";
                                            //else if (DATASET == 1)
                                            //    resultPath +="_ohsu.res";
                                            ofstream result(resultPath.c_str());
                                            ResultFile resultFile(1);
                                            resultFile.openForWrite(result,*ind);

                                            double relRetCounter = 0 , retCounter = 0 , relCounter = 0;
                                            vector<double> queriesPrecision,queriesRecall;
                                            while(qs->hasMore())
                                            {
                                                //myMethod->clearRelNonRelCountFlag();
#if UPDTHRMODE != 0
                                                myMethod->setThreshold(init_thr);
#endif


                                                double relSumScores =0.0,nonRelSumScores = 0.0;

                                                int numberOfNotShownDocs = 0,numberOfShownNonRelDocs = 0;

                                                vector<int> relJudgDocs,nonRelJudgDocs;
                                                results.clear();


                                                Document* d = qs->nextDoc();
                                                q = new TextQuery(*d);
                                                QueryRep *qr = myMethod->computeQueryRep(*q);
                                                cout<<"qid: "<<q->id()<<endl;

                                                bool newNonRel = false , newRel = false;
                                                vector<string> relDocs;

                                                if( queryRelDocsMap.find(q->id()) != queryRelDocsMap.end() )//find it!
                                                    relDocs = queryRelDocsMap[q->id()];
                                                else
                                                {
                                                    cerr<<"*******relSize**********\n";
                                                    continue;
                                                }

                                                //for(int docID = 1 ; docID < ind->docCount() ; docID++){ //compute for all doc
                                                vector <int> docids = queryDocList(ind,((TextQueryRep *)(qr)));


                                                for(int i = 0 ; i<docids.size(); i++) //compute for docs which have queryTerm
                                                {
                                                    //cout<<"relsize: "<<results.size()<<" i : "<<i<<endl<<endl;
                                                    int docID = docids[i];
                                                    //if (docID != ind->document("afp.com-20040109T173702Z-TX-SGE-UQQ37.xml"))
                                                    //	continue;
                                                    //cout<<"docid: "<< ind->document(docID)<<endl;
                                                    float sim = myMethod->computeProfDocSim(((TextQueryRep *)(qr)) ,docID, relJudgDocs , nonRelJudgDocs , newNonRel,newRel);

                                                    //cout<<sim<<endl;
                                                    if(sim >=  myMethod->getThreshold() )
                                                    {

                                                        numberOfNotShownDocs=0;

                                                        bool isRel = false;
                                                        for(int i = 0 ; i < relDocs.size() ; i++)
                                                        {
                                                            if(relDocs[i] == ind->document(docID) )
                                                            {
                                                                //myMethod->setFlags(true);

                                                                isRel = true;
                                                                newNonRel = false;
                                                                newRel = true;

                                                                relJudgDocs.push_back(docID);

                                                                relSumScores+=sim;

                                                                //global_rel_ret++;
                                                                //global_ret++;

                                                                break;
                                                            }
                                                        }
                                                        if(!isRel)
                                                        {
                                                           // myMethod->setFlags(false);

                                                            nonRelJudgDocs.push_back(docID);
                                                            //if(nonRelJudgDocs.size()%10 == 1)
                                                            newNonRel = true;
                                                            newRel = false;
                                                            //else
                                                            //	newNonRel = false;

                                                            nonRelSumScores+=sim;
                                                            numberOfShownNonRelDocs++;
                                                        }
                                                        results.PushValue(docID , sim);

                                                        if(results.size() > 200)
                                                        {
                                                            cout<<"BREAKKKKKKKKKK\n";
                                                            break;//user gived up
                                                        }


#if 0//FBMODE

                                                        myMethod->updateProfile(*((TextQueryRep *)(qr)),relJudgDocs , nonRelJudgDocs );
                                                        /*if (results.size() %20 == 0 && feedbackMode > 0)
                    {
                        //cout<<"Updating profile. Result size: "<<results.size()<<endl;
                        myMethod->updateProfile(*((TextQueryRep *)(qr)),relJudgDocs , nonRelJudgDocs );
                    }*/
#endif
#if UPDTHRMODE == 1
                                                        if(!isRel)
                                                            if( numberOfShownNonRelDocs == numOfShownNonRel )
                                                            {
                                                                myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,0,relSumScores,nonRelSumScores);//inc thr
                                                                numberOfShownNonRelDocs =0;
                                                            }
#endif

#if UPDTHRMODE == 2
                                                        if(!isRel)//FIXME!!!
                                                            myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,0,relSumScores,nonRelSumScores);//inc thr
#endif


                                                    }
                                                    else
                                                    {
                                                        newNonRel = false;//?????????????????
                                                        newRel = false;//?????????????????
                                                        numberOfNotShownDocs++;
                                                    }
#if UPDTHRMODE == 1
                                                    if(numberOfNotShownDocs == numOfnotShownDoc)//not show anything after |numOfnotShownDoc| docs! -->dec(thr)
                                                    {
                                                        myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,1,relSumScores,nonRelSumScores);//dec thr
                                                        numberOfNotShownDocs = 0;
                                                    }
#endif

#if UPDTHRMODE == 2
                                                    if(numberOfNotShownDocs==100)//FIXME!!!!
                                                        myMethod->updateThreshold(*((TextQueryRep *)(qr)), relJudgDocs , nonRelJudgDocs ,1,relSumScores,nonRelSumScores);//dec thr
#endif

                                                }//endfor docs

                                                results.Sort();
                                                resultFile.writeResults(q->id() ,&results,results.size());
                                                relRetCounter += relJudgDocs.size();
                                                retCounter += results.size();
                                                relCounter += relDocs.size();

                                                //global_all_rels += relCounter;
                                                //global_ret += retCounter;
                                                //global_rel_ret += relRetCounter;

                                                if(results.size() != 0)
                                                {
                                                    queriesPrecision.push_back((double)relJudgDocs.size() / results.size());
                                                    queriesRecall.push_back((double)relJudgDocs.size() / relDocs.size() );
                                                }else // have no suggestion for this query
                                                {
                                                    queriesPrecision.push_back(0.0);
                                                    queriesRecall.push_back(0.0);
                                                }


                                                //break;
                                                delete q;
                                                delete qr;

                                            }//end queries


                                            double avgPrec = 0.0 , avgRecall = 0.0;
                                            for(int i = 0 ; i < queriesPrecision.size() ; i++)
                                            {
                                                avgPrec+=queriesPrecision[i];
                                                avgRecall+= queriesRecall[i];
                                                out<<"Prec["<<i<<"] = "<<queriesPrecision[i]<<"\tRecall["<<i<<"] = "<<queriesRecall[i]<<endl;
                                            }
                                            avgPrec/=queriesPrecision.size();
                                            avgRecall/=queriesRecall.size();

#if UPDTHRMODE == 1
                                            out<<"C1: "<< c1<<"\nC2: "<<c2<<endl;
                                            out<<"numOfShownNonRel: "<<numOfShownNonRel<<"\nnumOfnotShownDoc: "<<numOfnotShownDoc<<endl;
#endif
                                            out<<"Avg Precision: "<<avgPrec<<endl;
                                            out<<"Avg Recall: "<<avgRecall<<endl;
                                            out<<"F-measure: "<<(2*avgPrec*avgRecall)/(avgPrec+avgRecall)<<endl<<endl;

                                            double pp = relRetCounter/retCounter;
                                            double dd = relRetCounter/relCounter;
                                            out<<"rel_ret: "<<relRetCounter<<" ret: "<<retCounter<<" rels: "<<relCounter<<endl;
                                            out<<"old_Avg Precision: "<<pp<<endl;
                                            out<<"old_Avg Recall: "<<dd<<endl;
                                            out<<"old_F-measure: "<<(2*pp*dd)/(pp+dd)<<endl<<endl;



                                            //break;
                                            //if(feedbackMode == 0)//no fb
                                            //    break;
                                            //if(numberOfQueries==2)//????????????????????????????????????????????????????????
                                            //    break;

#if RETMODE == 1 && FBMODE == 1
                                        }
#endif

#if UPDTHRMODE == 1
                                    }//end numOfnotShownDoc for
                                }//end numOfShownNonRel for
                            }//end c1 for
                        }//end c2 for
#endif
#if UPDTHRMODE == 2
                    }//end alpha for
#endif

                    //#if !FBMODE && !UPDTHRMODE
                    // }
                }
            }
        }
    }
    //#endif
    delete qs;
    delete myMethod;
}
void loadJudgment()
{
    int judg,temp;
    string docName,id;

    ifstream infile;
    infile.open (judgmentPath.c_str());

    string line;
    while (getline(infile,line))
    {
        stringstream ss(line);
        if(DATASET == 0)//infile
        {
            ss >> id >> temp >> docName >> judg;
            if(judg == 1)
            {
                queryRelDocsMap[id].push_back(docName);
                //cerr<<id<<" "<<docName<<endl;
            }
        }else if(DATASET == 1)//ohsu
        {
            ss >> id >> docName;
            queryRelDocsMap[id].push_back(docName);
        }
    }
    infile.close();


    //110,134,147 rel nadaran
    /*map<string , vector<string> >::iterator it;
    for(it = queryRelDocsMap.begin();it!= queryRelDocsMap.end() ; ++it)
        cerr<<it->first<<endl;*/

}
vector<int> queryDocList(Index* ind,TextQueryRep *textQR)
{
    vector<int> docids;
    set<int> docset;
    textQR->startIteration();
    while (textQR->hasMore()) {
        QueryTerm *qTerm = textQR->nextTerm();
        if(qTerm->id()==0){
            cerr<<"**********"<<endl;
            continue;
        }
        DocInfoList *dList = ind->docInfoList(qTerm->id());

        dList->startIteration();
        while (dList->hasMore()) {
            DocInfo *info = dList->nextEntry();
            DOCID_T id = info->docID();
            docset.insert(id);
        }
        delete dList;
        delete qTerm;
    }
    docids.assign(docset.begin(),docset.end());
    return docids;
}

void MonoKLModel(Index* ind){
    DocStream *qs = new BasicDocStream(queryPath.c_str()); // Your own path to topics
    ArrayAccumulator accumulator(ind->docCount());
    RetMethod *myMethod = new RetMethod(*ind,"",accumulator);
    IndexedRealVector results;
    qs->startDocIteration();
    TextQuery *q;

    ofstream result("res.my_ret_method");
    ResultFile resultFile(1);
    resultFile.openForWrite(result,*ind);
    PseudoFBDocs *fbDocs;
    while(qs->hasMore()){
        Document* d = qs->nextDoc();
        //d->startTermIteration(); // It is how to iterate over query terms
        //ofstream out ("QID.txt");
        //while(d->hasMore()){
        //	const Term* t = d->nextTerm();
        //	const char* q = t->spelling();
        //	int q_id = ind->term(q);
        //	out<<q_id<<endl;
        //}
        //out.close();
        q = new TextQuery(*d);
        QueryRep *qr = myMethod->computeQueryRep(*q);
        myMethod->scoreCollection(*qr,results);
        results.Sort();
        //fbDocs= new PseudoFBDocs(results,30,false);
        //myMethod->updateQuery(*qr,*fbDocs);
        //myMethod->scoreCollection(*qr,results);
        //results.Sort();
        resultFile.writeResults(q->id(),&results,results.size());
        cerr<<"qid "<<q->id()<<endl;
        break;
    }
}


#if 0
#include "pugixml.hpp"
using namespace pugi;
void ParseQuery(){
    ofstream out("topics.txt");
    xml_document doc;
    xml_parse_result result = doc.load_file("/home/hossein/Desktop/lemur/DataSets/Infile/Data/q_en.xml");// Your own path to original format of queries
    xml_node topics = doc.child("topics");
    for (xml_node_iterator topic = topics.begin(); topic != topics.end(); topic++){
        xml_node id = topic->child("identifier");
        xml_node title = topic->child("title");
        xml_node desc = topic->child("description");
        xml_node nar = topic->child("narrative");
        out<<"<DOC>"<<endl;
        out<<"<DOCNO>"<<id.first_child().value()<<"</DOCNO>"<<endl;
        out<<"<TEXT>"<<endl;
        out<<title.first_child().value()<<endl;
        out<<"</TEXT>"<<endl;
        out<<"</DOC>"<<endl;

    }
    printf("Query Parsed.\n");
}
#endif
#endif

