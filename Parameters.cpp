
#include <fstream>
#include <iostream>
#include <sstream>


using namespace std;

template <typename T>
string numToStrHM(T number)
{
    ostringstream s;
    s << number;
    return s.str();
}

double startThresholdHM , endThresholdHM , intervalThresholdHM ,negGenMUHM;

int RSMethodHM; // 0--> LM , 1--> RecSys
int negGenModeHM;//0 --> coll , 1--> nonRel


int WHO;// 0--> server , 1-->Mozhdeh, 2-->AP, other-->Hossein
string outputFileNameHM;
string resultFileNameHM;

void readParams(string paramFileName)
{
    cout<<"********************\n";
    ifstream in(paramFileName.c_str());
    if(in.is_open())
    {
        string temp;
        in>>temp;
        in>>WHO>>RSMethodHM>>negGenModeHM>>negGenMUHM>>startThresholdHM>>endThresholdHM>>intervalThresholdHM;
        cout<<WHO<<" "<<RSMethodHM<<" "<<negGenModeHM<<" "<<negGenMUHM<<" "<<startThresholdHM<<" "<<endThresholdHM<<" "<<intervalThresholdHM<<endl;
    }
    if(RSMethodHM==1)
    {
        if(negGenModeHM == 0)
        {
            outputFileNameHM = "out/RecSys_NegGenColl_";
            resultFileNameHM ="res/RecSys_NegGenColl_";
        }else if(negGenModeHM == 1)
        {
            outputFileNameHM += "out/RecSys_NegGenNonRel_";
            resultFileNameHM =+ "res/NegGenNonRel_";
        }
	outputFileNameHM+=numToStrHM(negGenMUHM)+"_";
	resultFileNameHM+=numToStrHM(negGenMUHM)+"_";
    }else if (RSMethodHM == 0)
    {
        outputFileNameHM += "out/LM_";
        resultFileNameHM += "res/LM_";
    }
    outputFileNameHM += numToStrHM(startThresholdHM)+":"+numToStrHM(endThresholdHM);

}
