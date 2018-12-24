#include<time.h>
#include "algo.h"

int main(int argc, char *argv[])
{
    clock_t start,finish;
    double totaltime;
    start=clock();

    testSeed("srcimage/"+ _myfilename);

    cout<<"Finished!"<<endl;
    finish=clock();
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"time: "<<totaltime<<" s."<<endl;

    return 0;
}
