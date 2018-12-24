#include<time.h>
#include "algo.h"

int main(int argc, char *argv[])
{
    /* 统计算法用时 */
    clock_t start,finish;
    double totaltime;
    /* 起始时间 */
    start=clock();

    testSingleSeed("srcimage/"+ _myfilename);

    cout<<"Finished!"<<endl;
    /* 终止时间 */
    finish=clock();
    /* 总时间 */
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    cout<<"time: "<<totaltime<<" s."<<endl;

    return 0;
}
