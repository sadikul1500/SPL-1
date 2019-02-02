#include<iostream>
#include<cstdlib>
#include<ctime>
#include<cmath>

using namespace std;

void initializeWeights(double* weight,int column)
{
     srand(time(NULL));

     for(int i=0;i<column+1;i++) weight[i]=-1+2*((double) rand())/RAND_MAX;


}

double normalizeValue(double x)
{
    return 1/(1+pow(2.71828,(-1*x)));
}

double avgSqrError(double* pcOutput,int * output,int row)
{
    double error=0;

    for(int i=0;i<row;i++)
    {
        double temp=pcOutput[i]-output[i];
        error+=temp*temp;
    }

    error/=row;

    return error;

}

void training(int** input,int* output,double* weight,int row,int col)
{
    double maximumError=.00001;
    double pcOutput[row];
    double error=10;
   // int  maximumIteration=10000000;


    for(int i=0;  i<1000000 && error>=maximumError;  i++ )
    {
        for(int j=0;j<row;j++)
        {
            double predictedOutput=weight[col];

            for(int i=0;i<col;i++)
            {
                predictedOutput+=weight[i]*input[j][i];
            }
            predictedOutput=normalizeValue(predictedOutput);

            pcOutput[j]=predictedOutput;

            for(int k=0; k<col; k++)  weight[k]+=.001*(output[j]-pcOutput[j])*pcOutput[j]*(1-pcOutput[j])*input[j][k];

            weight[col]+=.001*(output[j]-pcOutput[j])*pcOutput[j]*(1-pcOutput[j])*1;

            //weight[0]+=.001*(output[j]-pcOutput[j])*pcOutput[j]*(1-pcOutput[j])*input[j][0];
            //weight[1]+=.001*(output[j]-pcOutput[j])*pcOutput[j]*(1-pcOutput[j])*input[j][1];
          //  weight[2]+=.001*(output[j]-pcOutput[j])*pcOutput[j]*(1-pcOutput[j])*1;
        }

        error=avgSqrError(pcOutput,output,row);

    }
}

double getOutput(int* input,double *weight,int col)
{
   // double output=weight[0]*x+weight[1]*y+weight[2];
   double output=weight[col];

   for(int i=0;i<col;i++)
   {
       output+=weight[i]*input[i];
   }

    output=1/(1+exp(-1*output));

    return output;
}


int main()
{
    int row,column;

    cout<<"give number of data set : "<<endl;
    cin>>row;

    cout<<"give number of input in each set : "<<endl;
    cin>>column;

    int m=column+1;
    int **inputSet=new int* [row];

    for(int i=0;i<4;i++)
    {
        inputSet[i]=new int[column];
    }

    cout<<"give input set "<<endl;

    for(int i=0;i<row;i++)
    {
        for(int j=0;j<column;j++)
            cin>>inputSet[i][j];
    }

    int *andOutput=new int[row];

    cout<<"give and output "<<endl;

    for(int i=0;i<row;i++) cin>>andOutput[i];

    cout<<endl<<"give or gate output "<<endl;

    int *orOutput=new int[row];
    for(int j=0;j<row;j++) cin>>orOutput[j];


    double *andWeight=new double[m];
    double *orWeight=new double[m];

    initializeWeights(andWeight,column);
    initializeWeights(orWeight,column);

    training(inputSet,andOutput,andWeight,row,column);
    training(inputSet,orOutput,orWeight,row,column);

    while(1)
    {
        cout<< "give a  input to get test output : ";
        int * input=new int[column];

        for(int i=0;i<column;i++) cin>>input[i];

        cout<<"and operation result : " <<getOutput(input,andWeight,column)<<endl;
        cout<<"or operation result : "<<getOutput(input,orWeight,column);

        cout<<endl<<"press q to exit or any button to continue : ";

        char c;

        cin>>c;

        if(c=='q') break;
    }

    return 0;


}
