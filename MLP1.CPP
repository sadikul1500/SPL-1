#include<iostream>
#include<cmath>
#include<cstdlib>
#include<ctime>

using namespace std;

void initializeWeights(double* weights,int column)
{
    srand(time(NULL));

    for(int i=0;i<column+1;i++) weights[i]=-1+2*((double) rand())/RAND_MAX;


}

double normalizeValue(double x)
{
    return 1/(1+ exp(-x)) ;
}


double calculateAvgError(double* nn,int* pc,int row)
{
    double error=0;

    for(int i=0;i<row;i++)
    {
        double temp=(pc[i]-nn[i])*(pc[i]-nn[i]);
        error+=temp;
    }

    return error/row;
}


void training(int **input,double** weights_1,double *weights_2,int row,int column,double* h ,int* output)
{
    double learningRate=.1;
    double maxError=.00001;
    double error2=10;
    double *pcOutput=new double[row];

    int i=0;

    while(error2>maxError && i++<1000000)
    {
        for(int k=0;k<row;k++)
        {
            pcOutput[k]=weights_2[column+1];

            for(int j=0;j<column+1;j++)
            {
                h[j]=weights_1[j][column];

                for(int m=0;m<column;m++)
                {
                    h[j]+=input[k][m]*weights_1[j][m];
                }
                h[j]=normalizeValue(h[j]);
                pcOutput[k]+=h[j]*weights_2[j];
            }
            pcOutput[k]=normalizeValue(pcOutput[k]);

	    //hidden layer
            for(int p=0;p<column+1;p++)
            {
               // double del=.00001*(output[k]-pcOutput[k])*pcOutput[k]*(1-pcOutput[k])*h[p];
               // error1=del*weights_2[p];


                for(int q=0;q<column;q++)
                {
                    weights_1[p][q]+=.1*h[p]*(1-h[p])*input[k][q]*(output[k]-pcOutput[k])*weights_2[p]*pcOutput[k]*(1-pcOutput[k]);
                }

                weights_1[p][column]+=.1*h[p]*(1-h[p])*(output[k]-pcOutput[k])*weights_2[p]*pcOutput[k]*(1-pcOutput[k]);


            }

            for(int p=0;p<column+1;p++) weights_2[p]+=.1*h[p]*pcOutput[k]*(1-pcOutput[k])*(output[k]-pcOutput[k]);

            weights_2[column+1]+=.1*pcOutput[k]*(1-pcOutput[k])*(output[k]-pcOutput[k]);

        }


        error2=calculateAvgError(pcOutput,output,row);

        //if(i<30 || i>(1000000-50) )  cout<<"error  : "<<error2<<endl;

    }
}

double getOutput(double** weight1,double* weight2,double* h,int* input,int column,int row)
{
    for(int i=0;i<column+1;i++)
    {
        h[i]=weight1[i][column];

        for(int j=0;j<column;j++)
        {
            h[i]+=input[j]*weight1[i][j];
        }

        h[i]=normalizeValue(h[i]);
    }

    double output=weight2[column+1];

    for(int i=0;i<column+1;i++)
    {
       output+=h[i]*weight2[i];
    }

    output=normalizeValue(output);

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

    for(int i=0;i<row;i++)
    {
        inputSet[i]=new int[column];
    }

    cout<<"give input set "<<endl;

    for(int i=0;i<row;i++)
    {
        for(int j=0;j<column;j++)
            cin>>inputSet[i][j];
    }

    int *output=new int[row];

    cout<<"give  output "<<endl;

    for(int i=0;i<row;i++) cin>>output[i];

    double* hiddenLayer=new double[column+1];
    double* weights_2=new  double[column+2];
    double** weights_1=new  double*[column+1];

    for(int i=0;i<column+1;i++)
    {
        weights_1[i]=new double[column+1];
        initializeWeights(weights_1[i],column);
    }

    initializeWeights(weights_2,column+1);
    training(inputSet,weights_1,weights_2,row,column,hiddenLayer,output);


        while(1)
    {
        cout<< "give a  input to get test output : ";
        int * input=new int[column];

        for(int i=0;i<column;i++) cin>>input[i];

        cout<<" operation result : " <<getOutput(weights_1,weights_2,hiddenLayer,input,column,row)<<endl;
        //cout<<"or operation result : "<<getOutput(input,orWeight,column);

        cout<<endl<<"press q to exit or any button to continue : ";

        char c;

        cin>>c;

        if(c=='q') break;
    }
}
