//this file holds data protaining to backprogation of networks
#include "fcnn.h"

bool _END_ = false;

class Backpropagation{
  private:

    //run backpropagation on FCNN
    void bpFCNN(){
      this->Network->TError = 0;
      int batch_size = this->Inputs.size()/this->Batches;
      //loop through batches
      for(int b = 0; b != this->Batches;b++){
        //loop though training data
        for(int i = batch_size*b; i != batch_size+(batch_size*b);i++){
          //copy training data
          std::vector<double> inp = this->Inputs[i];
          std::vector<double> exp = this->Expected[i];


          //run
          this->Network->Run(&inp);



          //reset
          this->Network->Reset();


          //calculatue error
          this->Network->CalculateError(&exp);


        }

        //print out error if on verbose
        if(this->Verbose)std::cout<<" Batch "<<b<<": "<<this->Network->TError<<std::endl;



        if(this->Network->TError <= this->Stop && this->Network->TError != 0){
          _END_ = true;
          return;
        }

        //adjust
        Network->Adjust(this->LearningRate, batch_size);
    }
  }

  public:
    //FCNN network being used
    FCNN* Network;

    //array of input data
    std::vector<std::vector<double>> Inputs;

    //array of expecteded data
    std::vector<std::vector<double>> Expected;

    //number of epoches
    int Epoches;

    //number to stop all training at
    double Stop;

    //learning rate of ai
    double LearningRate;

    //Print out current error of network on each epoch
    bool Verbose;

    //num of batches per a data set
    int Batches;

    //prep FCNN for backpropagation
    Backpropagation(FCNN* network,int epoches = 1000){
      this->Network = network;
      this->Epoches = epoches;
      this->Stop = -1;
      this->LearningRate = 0.1;
      this->Verbose = true;
      this->Batches = 1;
    }

    //add training data
    void Add(std::vector<double> inp,std::vector<double> exp){
      this->Inputs.push_back(inp);
      this->Expected.push_back(exp);
    }

    //run backpropagation algorithm
    void Run(){
      if(this->Network != NULL ){
        //run epoch batch x amount of times
        for(int e = 0; e != this->Epoches && !_END_;e++){
          if(this->Verbose) std::cout<<"Epoch "<<e<<" Error: "<<std::endl;
          bpFCNN();
        }
        _END_ = false;
      }
    }

};
