//this file holds data about a Fully Connected Neural Network (fcnn)
#include "neuron.h"
#include "fstream"

#pragma once

class FCNN{
  public:
    //the network matrix storage
    std::vector<std::vector<Neuron*>> Network;

    double TError;

    bool Softmax;

    //basic constructor
    FCNN(){
        //get unix time as bases for Random
        srand(time(0));

        //set total error
        this->TError = 0;

        //should this network return a softmax function
        this->Softmax = false;
    }

    //generate new network with given paramaters
    void Generate(std::vector<int> matrix,int num_inputs,std::vector<ActivationFunctions> functions){
      //loop through matrix
      for(int l = 0; l != matrix.size();l++){
        //bias for this layer
        double lbias = Neuron::Random();

        //layer array
        std::vector<Neuron*> layer;

        //loop and make neurons
        for(int n = 0; n < matrix[l];n++){
          //make neuron
          Neuron* temp = new Neuron(lbias,functions[l]);

          //make weights
          //if on layer 0, just add weights for the number of inputs
          if(l == 0){
            //make weights by num inputs
            for(int w = 0; w < num_inputs;w++)
              (*temp).AddWeight();

          }else{
            //make weights by num of neurons in previous layer
            for(int w = 0; w < matrix[l-1];w++)
              (*temp).Connect(Network[l-1][w]);

          }
          //add temp to layer
          layer.push_back(temp);
        }
        //add layer to network
        this->Network.push_back(layer);
      }
    }

    //generate new network with given paramaters but with all the same activation
    void Generate(std::vector<int> matrix,int num_inputs,ActivationFunctions function){
      //make array of all the same function
      std::vector<ActivationFunctions> arr;
      for(int l = 0; l != matrix.size();l++)
        arr.push_back(function);

      //generate
      this->Generate(matrix,num_inputs, arr);
    }

    //run network and return via input
    void Run(std::vector<double>* input){
      std::vector<double> out;
      //run and collect
      for(int i = 0; this->Network.size() > 0 && i != this->Network[this->Network.size()-1].size();i++){
        this->Network[this->Network.size()-1][i]->Run(input);

        out.push_back(this->Network[this->Network.size()-1][i]->Output);
      }
      (*input) = out;
    }

    //reset network
    void Reset(){
      //reset each neuron
      for(int i = 0; this->Network.size() > 0 && i != this->Network[this->Network.size()-1].size();i++)
        this->Network[this->Network.size()-1][i]->Reset();
    }

    //calculate error of this network
    void CalculateError(std::vector<double>* expected){
      //loop through layers
      for(int l = this->Network.size()-1; l != -1;l--){
        //loop through neurons
        for(int n = 0; n != this->Network[l].size();n++){
          //if on last layer, use calculated error
          if(l == this->Network.size()-1){
            //calculate errors
            this->Network[l][n]->CalcError((*expected)[n]);


            //add to total error
            this->TError += this->Network[l][n]->Hold;

          }else{
            //else just calculate without input
            this->Network[l][n]->CalcError();
          }
        }
      }
    }

    //adjust weights in network
    void Adjust(double learning_rate,int num_passes){
      for(int i = 0; this->Network.size() > 0 && i != this->Network[this->Network.size()-1].size();i++){
        //adjust neurons
        this->Network[this->Network.size()-1][i]->Adjust(learning_rate,num_passes);
      }

      //reset network
      this->Reset();
      this->TError = 0;
    }


    //save data about this network to a file
    void Save(std::string filename){
      //make data
      nlohmann::json data;

      //save all neurons
      for(int l = 0; l != this->Network.size();l++){
        for(int n = 0; n != this->Network[l].size();n++){
          data[l][n] = this->Network[l][n]->Save();
        }
      }

      std::fstream file;
      file.open(filename,std::fstream::out|std::ios::trunc);
      file << data;
      file.close();

    }

    //load data about this network from file
    void Load(std::string filename){

      //load data
      nlohmann::json data = nlohmann::json::parse(std::ifstream(filename));


      //blank old data
      this->Network = {};

      //load all neurons
      for(int l = 0; l != data.size();l++){
        this->Network.push_back({});
        for(int n = 0; n != data[l].size();n++){
          this->Network[l].push_back(new Neuron(1,ActivationFunctions::Sigmoid));
          this->Network[l][n]->Load(data[l][n]);

          if(l != 0){
            //connect neurons to previous layer
            for(int i = 0; i != data[l-1].size();i++){
              this->Network[l][n]->ConnectWO(this->Network[l-1][i]);
            }
          }
        }
      }
    }
};
