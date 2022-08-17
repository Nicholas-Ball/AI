
//this file holds d
#include "math.h"
#include <iostream>
#include <vector>
#include <time.h>
#include "json.hpp"

#pragma once

//type that takes a number pointer as an input
typedef void (*FuncMet) (double*);

//array of activation functions
const FuncMet ActivationMethods[] = {AIMath::Functions::Sigmoid,AIMath::Functions::Lrelu,AIMath::Functions::Relu,AIMath::Functions::Tanh};

//array of prime activation functions
const FuncMet PrimeActivationMethods[] = {AIMath::PrimeFunctions::Sigmoid,AIMath::PrimeFunctions::Lrelu,AIMath::PrimeFunctions::Relu,AIMath::PrimeFunctions::Tanh};

//types of activation functions
enum ActivationFunctions {Sigmoid,LRelu,Relu,Tanh};

class Neuron{
  private:
  public:
    //numbers related to this neuron
    double Bias,PreActivation,Output,Error,Hold;

    //has this neuron been activated?
    bool Active;

    //array of connected neurons (neurons that send inputs to this one)
    std::vector<Neuron*> Connected;

    //array of data that is important to this neuron
    std::vector<double> Weights,Delta,Inputs;

    //function type
    ActivationFunctions Function;

      //make random number
      static double Random(){
        //make random number between -1 to 1
        return ((double)(rand() % 1000000) / 100000000 );
      }


    //create neuron with only bias and type
    Neuron(double bias,ActivationFunctions af){
      //set values
      this->Bias = bias;
      this->Function = af;
      this->Active = false;
      this->Hold = 0;
      this->Output = 0;
      this->PreActivation = 0;
      this->Error = 0;

      //make delta for bias
      this->Delta.push_back(0);
    }

    //add weight to neuron
    void AddWeight(){
      //add weight
      this->Weights.push_back(Random());

      //add delta allocation
      this->Delta.push_back(0);

      //add new input allocation
      this->Inputs.push_back(0);
    }

    //connect neuron to this neuron
    void Connect(Neuron* toConnect){
      //add connection
      this->Connected.push_back(toConnect);

      //add new weight
      AddWeight();
    }

    //connect neuron to this neuron w/o adding weights
    void ConnectWO(Neuron* toConnect){
      //add connection
      this->Connected.push_back(toConnect);
    }

    //save data about this neuron
    nlohmann::json Save(){
      nlohmann::json data;
      data["Bias"] = this->Bias;
      data["Function"] = (int)this->Function;
      data["Weights"] = this->Weights;

      return data;
    }

    //load data about this neuron
    void Load(nlohmann::json data){
      //blank old data
      this->Delta = {};
      this->Bias = 1;
      this->Active = false;
      this->Hold = 0;
      this->Output = 0;
      this->PreActivation = 0;
      this->Error = 0;

      //make delta for bias
      this->Delta.push_back(0);

      this->Bias = data["Bias"].get<double>();
      this->Function = static_cast<ActivationFunctions>(data["Function"].get<int>());
      this->Weights = data["Weights"].get<std::vector<double>>();

      //load data about connections
      for(int i = 0; i != this->Weights.size();i++){
        //add delta allocation
        this->Delta.push_back(0);

        //add new input allocation
        this->Inputs.push_back(0);
      }

    }

    //run neuron with inputs (if this has connected neuron run those else use input data)
    void Run(std::vector<double> *input){
      //if this neuron is already active, just return
      if(this->Active){
        return;

        //if has connected neurons, run those
      }else if(this->Connected.size() > 0){

        //loop through connected neuron
        for(int i = 0; i != this->Connected.size();i++){
          //run neuron
          (*this->Connected[i]).Run(input);

          //add to preactivation
          this->PreActivation += (*this->Connected[i]).Output*this->Weights[i];

          //collect input
          this->Inputs[i] = (*this->Connected[i]).Output;
        }

        //if there are no connected Neurons, use input data and be sure
      }else{
        //go through each input
        for(int i = 0; i != (*input).size();i++){
          this->PreActivation += (*input)[i] * this->Weights[i];

          //collect input
          this->Inputs[i] = (*input)[i];
        }
      }


      //add bias
      this->PreActivation += this->Bias;

      //send preactivation to output for proccessing
      this->Output = this->PreActivation;

      //run activation and send to input
      ActivationMethods[(int)this->Function](&this->Output);

      //std::cout<<"Out: "<<this->Output<<std::endl;

      //activate neuron
      this->Active = true;
    }

    //calculate error without input
    void CalcError(){
      //make deltas
      //-----
      //loop through weights
      for(int i = 0; i != this->Weights.size();i++){
        double funcPrime = this->PreActivation;

        PrimeActivationMethods[(int)this->Function](&funcPrime);


        //weight_delta = pre_output*func_prime(pre_act)*dCost
        this->Delta[i] += (this->Inputs[i])*(funcPrime*this->Error);

        //std::cout<<" Out: "<<Connected[i]->Output<<" Prime*error: "<<funcPrime*this->Error<<" Delta: "<<this->Connected[i]->Output*(funcPrime*this->Error)<<std::endl;

        if(this->Connected.size() > i){
          //d_pre_cost += weight * func_prime(pre_act) * dCost
          this->Connected[i]->Error += this->Weights[i] * funcPrime*this->Error;
        }
      }

      this->Error = 0;
    }

    //Calculate error of this neuron and any connecting ones
    void CalcError(double expected){
      //calculate cost
      AIMath::Functions::Cost(&expected,&this->Output);
      this->Hold = expected;

      AIMath::PrimeFunctions::Cost(&expected,&this->Output);
      this->Error = expected;

      this->CalcError();
    }

    //use deltas and adjust weights and bias
    void Adjust(double learning_rate,int num_passes){
      if(this->Active)
        return;
      //if this has connected neurons, adjust those first
      for(int i = 0; i != this->Connected.size();i++){
        //reset neuron
        (*this->Connected[i]).Adjust(learning_rate,num_passes);
      }

      //loop through weights
      for(int w = 0; w != this->Weights.size();w++){
        //weight = weight - (learning_rate*delta)
        this->Weights[w] -= learning_rate*(this->Delta[w]/num_passes);

        //std::cout<<"Delta: "<<this->Delta[w]<<std::endl;

        //reset
        this->Delta[w] = 0;
      }

      //bias = bias - (learning_rate*delta)
      this->Bias -= learning_rate*(this->Delta[this->Delta.size()-1]/num_passes);

      //reset
      this->Hold = 0;
      this->Delta[this->Delta.size()-1] = 0;
      this->Error = 0;
      this->PreActivation = 0;
      this->Active = true;
      this->Output = 0;
    }


    //reset this neuron and any connected ones as well
    void Reset(){
      //return if already set to false
      if(!this->Active)
        return;

      //if this has connected neurons, reset those first
      for(int i = 0; i != this->Connected.size();i++){
        //reset neuron
        (*this->Connected[i]).Reset();
      }
      this->Active = false;
    }
};
