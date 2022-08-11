#include <iostream>
#include <fstream>
#include "10s.hpp"
#include "include/fcnn.h"
#include "include/backpropagation.h"

int main(){

  FCNN test;


  try{
    test.Load("brain.json");
  }catch(...){
    test.Generate({10,1},10,{ActivationFunctions::LRelu,ActivationFunctions::Sigmoid});
  }

  //make a new game
  Game_10s game;

  /**/
  //train neuron with data
  Backpropagation bp(&test,10000);


  //make data
  for(int i = 0; i != 100;i++){
    //generate new game
    //game.Generate();

    //add as input data
    bp.Add({1,2,3,4,5,6,7,8,9,10},{1});
    bp.Add({10,9,8,7,6,5,4,3,2,1},{0});

  }
  bp.LearningRate = 0.1;
  bp.Stop = 1;

  bp.Batches = 20;



  //run backprogation
  //bp.Run();/**/

  //std::vector<double> data = {10,9,8,7,6,5,4,3,2,1};
  std::vector<double> data = {10,9,8,7,6,5,4,3,2,1};
  test.Run(&data);
  std::cout<<data[0]<<std::endl;

  test.Save("brain.json");




  return 0;
}
