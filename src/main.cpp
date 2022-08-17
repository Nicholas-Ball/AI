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
    test.Generate({10,2},10,{ActivationFunctions::Sigmoid,ActivationFunctions::Sigmoid,ActivationFunctions::Sigmoid,ActivationFunctions::Sigmoid,ActivationFunctions::Sigmoid,ActivationFunctions::LRelu});
  }

  //make a new game
  Game_10s game;

  /**/
  //train neuron with data
  Backpropagation bp(&test,10000);

  bool switc = false;
  //make data
  for(int i = 0; i != 100;i++){
    //generate new game
    while(switc != game.Signal)
      game.Generate();

    //add as input data
    //bp.Add({1,2,3,4,5,6,7,8,9,10},{1,0});
    //bp.Add({10,9,8,7,6,5,4,3,2,1},{0,1});
    bp.Add(game.Nums,{(double)(int)game.Signal,(double)(int)!game.Signal});

    switc = !switc;

  }
  bp.LearningRate = 0.001;
  bp.Stop = 0.1;

  bp.Batches = 100;

  bp.Verbose = true;



  //run backprogation
  bp.Run();/**/
  while(!game.Signal)
    game.Generate();

  //std::vector<double> data = {10,9,8,7,6,5,4,3,2,1};
  std::vector<double> data = game.Nums;
  test.Run(&data);
  std::cout<<"Game: "<<data[0]<<" No Game: "<<data[1]<<std::endl;
  std::cout<<"Signal: "<<game.Signal<<std::endl;

  test.Save("brain.json");




  return 0;
}
