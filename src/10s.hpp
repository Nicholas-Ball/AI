//10s game
//10 numbers are randomly generated. if 3 or more "7"s are gnerated this a positive signal else it's not

#include <vector>

class Game_10s{
  private:
    //make random number
  double Random(){
    return (rand() % 10);
  }

  //check if list has 3 "7"s
  void Check(){
    //sevens counter
    int sevens = 0;

    //count the number of sevens
    for(int i = 0; i != 10;i++)
      sevens += (this->Nums[i] == 7);


    //set Signal if this has 3 or more sevens
    this->Signal = (sevens >= 3);
  }


  public:
    std::vector<double> Nums;
    bool Signal;

    //construct game
    Game_10s(){
      srand(time(0));
      this->Nums = {};
      this->Signal = false;
      this->Generate();
    }

    //generate nums
    void Generate(){
      //blank nums
      this->Nums = {};

      //make random numbers
      for(int i = 0; i != 10; i++)
        this->Nums.push_back(Random());

      //check these numbers
      this->Check();
    }
};
