// This file holds all math functions and related functions for AI
#include <math.h>

#pragma once


class AIMath{
  public:
    //math functions for ai
    class Functions{
      public:
        // sigmoid(x) = 1/(1 +e^-x)
        //preform sigmoid function on A
        static inline void Sigmoid(double* a){
          //std::cout<<"Sigmoid in: "<<*a<<std::endl;
          double inp = (*a);
          (*a) = (double)(1 / (1 + exp(-inp)));
          //std::cout<<"Sigmoid out: "<<*a<<std::endl;

        }

        // lrelu(x) = {if x < 0 return 0.1*x else return x}
        //preform Leaky relu function on A
        static inline void Lrelu(double* a){
          //std::cout<<"lrelu: "<<*a<<std::endl;
          //(*a) = ((*a)*((*a) > 0)) + ((*a)*0.1*((*a) <= 0));
          if((*a) <= 0){
            (*a) *= 0.1;
          }
        }

        //relu(x) = {if x < o return 0 else return x}
        //preform normal relu
        static inline void Relu(double* a){
          (*a) = (*a) * ((*a) > 0);
        }

        static inline void Tanh(double* a){
          double inp = (*a);
          (*a) = (double)(2 / (1 + exp(-2*inp)))-1;
        }


        //cost(p,d) = (p-d)^2
        //cost of two numbers. The number is returned via depicted value
        static inline void Cost(double* depicted,double* predicted){
          (*depicted) = powf(*predicted-*depicted,2);
        }
    };

    //prime math functions
    class PrimeFunctions{
        public:
          // prime_sigmoid(x) = sigmoid(x)(1 - sigmoid(x))
          //preform prime sigmoid function on A
          static inline void Sigmoid(double* a){
            double inp = (*a);
            //preform sigmoid on a
            Functions::Sigmoid(&inp);
            (*a) = inp * (1 - inp);
          }

          //prime_leaky = {if x < 0 return 0.1 else return 1}
          //preform prime relu function on A
          static inline void Lrelu(double* a){
            if((*a) <= 0){
              (*a) = 0.1;
            }else{
              (*a) = 1;
            }
          }

          //prime_relu(x) = {if x < 0 return 0 else return 1}
          //preform normal prime relu
          static inline void Relu(double* a){
            (*a) = ((*a) > 0);
          }

          static inline void Tanh(double* a){
            double inp = (*a);
            Functions::Tanh(&inp);
            (*a) = 1-pow(inp,2);
          }


          static inline void Cost(double* depicted,double* predicted){
            (*depicted) = 2*(*predicted - *depicted);
          }
    };

};
