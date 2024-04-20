#ifndef INIT_
#define INIT_

#include "tensor_load.hpp"

class Glorot{
    public:
        Tensor initialize(double n_rows,double n_cols){
            double a = 2 / (n_rows + n_cols);
            double scale = std::sqrt(a);
            Tensor param = xt::random::randn<double>({n_rows,n_cols}) * scale;
            return param;
        }
};

class He{
    Tensor initialize(double n_rows, double n_cols){
        double  bound = std::sqrt(6/n_rows);
        Tensor rand_values = xt::random::rand<double>({n_rows,n_cols});
        Tensor scaled_values = rand_values * 2 * bound;
        Tensor shifted_values = scaled_values - bound;
        return shifted_values;
    }
};

class LSUV{
    public:
        double input_stddev;
        LSUV(double scale){
            input_stddev = scale;
        }

        Tensor initialize(double n_rows,double n_cols){
            double s = 1;
            Tensor rand_t = xt::random::rand<double>({n_rows,n_cols});
            Tensor weights = xt::random::randn<double>({n_rows,n_cols}) * input_stddev;
            Tensor input_values = xt::random::randn<double>({s,n_rows}) * input_stddev;
            Tensor output_values = xt::sum(input_values * weights,{0});
            Tensor std_dev = std::sqrt(xt::sum(xt::square(output_values))[0] / n_cols);
            return weights / std_dev;
        }

};

#endif