#include "storage.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "utilities.hpp"
#include <memory>
#include <iostream>



int main(){
    Tensor t{{1,2,3,4}};
    t.permute({2,1,0,3});
    std::cout << t;
    return 0;
}
