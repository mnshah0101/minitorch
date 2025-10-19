#include "storage.hpp"
#include "tensor.hpp"
#include "types.hpp"
#include "utilities.hpp"
#include <memory>
#include <iostream>



int main(){
    Tensor t{{1,2,3,4}};
    std::cout << t.dtype;
    return 0;
}
