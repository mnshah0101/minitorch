#pragma once
#include "types.hpp"
#include "storage.hpp"
#include "utilities.hpp"
#include <memory>
#include <sstream>
#include <algorithm>

class Tensor{
    public: 
    Storage* storage;
    Shape shape;
    uint8_t ndim;
    Stride stride;
    Offset offset;
    DType dtype;
    bool is_leaf;
    bool requires_grad;
    Tensor* grad;



    Tensor(Shape shape, DType dtype = float32, Device device = std::make_pair("cpu",0), bool is_leaf = true, bool requires_grad = false){
        this->shape = shape;
        this->dtype = dtype;
        
        stride.resize(shape.size());
        size_t running = 1;
        for(int d= (int) shape.size() - 1; d >= 0; d--){
            stride[d] = running;
            running *= shape[d];
        }
        this->offset = 0;
        this->ndim = shape.size();
        this->is_leaf = is_leaf;
        this->requires_grad = requires_grad;
       
        this->grad = nullptr;

        uint64_t bytes = 1;
        for (auto val:shape){
            bytes *= val;
        }
        bytes *= size_of_dtype(dtype);
        
        this->storage = new Storage(bytes, device, 1);
        
    }


    uint64_t nbytes(){
        return this->storage->nbytes;
    }

    bool is_contiguous(){

        size_t expected = 1;
        for(int d = (int) ndim - 1; d>= 0; --d){
            if (shape[d] ==1){continue;}
            if (stride[d] != expected){return false;}
            expected *= shape[d];
        
        
        }

        return true;


    };


    void slice(Slices& slice){

        if (slice.size() != this->shape.size()){
            throw std::invalid_argument("slice rank mismatch");
        }

        auto norm = [](long i, long dim) -> long {
            if (i < 0) i+= dim;
            return i;
        };

        auto ceildiv_pos = [](long a, long b) -> long {

            return (a + b - 1) / b;
        };
        //make a copy so if we error we leave in good tensor in a good state
        Shape new_shape = shape;
        Stride new_stride = stride;
        ptrdiff_t new_offset = (ptrdiff_t)offset;

        ptrdiff_t n_offset = (ptrdiff_t) offset;

        // offset becomes new_offset = old_offset + start0 * stride[0] + start1 * stride[1] + ...
        // len_i = ceil( (end - start) / step )
        // new_stride[i] = old_stride[i] * step

        for (size_t i = 0; i < slice.size(); ++i){
            auto [start, end, step] = slice[i];
            if (step == 0) throw std::invalid_argument("step cannot be 0");
            start= (long) norm(start, shape[i]);
            end = norm(end, shape[i]);

            const long dim = (long) shape[i];

            if (step > 0) {
                start = std::clamp(start, 0L, dim);
                end = std::clamp( end, 0L, dim);

                long len = (end <= start) ? 0 : ceildiv_pos(end - start, step);

                new_shape[i] = (size_t)len;
                new_stride[i] = (ptrdiff_t)new_stride[i] * step;
                new_offset += (ptrdiff_t)start * new_stride[i];

            } else {
                start = std::clamp(start, -1L, dim -1L);
                end = std::clamp(end, -1L, dim -1L);
                long pstep = -step;

                long span = (start - end);
                long len = (span <= 0) ? 0 : ((span + pstep - 1) / step);
                new_shape[i] = (size_t)len;
                new_stride[i] = (ptrdiff_t)new_stride[i] * step;
                new_offset += (ptrdiff_t)start * (ptrdiff_t)stride[i];
        }
    }

        // commit
        shape = std::move(new_shape);
        stride = std::move(new_stride);
        offset = (Offset)new_offset;
        ndim = (uint8_t)shape.size();
    };


    void permute(const Shape& perm_arg){
        //use the shape as a permute arguement
        /*
        a0 a1 
        b0 b1

        is seen in memory as

        a0
        a1
        b0
        b1

        becomes

        a0 b0
        a1 b1
        */

        if (perm_arg.size() != ndim)
        {
            throw std::invalid_argument("permute rank mismatch");
        }

        std::vector<int> perm(perm_arg.size());
        for (size_t i = 0; i < perm_arg.size(); ++i)
        {
            int d = perm_arg[i];
            if (d < 0)
                d += ndim;
            perm[i] = d;
        }

        std::vector<char> seen(ndim, 0);
        for (int d : perm)
        {
            if (d < 0 || d >= ndim)
                throw std::out_of_range("permute: dim index out of range");
            if (seen[d])
                throw std::invalid_argument("permute: duplicate dim in permutation");
            seen[d] = 1;

        }

        Shape new_shape(ndim);
        Stride new_stride(ndim);
        for (size_t i = 0; i < perm.size(); ++i)
        {
            new_shape[i] = shape[perm[i]];
            new_stride[i] = stride[perm[i]];
        }

        shape.swap(new_shape);
        stride.swap(new_stride);
    };




    




    uint64_t numel(){
        u_int64_t numel_ = 1;

        for (auto s : shape)
            numel_ *= (u_int64_t)s;

        return numel_;
    };





    friend std::ostream& operator <<(std::ostream& os, const Tensor& t){

            os << "Tensor(" << "shape=" << t.shape << ", stride=" << t.stride << ", " << t.offset << ", " << t.dtype << ", " << t.storage->device << ", " << t.is_leaf << ", " << t.requires_grad << ")";

           

            return os;
    }







};



