include("Node.jl")
include("VectorSpace.jl")

abstract type AbstractTensor <: Node end

# TODO Is this inheritance a good idea?
struct ScalarVariable <: AbstractTensor end

Scalar = Union{ScalarVariable, Base.Complex, Base.Real}

struct Tensor <: AbstractTensor
    # Some kind of a collection of Scalars
    value
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children
    # TODO Check x consists of scalars, if possible
    function Tensor(x::AbstractArray, Vs::Vararg{AbstractVectorSpace})
        if ndims(x) != length(Vs)
            error("Wrong number of vector spaces")
        end
        for i in 1:ndims(x)
            if size(x)[i] != dim(Vs[i])
                error("Dimension does not match with vector space rank")
            end
        end
        new(x, Vs, ())
    end
end
