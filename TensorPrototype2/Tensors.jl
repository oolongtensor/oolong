include("Node.jl")

abstract type AbstractTensor <: Node end

struct ScalarVariable <: AbstractTensor end

Scalar = Union{ScalarVariable, Base.Complex, Base.Real}

struct ConcreteTensor <: AbstractTensor
    # shape
    # indices

    # Some kind of a collection of Scalars
    value
    children
end

ConcreteTensor(x::Scalar) = ConcreteTensor([x], ())
