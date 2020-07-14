import Base

abstract type AbstractVectorSpace end

counter = 0

struct VectorSpace <: AbstractVectorSpace
    dim::Union{Int, Nothing}
    id::Int
    function VectorSpace(dim::Union{Int, Nothing})
        global counter
        counter += 1
        new(dim, counter)
    end
end

VectorSpace() = VectorSpace(nothing)

struct DualVectorSpace <: AbstractVectorSpace
    vectorSpace::AbstractVectorSpace
end

struct RnSpace <: AbstractVectorSpace
    dim::Int
end

dual(V::VectorSpace) = DualVectorSpace(V)
dual(Vstar::DualVectorSpace) = Vstar.vectorSpace
dual(R::RnSpace) = R
dim(V::VectorSpace) = V.dim
dim(Vstar::DualVectorSpace) = Vstar.vectorSpace.dim
dim(R::RnSpace) = R.dim

Base.adjoint(V::AbstractVectorSpace) = dual(V)
