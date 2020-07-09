abstract type AbstractVectorSpace end

counter = 0

struct VectorSpace <: AbstractVectorSpace
    dim::Int
    id::Int
    function VectorSpace(dim::Integer)
        global counter
        counter += 1
        new(dim, counter)
    end
end

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
