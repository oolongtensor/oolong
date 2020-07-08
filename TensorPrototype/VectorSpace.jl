abstract type AbstractVectorSpace end

counter = 0

struct VectorSpace <: AbstractVectorSpace
    dim
    id
    function VectorSpace(dim::Integer)
        global counter
        counter += 1
        new(dim, counter)
    end
end

struct DualVectorSpace <: AbstractVectorSpace
    vectorSpace::AbstractVectorSpace
end

dual(V::VectorSpace) = DualVectorSpace(V)
dual(Vstar::DualVectorSpace) = Vstar.vectorSpace
dim(V::VectorSpace) = V.dim
dim(V::DualVectorSpace) = V.vectorSpace.dim
