"""A supertype for all tensor operations."""
abstract type Operation{rank} <: AbstractTensor{rank} end

"""Symbolises summing over given indices."""
struct IndexSumOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
end

IndexSumOperation(A::AbstractTensor, indices::Indices, freeindices::Vararg{FreeIndex}) = IndexSumOperation{length(A.shape)}(A.shape, (A, indices), freeindices)

"""Checks if we have have an upper and lower index. If there is, we create
an IndexSumOperation node."""
function contractioncheck(A::AbstractTensor)
    contractionindices = []
    for (i, index) in enumerate(A.freeindices)
        if index' in A.freeindices[(i+1):end]
            push!(contractionindices, index)
        end
    end
    if isempty(contractionindices)
        return A
    end
    freeindices = tuple(setdiff(A.freeindices, [i for i in contractionindices], [i' for i in contractionindices])...)
    return IndexSumOperation(A, Indices(contractionindices...), freeindices...)
end

"""A node symbolising addition of tensors."""
struct AddOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{Vararg{AbstractTensor}}
    freeindices::Tuple{Vararg{FreeIndex}}
    function AddOperation(children::Tuple{Vararg{AbstractTensor{rank}}}, freeindices::Tuple{Vararg{FreeIndex}}) where rank
        if length(children) > 1
            for child in children[2:length(children)]
                if child.shape != children[1].shape
                    throw(DimensionMismatch(string("Shapes ", children[1].shape, " and ", child.shape, " don't match")))
                end
            end
        end
        newchildren = tuple(filter!(x -> !(x isa ZeroTensor), [children...])...)
        if length(newchildren) == 1
            return newchildren[1]
        end
        if length(newchildren) == 0
            return children[1]
        end
        return contractioncheck(new{rank}(children[1].shape, newchildren, freeindices))
    end
end

function Base.:+(nodes::Vararg{Node})
    return AddOperation(nodes, tuple(union([node.freeindices for node=nodes]...)...))
end

"""A node symbolising tensor outer product."""
struct OuterProductOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, AbstractTensor}
    freeindices::Tuple{Vararg{FreeIndex}}
    function OuterProductOperation(shape::Tuple{Vararg{AbstractVectorSpace}}, children::Tuple{AbstractTensor, AbstractTensor}, freeindices::Tuple{Vararg{FreeIndex}})
        return contractioncheck(new{length(shape)}(shape, children, freeindices))
    end
end

function ⊗(x::AbstractTensor, y::AbstractTensor)
    return OuterProductOperation((x.shape..., y.shape...), (x, y),  (x.freeindices..., y.freeindices...))
end

"""Shorthand for multiplication of two scalars. Returns an
OuterProductOperation and simplifies for multiplication by zero or one.
    
If multiplying a number by a variable, the number must be first."""
function Base.:*(x::Scalar, y::AbstractTensor{0})
    if x == 1 || x == ConstantTensor(1)
        return Tensor(y)
    elseif y == ConstantTensor(1)
        return Tensor(x)
    elseif x == 0 || x isa ZeroTensor || y isa ZeroTensor
        return ZeroTensor()
    else
        return Tensor(x) ⊗ Tensor(y)
    end
end

"""Shorthand for multiplying a tensor by a scalar."""
function Base.:*(x::Scalar, A::AbstractTensor)
    if x == 1 || x == ConstantTensor(1)
        return A
    end
    return Tensor(x) ⊗ A
end

function Base.:-(A::AbstractTensor)
    return -1*A
end

function Base.:-(A::AbstractTensor{rank}, B::AbstractTensor{rank}) where rank
    return A + (-1*B)
end

struct DivisionOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, AbstractTensor{0}}
    freeindices::Tuple{Vararg{FreeIndex}}
    function DivisionOperation(A::AbstractTensor, y::AbstractTensor{0})
        if y isa ZeroTensor
            throw(DivideError())
        elseif y == ConstantTensor(1)
            return A
        end
        return contractioncheck(new{length(A.shape)}((A.shape), (A, y), (A.freeindices..., y.freeindices...)))
    end
end

function Base.:/(A::AbstractTensor, y::AbstractTensor{0})
    return DivisionOperation(A, y)
end

function Base.:/(A::AbstractTensor, y::Number)
    return DivisionOperation(A, Tensor(y))
end

"""A node symbolising indexing a tensor by its every dimension."""
struct IndexingOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
    function IndexingOperation(x::AbstractTensor, indices::Indices)
        return contractioncheck(new{0}((),(x, indices), tuple(x.freeindices..., [i for i=indices.indices if i isa FreeIndex]...)))
    end
end

"""A node symbolising component tensor."""
struct ComponentTensorOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
    function ComponentTensorOperation(shape::Tuple{Vararg{AbstractVectorSpace}}, children::Tuple{AbstractTensor, Indices}, freeindices::Tuple{Vararg{FreeIndex}})
        new{length(shape)}(shape, children, freeindices)
    end
end

"""Creates a component tensor of A over indices. Indices must be a
subset of the free indices of A."""
function componenttensor(A::AbstractTensor, indices::Vararg{Index})
    if length(indices) == 0
        return A
    end
    if !(indices ⊆ A.freeindices)
        throw(DomainError(string("The free indices ", indices, " are not a subset of ", A.freeindices)))
    end
    shape = tuple(A.shape..., [i.V for i in indices]...)
    freeindices = tuple(setdiff(A.freeindices, indices, [i' for i in indices])...)
    return ComponentTensorOperation(shape, (A, Indices(indices...)), freeindices)
end

idcounter = 0

"""Creates an IndexingOperation of A indexed by ys. If every dimension of A is
not indexed, creates a ComponentTensorOperation over the unindexed dimensions."""
function Base.getindex(A::AbstractTensor, ys::Vararg{Index})
    if length(ys) == 0
        return A
    end
    if length(A.shape) < length(ys)
        throw(BoundsError(A.shape, ys))
    end
    for i in 1:length(ys)
        if A.shape[i] != ys[i].V
            throw(DimensionMismatch(string(ys[i], " is not in the vector space ", A.shape[i])))
        end
    end
    addedindices = []
    global idcounter
    for i in 1:(length(A.shape) - length(ys))
        push!(addedindices, FreeIndex(A.shape[length(ys) + i], "", idcounter))
        idcounter += 1
    end
    return componenttensor(IndexingOperation(A, Indices(ys..., addedindices...)), addedindices...)
end

"""A convenience function that allows indexing a tensor by an integer. The
function creates a corresponding FixedIndex object in the appropiate vector
space.

Technically the function also does the same for a string, but this is not
recommended, because the syntax makes no distinction between upper and
lower indices.
"""
function Base.getindex(A::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
    indexarray = []
    for i in 1:length(ys)
        push!(indexarray, toindex(A.shape[i], ys[i]))
    end
    return Base.getindex(A, indexarray...)
end

function Base.transpose(A::AbstractTensor)
    indices = []
    for V in A.shape
        global idcounter
        push!(indices, FreeIndex(V, "tranpose", idcounter))
        idcounter += 1
    end
    return componenttensor(A[indices...], reverse(indices)...)
end

function Base.show(io::IO, op::Operation, depth::Int)
    println(io, ["\t" for i in 1:depth]..., typeof(op))
    for child in op.children
        if child isa Operation
            Base.show(io, child, depth + 1)
        else
            println(io, ["\t" for i in 1:(depth+1)]..., child)
        end
    end
end

Base.show(io::IO, op::Operation) = Base.show(io, op, 0)