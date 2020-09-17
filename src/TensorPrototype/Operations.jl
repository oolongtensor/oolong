"""
Operation{rank} <: AbstractTensor{rank}

A supertype for all tensor operations.
"""
abstract type Operation{rank} <: AbstractTensor{rank} end

"""
    IndexSumOperation(A::AbstractTensor, indices::Indices, freeindices::Vararg{FreeIndex})

Symbolises summing over given free index of a tensor.
"""
struct IndexSumOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
    function IndexSumOperation(A::AbstractTensor, indices::Indices, freeindices::Vararg{FreeIndex})
        new{length(A.shape)}(A.shape, (A, indices), freeindices)
    end
end

"""
    contractioncheck(A::AbstractTensor)
Checks if we have have an upper and lower index. If there is, we create
an [`IndexSumOperation`](@ref) node.
"""
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
    freeindices = tuple(setdiff(A.freeindices, [i for i in contractionindices],
        [i' for i in contractionindices])...)
    return IndexSumOperation(A, Indices(contractionindices...), freeindices...)
end

"""
    AddOperation(children::Tuple{Vararg{AbstractTensor{rank}}},
        freeindices::Tuple{Vararg{FreeIndex}}) where rank

A node symbolising addition of tensors.
"""
struct AddOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{Vararg{AbstractTensor}}
    freeindices::Tuple{Vararg{FreeIndex}}
    function AddOperation(children::Tuple{Vararg{AbstractTensor{rank}}},
            freeindices::Tuple{Vararg{FreeIndex}}) where rank
        if length(children) > 1
            for child in children[2:length(children)]
                if child.shape != children[1].shape
                    throw(DimensionMismatch(string(
                        "Shapes ", children[1].shape, " and ", child.shape,
                        " don't match")))
                end
            end
        end
        newchildren = tuple(filter!(x -> !(x isa ZeroTensor),
            [children...])...)
        if length(newchildren) == 1
            return newchildren[1]
        end
        if length(newchildren) == 0
            return children[1]
        end
        return contractioncheck(new{rank}(children[1].shape, newchildren,
            freeindices))
    end
end

"""
    Base.:+(nodes::Vararg{Node})

Creates an [`AddOperation`](@ref) whose children are the nodes. The shapes of
the nodes must match.
"""
function Base.:+(nodes::Vararg{Node})
    return AddOperation(nodes, tuple(union([node.freeindices for node=nodes]...)...))
end

"""
    OuterProductOperation(shape::Tuple{Vararg{AbstractVectorSpace}},
        children::Tuple{AbstractTensor, AbstractTensor},

A node symbolising tensor outer product."""
struct OuterProductOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, AbstractTensor}
    freeindices::Tuple{Vararg{FreeIndex}}
    function OuterProductOperation(shape::Tuple{Vararg{AbstractVectorSpace}},
            children::Tuple{AbstractTensor, AbstractTensor},
            freeindices::Tuple{Vararg{FreeIndex}})
        return contractioncheck(new{length(shape)}(shape, children,
            freeindices))
    end
end

"""
    ⊗(A::AbstractTensor, B::AbstractTensor)

Returns an [`OuterProductOperation`](@ref) with A and B as its children.
"""
function ⊗(A::AbstractTensor, B::AbstractTensor)
    return OuterProductOperation((A.shape..., B.shape...), (A, B),  (A.freeindices..., B.freeindices...))
end

"""
    Base.:*(x::Scalar, A::AbstractTensor)

Shorthand for multiplying a tensor by a scalar. 
If multiplying a number by a variable, the number must be first.
"""
function Base.:*(x::Scalar, A::AbstractTensor)
    if x == 1 || x == ConstantTensor(1)
        return A
    elseif A == ConstantTensor(1)
        return Tensor(x)
    elseif x == 0 || x isa ZeroTensor || A isa ZeroTensor
        return ZeroTensor(A.shape...)
    else
        return Tensor(x) ⊗ A
    end
end

"""
    Base.:-(A::AbstractTensor)

Unary minus. Creates an [`OuterProductOperation`](@ref) between ConstantTensor(-1) and A.
"""
function Base.:-(A::AbstractTensor)
    return -1*A
end

"""
    Base.:-(A::AbstractTensor{rank}, B::AbstractTensor{rank}) where rank

Binary minus. Creates an [`AddOperation`](@ref) of A and -1*B.
"""
function Base.:-(A::AbstractTensor{rank}, B::AbstractTensor{rank}) where rank
    return A + (-1*B)
end


"""
    PowerOperation(x::AbstractTensor{0}, y::AbstractTensor{0})

Symbolises raising a scalar-shaped tensor x to a scalar-shaped power y.
"""
struct PowerOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor{0}, AbstractTensor{0}}
    freeindices::Tuple{Vararg{FreeIndex}}
    function PowerOperation(x::AbstractTensor{0}, y::AbstractTensor{0})
        new{0}((), (x, y), (x.freeindices..., y.freeindices...))
    end
end

"""
    Base.:^(x::AbstractTensor{0}, y::AbstractTensor{0})

Raises the scalar-shaped tensor x to the scalar-shaped power y. Returns a
[`PowerOperation`](@ref).
"""
function Base.:^(x::AbstractTensor{0}, y::AbstractTensor{0})
    return PowerOperation(x, y)
end


"""
    Base.:^(x::AbstractTensor{0}, y::Number)

Raises the scalar-shaped tensor x to the power y. Returns a
[`PowerOperation`](@ref).
"""
function Base.:^(x::AbstractTensor{0}, y::Number)
    return x^Tensor(y)
end


"""
    Base.sqrt(x::AbstractTensor{0})

Returns the square root of x as a [`PowerOperation`](@ref).
"""
function Base.sqrt(x::AbstractTensor{0})
    return PowerOperation(x, Tensor(1//2))
end


"""
    DivisionOperation(A::AbstractTensor, y::AbstractTensor{0})

Symbolises dividing tensor A by a scalar-shaped tensor y.
"""
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

"""
    Base.:/(A::AbstractTensor, y::AbstractTensor{0})

Creates a DivisionOperation where A is divived by y. y must be scalar-shaped.
"""
function Base.:/(A::AbstractTensor, y::AbstractTensor{0})
    return DivisionOperation(A, y)
end

"""
    Base.:/(A::AbstractTensor, y::Number)

Creates a DivisionOperation where A is divived by y.
"""
function Base.:/(A::AbstractTensor, y::Number)
    return DivisionOperation(A, Tensor(y))
end


"""
    Base.:/(y::Number, A::AbstractTensor{0})

Creates a DivisionOperation where y is divided by A, where A must be a scalar.
"""
function Base.:/(y::Number, A::AbstractTensor{0})
    return DivisionOperation(Tensor(y), A)
end

"""
    IndexingOperation(x::AbstractTensor, indices::Indices)

A node symbolising indexing a tensor by its every dimension.
"""
struct IndexingOperation{rank} <: Operation{rank}
    shape::Tuple{}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
    function IndexingOperation(x::AbstractTensor, indices::Indices)
        return contractioncheck(new{0}((),(x, indices), tuple(x.freeindices..., [i for i=indices.indices if i isa FreeIndex]...)))
    end
end

"""
    ComponentTensorOperation(shape::Tuple{Vararg{AbstractVectorSpace}},
        children::Tuple{AbstractTensor, Indices},
        freeindices::Tuple{Vararg{FreeIndex}})

A node symbolising component tensor.
"""
struct ComponentTensorOperation{rank} <: Operation{rank}
    shape::Tuple{Vararg{AbstractVectorSpace}}
    children::Tuple{AbstractTensor, Indices}
    freeindices::Tuple{Vararg{FreeIndex}}
    function ComponentTensorOperation(
            shape::Tuple{Vararg{AbstractVectorSpace}},
            children::Tuple{AbstractTensor, Indices},
            freeindices::Tuple{Vararg{FreeIndex}})
        new{length(shape)}(shape, children, freeindices)
    end
end

"""
    componenttensor(A::AbstractTensor, indices::Vararg{Index})

Creates a [`ComponentTensorOperation`](@ref) of A over indices. Indices must be a
subset of the free indices of A.
"""
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

"""
    Base.getindex(A::AbstractTensor, ys::Vararg{Index})

Creates an [`IndexingOperation`](@ref) of A indexed by ys. If every dimension of A is
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

"""
    Base.getindex(A::AbstractTensor, ys::Vararg{Union{String, Int, Index}})

A convenience function that allows indexing a tensor by an integer. The
function creates a corresponding [`FixedIndex`](@ref) object in the appropiate
vector space.

Technically the function can also create a [`FreeIndex`](@ref) from a string,
but this is not recommended, because the syntax makes no distinction between
upper and lower indices.
"""
function Base.getindex(A::AbstractTensor, ys::Vararg{Union{String, Int, Index}})
    indexarray = []
    for i in 1:length(ys)
        push!(indexarray, toindex(A.shape[i], ys[i]))
    end
    return Base.getindex(A, indexarray...)
end

"""
    Base.transpose(A::AbstractTensor)

Reverses the shape of A. Does this by indexing A and then creating a
[`ComponentTensorOperation`](@ref) with the indices reversed.
"""
function Base.transpose(A::AbstractTensor)
    indices = []
    for V in A.shape
        global idcounter
        push!(indices, FreeIndex(V, "tranpose", idcounter))
        idcounter += 1
    end
    return componenttensor(A[indices...], reverse(indices)...)
end

"""
    trace(A::AbstractTensor{2})

Returns the trace of a matrix of the shape (V, V'). 
"""
function trace(A::AbstractTensor{2})
    if A.shape[1] != dual(A.shape[2])
        throw(DomainError(A, "Cannot get a trace from a tensor whose shape is not of the form (V, V')"))
    end
    global counter
    counter += 1
    return A[FreeIndex(A.shape[1], "trace", counter), FreeIndex(A.shape[2], "trace", counter)]
end

function Base.show(io::IO, op::Operation, depth::Int)
    println(io, ["    " for i in 1:depth]..., typeof(op))
    for child in op.children
        if child isa Operation
            Base.show(io, child, depth + 1)
        else
            println(io, ["    " for i in 1:(depth+1)]..., child)
        end
    end
end

Base.show(io::IO, op::Operation) = Base.show(io, op, 0)