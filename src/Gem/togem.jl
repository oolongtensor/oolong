function _togem(A::Tensor{T}) where T<:Number
    return gem.Literal(A.value)
end

function _togem(A::Tensor{T,0}) where T
    return togem(A.value)
end

function _togem(A::Tensor{T}) where T
    shape = size(A.value)
    gems = permutedims(togem.(A.value))
    for i in reverse(shape)
        gems = [gem.ListTensor(gems[(i*(j-1)+1):i*j]) for j in 1:div(length(gems), i)]
    end
    return gems[1]
end

function _listtensor(A::Array{T}) where T
    step = strides(A)[end]
    num_steps = length(A) / step
    return _listtensor([reshape(A[(i-1)*step : i*step], size(A)[1:(end-1)]) for i in 1:num_steps])
end

function _togem(A::ConstantTensor{T}) where T<:Number
    # TODO this does not work if any vector space has unknown dimension
    return gem.Literal(fill(A.value, tuple([dim(V) for V in A.shape]...)))
end

#=function _togem(A::DeltaTensor)
    return IdentityGemTensor([dim(V) for V in A.shape]...)
end=#

function _togem(A::ZeroTensor)
    return gem.Zero(tuple([dim(V) for V in A.shape]...))
end

function _togem(A::VariableTensor)
    for V in A.shape
        if dim(V) === nothing
            throw(DomainError(A.shape, "The tensor must have a known shape."))
        end
    end
    return gem.Variable(A.name, tuple([dim(V) for V in A.shape]...))
end

GemIndexTypes = Union{Int, PyObject}

function _togem(in::IndexingOperation, A::PyObject, indices::Tuple{Vararg{GemIndexTypes}})
    return gem.Indexed(A, indices)
end

freeIndices = Dict{FreeIndex, PyObject}()

function _togem(i::FreeIndex)
    global freeIndices
    # Gem gives indices their owns ids, and we want them to be consistent
    if haskey(freeIndices, i)
        return freeIndices[i]
    elseif haskey(freeIndices, i')
        return freeIndices[i']
    else
        gemindex = gem.Index(i.name, dim(i.V))
        freeIndices[i] = gemindex
        return gemindex
    end
end

function _togem(i::FixedIndex)
    # Python starts indexing from 0 and Julia from 1
    return i.value - 1
end

function _togem(indices::Indices)
    return tuple([_togem(i) for i in indices.indices]...)
end

```Helper function that indexes tensors. Returns the tensors and the indices.
```
function _toscalar(tensors::Tuple{Vararg{PyObject}})
    indices = []
    for i in 1:length(tensors[1].shape)
        push!(indices, gem.Index(nothing, tensors[1].shape[i]))
    end
    return [gem.Indexed(tensor, indices) for tensor in tensors], tuple(indices...)
end

function _toscalar(tensor::PyObject)
    scalars, indices = _toscalar((tensor,))
    return scalars[1], indices
end

function _togem(add::AddOperation, children::Vararg{PyObject})
    indexed, indices = _toscalar(children)
    sum = indexed[1]
    for expr in indexed[2:end]
        sum += expr
    end
    return gem.ComponentTensor(sum, tuple(indices...))
end


function _togem(ou::OuterProductOperation, A::PyObject, B::PyObject)
    exprA, indicesA = _toscalar(A)
    exprB, indicesB = _toscalar(B)
    return gem.ComponentTensor(gem.Product(exprA, exprB), (indicesA..., indicesB...))
end

function _togem(comp::ComponentTensorOperation, expr::PyObject, indices::Tuple{Vararg{PyObject}})
    if expr.shape != ()
        expr, new_indices = _toscalar(expr)
        return gem.ComponentTensor(expr, (new_indices..., indices...))
    else
        return gem.ComponentTensor(expr, indices)
    end
end

function _togem(is::IndexSumOperation, expr::PyObject, indices::Tuple{Vararg{PyObject}})
    indexed, new_indices = _toscalar(expr)
    return gem.ComponentTensor(gem.IndexSum(indexed, indices), new_indices)
end

function _togem(root::RootNode, node)
    return RootNode(node)
end

function togem(node::Union{Node})
    return traversal(node, x-> x, _togem, nothing, nothing)
end

function togem(i::Number)
    return gem.Literal(i)
end