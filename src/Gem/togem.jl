function _togem(visited, A::Tensor{T}) where T<:Number
    return gem.Literal(A.value)
end

function _togem(visited, A::Tensor{T,0}) where T
    return togem(A.value, visited)
end

function _togem(visited, A::Tensor{T}) where T
    shape = size(A.value)
    gems = PermutedDimsArray((x -> togem(x, visited)).(A.value), reverse([i for i in 1:length(shape)]))
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

function _togem(visited, A::ConstantTensor{T}) where T<:Number
    # TODO this does not work if any vector space has unknown dimension
    return gem.Literal(fill(A.value, tuple([dim(V) for V in A.shape]...)))
end

#=function _togem(visited, A::DeltaTensor)
    return IdentityGemTensor([dim(V) for V in A.shape]...)
end=#

function _togem(visited, A::ZeroTensor)
    return gem.Zero(tuple([dim(V) for V in A.shape]...))
end

function _togem(visited, A::VariableTensor)
    for V in A.shape
        if dim(V) === nothing
            throw(DomainError(A.shape, "The tensor must have a known shape."))
        end
    end
    return gem.Variable(A.name, tuple([dim(V) for V in A.shape]...))
end

GemIndexTypes = Union{Int, PyObject}

function _togem(visited, in::IndexingOperation, A::PyObject, indices::Tuple{Vararg{GemIndexTypes}})
    return gem.Indexed(A, indices)
end

freeIndices = Dict{FreeIndex, PyObject}()

function _togem(visited, i::FreeIndex)
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

function _togem(visited, i::FixedIndex)
    # Python starts indexing from 0 and Julia from 1
    return i.value - 1
end

function _togem(visited, indices::Indices)
    return tuple([_togem(visited, i) for i in indices.indices]...)
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

function _togem(visited, add::AddOperation, children::Vararg{PyObject})
    indexed, indices = _toscalar(children)
    sum = indexed[1]
    for expr in indexed[2:end]
        sum += expr
    end
    return gem.ComponentTensor(sum, tuple(indices...))
end


function _togem(visited, ou::OuterProductOperation, A::PyObject, B::PyObject)
    exprA, indicesA = _toscalar(A)
    exprB, indicesB = _toscalar(B)
    return gem.ComponentTensor(gem.Product(exprA, exprB), (indicesA..., indicesB...))
end

function _togem(visited, div::DivisionOperation, A::PyObject, B::PyObject)
    expr, indices = _toscalar(A)
    return gem.ComponentTensor(gem.Division(expr, B), indices)
end

function _togem(visited, comp::ComponentTensorOperation, expr::PyObject, indices::Tuple{Vararg{PyObject}})
    if expr.shape != ()
        expr, new_indices = _toscalar(expr)
        return gem.ComponentTensor(expr, (new_indices..., indices...))
    else
        return gem.ComponentTensor(expr, indices)
    end
end

function _togem(visited, is::IndexSumOperation, expr::PyObject, indices::Tuple{Vararg{PyObject}})
    indexed, new_indices = _toscalar(expr)
    return gem.ComponentTensor(gem.IndexSum(indexed, indices), new_indices)
end

function _togem(visited, po::PowerOperation, expr::PyObject, pow::PyObject)
    return gem.Power(expr, pow)
end

function _togem(visited, sin::SineOperation, expr::PyObject)
    return gem.MathFunction("sin", expr)
end

function _togem(visited, cos::CosineOperation, expr::PyObject)
    return gem.MathFunction("cos", expr)
end

function _togem(visited, tan::TangentOperation, expr::PyObject)
    return gem.MathFunction("tan", expr)
end

function _togem(visited, asin::ArcsineOperation, expr::PyObject)
    return gem.MathFunction("asin", expr)
end

function _togem(visited, acos::ArccosineOperation, expr::PyObject)
    return gem.MathFunction("acos", expr)
end

function _togem(visited, atan::ArctangentOperation, expr::PyObject)
    return gem.MathFunction("atan", expr)
end

function _togem(visited, root::RootNode, node)
    return RootNode(node)
end

function togem(node::Union{Node}, visited::Union{Dict, Nothing})
    return traversal(node, x-> x, _togem, nothing, nothing, visited)
end

function togem(node::Union{Node})
    return togem(node, nothing)
end

function togem(i::Number, visited)
    return gem.Literal(i)
end