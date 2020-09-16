function execute(node::Node, variables::Array)
    return executegem(togem(node), findgemvariables(node), variables)
end

function execute(node::Node)
    return execute(node, [])
end

"""
    Kernel(node::Node)

Creates a PyOp2 kernel from a given node. This can be executed repeatedly.
"""
struct Kernel
    knl::PyObject
    shape::Tuple{Vararg{Int}}
    variables::Array{PyObject}
    function Kernel(node::Node)
        variables = findgemvariables(node)
        gemexpr = togem(node)
        shape = gemexpr.shape
        knl = gemtoop2knl(gemexpr, variables)
        new(knl, shape, variables)
    end
end

"""
    execute(expr::Union{Kernel, Node}, variables::Dict{String, T}) where T

Execute the [`Kernel`](@ref) knl with the variable-assignments in variables.

## Example
```
julia> A = VariableTensor("A", RnSpace(3))
julia> knl = Kernel(A[1])
julia> execute(knl, Dict("A"=>[1.0, 2.0, 3.0]))
1-element Array{Float64,1}:
 1.0
```
"""
function execute(knl::Kernel, variables::Dict{String, T}) where T
    _preprocessvariables(variables)
    return executeop2knl(knl.knl, knl.shape, [variables[var.name] for var in knl.variables])
end

function execute(node::Node, variables::Dict{String, T}) where T
    knl = Kernel(node)
    return execute(knl, variables)
end

"""
    execute(expr::Union{Node, Kernel}, variables::Dict{String, T}) where T

Execute the [`Kernel`](@ref) knl or [`Node`](@ref) with the
variable-assignments in variables.

## Example
```
julia> A = VariableTensor("A", RnSpace(3))
julia> execute(A[1], "A"=>[1.0, 2.0, 3.0])
1-element Array{Float64,1}:
 1.0
```
"""
function execute(expr::Union{Node, Kernel}, variables::Vararg{Pair})
    return execute(expr, Dict{String, Any}(variables))
end

function _findvariables(visited, tensor::VariableTensor)
    return Set{VariableTensor}([tensor])
end

function _findvariables(visited, tensor::Tensor{T}) where T<:Union{Any, Tensor}
    return union(findvariables.(tensor.value)...)
end

function _findvariables(visited, tensor::ConstantTensor{T}) where T<:VariableTensor
    return Set{VariableTensor}([tensor.value])
end

function _findvariables(visited, node::Node)
    return Set{VariableTensor}()
end

function _findvariables(visited, root::RootNode, found::Vararg{Set{VariableTensor}})
    return RootNode(union(found...))
end

function _findvariables(visited, node::Node, found::Vararg{Set{VariableTensor}})
    return union(found...)
end

function findvariables(node::Node)
    return traversal(node, x->x, _findvariables, nothing, nothing, nothing)
end

function findvariables(n::Number)
    return Set{VariableTensor}()
end

function findgemvariables(node::Node)
    return [togem(var) for var in findvariables(node)]
end

function _preprocessvariables(variables::Dict{String, Any})
    for pair in variables
        if variables[first(pair)] isa Number
            variables[first(pair)] = fill(last(pair), ())
        end
    end
end

function _preprocessvariables(variables::Dict{String, A}) where A<:Array
    return variables
end
