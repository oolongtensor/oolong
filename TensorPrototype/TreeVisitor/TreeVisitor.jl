include("../Operations.jl")
include("ZeroRemover.jl")

function _traversal(node::Node, visitfn!::Function, visited::Union{BitArray, Nothing})
    if visited === nothing || !visited[node.index]
        visitfn!(node)
        if visited !== nothing
            visited[node.index] = true
        end
        for child in node.children
            _traversal(child, visitfn!, visited)
        end
    end
end

function traversal(node::Node, visitfn!::Function, ignorevisited::Bool)
    if ignorevisited
        visited = BitArray(undef, getnumberofnodes())
    else
        visited = nothing
    end
    _traversal(node, visitfn!, visited)
    return node
end
