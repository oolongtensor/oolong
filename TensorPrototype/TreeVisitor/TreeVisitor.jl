include("../Operations.jl")

function _travelsal(node::Node, visitfn!::Function, visited::Union{BitArray, Nothing})
    if ignorevisited == nothing || !visited[node.index]
        visitfn!(node)
        if ignorevisited != nothing
            visited[node.index] = true
        end
        for child in node.children
            _traversal(node, visitfn!, visited)
        end
    end
end

function travelsal(node::Node, visitfn!::Function, ignorevisited::Boolean)
    if ignorevisited
        visited = BitArray(undef, getnumberofnodes())
    else
        visited = nothing
    end
    _traversal(node, visitfn!, visited)
end
