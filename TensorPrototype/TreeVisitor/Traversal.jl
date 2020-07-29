include("../Operations.jl")

struct RootNode <: Node
    children::Array{Node}
    index::Int
end

RootNode(node::Node) = RootNode([node], getnumberofnodes() + 1)

function _traversal(node::Node, visitfn!::Function, visited::Union{BitArray, Nothing})
    if visited === nothing || !visited[node.index]
        for child in node.children
            _traversal(child, visitfn!, visited)
        end
        visitfn!(node)
        if visited !== nothing
            visited[node.index] = true
        end
    end
end

function traversal(node::Node, visitfn!::Function, ignorevisited::Bool)
    root = RootNode(node)
    if ignorevisited
        visited = BitArray(undef, getnumberofnodes() + 1)
    else
        visited = nothing
    end
    _traversal(root, visitfn!, visited)
    return root.children[1]
end
