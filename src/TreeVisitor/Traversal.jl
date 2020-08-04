include("../TensorPrototype/Node.jl")

struct RootNode <: Node
    children::Array{Node}
    index::Int
end

RootNode(node::Node) = RootNode([node], getnumberofnodes() + 1)

function _traversal(node::Node, visitfn!::Function)
    visitfn!(node)
    for child in node.children
        _traversal(child, visitfn!)
    end
end

function traversal(node::Node, visitfn!::Function)
    root = RootNode(node)
    _traversal(root, visitfn!)
    return root.children[1]
end