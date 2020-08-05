include("../TensorPrototype/Node.jl")

struct RootNode <: Node
    children::Tuple{Node}
end

RootNode(node::Node) = RootNode((node,))

updatechildren(root::RootNode, node::Node) = RootNode((node,))

function _traversal(node::Node, visitfn::Function, visitfnargs)
    new_children = [_traversal(child, visitfn, visitfnargs) for child in node.children]
    return visitfn(updatechildren(node, new_children...), visitfnargs)
end

function traversal(node::Node, pretraversalfn::Function, visitfn::Function, pretraversalfnargs, visitfnargs)
    root = RootNode(node)
    println(pretraversalfn)
    if pretraversalfnargs !== nothing
        root = pretraversalfn(root, pretraversalfnargs)
    else
        root = pretraversalfn(root)
    end
    _traversal(root, visitfn, visitfnargs)
    return root.children[1]
end