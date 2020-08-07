struct RootNode <: Node
    children::Tuple{Node}
end

RootNode(node::Node) = RootNode((node,))

updatechildren(root::RootNode, node::Node) = RootNode((node,))

function _traversal(node::Node, visitfn::Function, visitfnargs::Union{Any, Nothing})
    new_children = [_traversal(child, visitfn, visitfnargs) for child in node.children]
    return visitfnargs !== nothing ? visitfn(node, visitfnargs, new_children...) : visitfn(node, new_children...)
end

function traversal(node::Node, pretraversalfn::Function, visitfn::Function,
        pretraversalfnargs::Union{Any, Nothing}, visitfnargs::Union{Any, Nothing})
    root = RootNode(node)
    root = pretraversalfnargs !== nothing ? pretraversalfn(root, pretraversalfnargs) : pretraversalfn(root)
    root = _traversal(root, visitfn, visitfnargs)
    return root.children[1]
end