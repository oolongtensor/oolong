struct RootNode <: Node
    children::Tuple{Node}
end

RootNode(node::Node) = RootNode((node,))

function _traversal(node::Node, visitfn::Function, visitfnargs::Union{Any, Nothing}, visited)
    if haskey(visited, node)
        return visited(node)
    else
        new_children = [_traversal(child, visitfn, visitfnargs, visited) for child in node.children]
        result = visitfnargs !== nothing ?
            visitfn(node, visitfnargs, new_children...) : visitfn(node, new_children...)
        visited[node] = result
        return result
    end
end

function traversal(node::Node, pretraversalfn::Function, visitfn::Function,
        pretraversalfnargs::Union{Any, Nothing}, visitfnargs::Union{Any, Nothing})
    root = RootNode(node)
    root = pretraversalfnargs !== nothing ? pretraversalfn(root, pretraversalfnargs) : pretraversalfn(root)
    visited = Dict{Node, Node}()
    root = _traversal(root, visitfn, visitfnargs, visited)
    return root.children[1]
end