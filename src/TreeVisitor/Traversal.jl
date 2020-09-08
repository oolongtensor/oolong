struct RootNode <: Node
    children::Tuple{Any}
    function RootNode(node)
        new((node,))
    end
end

function _traversal(node, visitfn::Function, visitfnargs::Union{Any, Nothing}, visited)
    if haskey(visited, node)
        return visited[node]
    else
        new_children = [_traversal(child, visitfn, visitfnargs, visited) for child in node.children]
        result = visitfnargs !== nothing ?
            visitfn(visited, node, visitfnargs, new_children...) :
                visitfn(visited, node, new_children...)
        visited[node] = result
        return result
    end
end

"""
visitfn requires the interface node::Node, visitfnargs, _posttraversal
or node::Node, _posttraversal if visitfnargs is nothing.
"""
function _posttraversal(node, visitfn::Function, visitfnargs::Union{Any, Nothing}, visited)
    if haskey(visited, node)
        return visited[node]
    else
        result = visitfnargs !== nothing ?
            visitfn(node, visitfnargs, (x -> _posttraversal(x, visitfn, visitfnargs, visited))) :
                visitfn(node, (x -> _posttraversal(x, visitfn, visitfnargs, visited)))
        visited[node] = result
        return result
    end
end

function traversal(node, pretraversalfn::Function, visitfn::Function,
        pretraversalfnargs::Union{Any, Nothing}, visitfnargs::Union{Any, Nothing},
        visited::Union{Dict, Nothing}, posttraversal::Bool)
    if visited === nothing
        visited = Dict{Any, Any}()
    end
    root = RootNode(node)
    root = pretraversalfnargs !== nothing ? pretraversalfn(root, pretraversalfnargs...) : pretraversalfn(root)
    root = posttraversal ? _posttraversal(
        root, visitfn, visitfnargs, visited) : _traversal(
                root, visitfn, visitfnargs, visited)
    return root.children[1]
end

function traversal(node, pretraversalfn::Function, visitfn::Function,
        pretraversalfnargs::Union{Any, Nothing}, visitfnargs::Union{Any, Nothing},
        visited::Union{Dict, Nothing})
    return traversal(node, pretraversalfn, visitfn, pretraversalfnargs,
            visitfnargs, visited, false)
end