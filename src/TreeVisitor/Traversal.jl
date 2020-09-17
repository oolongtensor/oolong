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

"""
    traversal(node, pretraversalfn::Function, visitfn::Function,
        pretraversalfnargs::Union{Any, Nothing}, visitfnargs::Union{Any, Nothing},
        visited::Union{Dict, Nothing}, posttraversal=false)

Traverses an AST, calling visitfn on the nodes of the tree.

# Arguments
- `node`: the root of the tree to traverse. Assumes that each node has children
attribute.
- `pretraversalfn::Function` A function to be executed before the traversal
- `visitfn::Function` The function that is executed on each visit. If
posttraversal is false, and visitfnargs nothing, the function must have signature 
`visitfn(visited, node, new_children...)`. If visitfnargs is not nothing, the
signature is `visitfn(visited, node, visitfnargs, new_children...). If
posttraversal is true, the signature must be `visitfn(node, fn)`,
or `visitfn(node, visitfn, fn)`, where `fn` is a function that calls traversal
on a node.
- `pretraversalfnargs::Union{Any, Nothing}` Arguments for the pretraversalfn.
- `visitfnargs::Union{Any, Nothing}` Arguments for traversalfn.
- `visited::Union{Dict, Nothing}` If not nothing, a dictionary with the results
of calling traversalfn on different nothing. If nothing, this dictionary is
created in traversal.
- `posttraversal=false` Determines the way traversal traverses each node. If
`posttraversal=false`, the traversal first processes the children of a node,
and then the parent node (example: [`togem(node::Node)`](@ref)). If
`posttraversal=true`, the visitfn first processes the parent and then calls
visitfn on its children if needed (example:
[`differentiate(A::AbstractTensor{0}, x::VariableTensor{0})`](@ref)).
"""
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