using LightGraphs, MetaGraphs

abstract type Node end

_AST = MetaDiGraph()

function preparegraph(children::Vararg{Node})
    add_vertex!(_AST)
    for node in children
        add_edge!(_AST, node.index, nv(_AST))
    end
    return nv(_AST)
end

function addnodetograph(node::Node)
    if has_prop(_AST, node.index, :node)
        throw(DomainError("Property has already been set"))
    end
    set_prop!(_AST, node.index, :node, node)
    # Enables using this as the last method of a constructor
    return node
end

function children(node::Node)
    return [get_prop(_AST, vertex, :node) for vertex in inneighbors(_AST, node.index)]
end
