using LightGraphs, MetaGraphs

abstract type Node end

_AST = MetaDiGraph()

function addnode!(node::Node, children::Vararg{Node})
    add_vertex!(_AST)
    set_prop!(_AST, nv(_AST), :node, node)
    for node in children
        add_edge!(_AST, node.index, nv(_AST))
    end
end
