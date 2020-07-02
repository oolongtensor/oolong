using LightGraphs, MetaGraphs

mutable struct AST
    ast::MetaDiGraph
    root::Integer
end

AST() = AST(MetaDiGraph(), 1)

function addchild!(ast::AST, child, node_num)
    addnode!(ast, child)
    if nv(ast.ast) > 1
        add_edge!(ast.ast, node_num, nv(ast.ast))
    end
end

function addnode!(ast::AST, expression)
    add_vertex!(ast.ast)
    set_prop!(ast.ast, nv(ast.ast), :expr, expression)
    return nv(ast.ast)
end

function addroot!(ast::AST, parent, child_nums...)
    addnode!(ast, parent)
    ast.root = nv(ast.ast)
    for child_num in child_nums
        add_edge!(ast.ast, nv(ast.ast), child_num)
    end
end
