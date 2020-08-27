function toloopy(node::Node)
    gem_expr = togem(node)
    return gemtoloopy(gem_expr)
end