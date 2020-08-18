function _betareduction(node::Node)
    return node
end

function _betareduction(in::IndexSumGem, indices::pair{GemIndex, GemIndex}, A::ScalarGem)
end

```Replaces a GemIndex in an expression with another GemIndex.
```
function betareduction(node::Node, reduction::pair{GemIndex, GemIndex})
    traversal(node, x -> x, _betareduction, nothing, reduction)
end