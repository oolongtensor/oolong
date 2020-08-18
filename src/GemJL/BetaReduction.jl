function _betareduction(in::IndexSumGem, indices::pair{GemIndex, GemIndex}, A::ScalarGem)
    i, j = indices
    return in.index == i ? IndexSumGem(A, j) : IndexSumGem(A, in.index)
end

function _betareduction(node::Node, indices::pair{GemIndex, GemIndex})
    return node
end

```Replaces a GemIndex in an expression with another GemIndex.
```
function betareduction(node::Node, reduction::pair{GemIndex, GemIndex})
    traversal(node, x -> x, _betareduction, nothing, reduction)
end