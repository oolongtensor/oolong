module TensorDSL

export
Node,

AbstractVectorSpace, VectorSpace, DualVectorSpace, RnSpace, dual, dim,

Index, FreeIndex, FixedIndex, Indices, toindex,

AbstractTensor, TerminalTensor, ScalarVariable, Scalar, VariableTensor,
Tensor, DeltaTensor, ZeroTensor, ConstantTensor,

Operation, IndexSumOperation, AddOperation, OuterProductOperation, ⊗,
IndexingOperation, ComponentTensorOperation, componenttensor,

SineOperation, sin, CosineOperation, cos, TangentOperation, tan,

diff,

Assignment, assign,

updatechildren, updatevectorspace,

GemNode, GemTensor, ScalarGem, ScalarExprGem, GemTerminal, GemConstant,
VariableGemIndex, GemIndex, GemIndexTypes, LiteralGemTensor, ZeroGemTensor,
IdentityGemTensor, VariableGemTensor, shape, IndexSumGem, ComponentTensorGem,
IndexedGem, MathFunctionGem, SumGem, ProductGem,

togem

TensorDSL
include("TensorPrototype/Node.jl")
include("TensorPrototype/VectorSpace.jl")
include("TensorPrototype/Indices.jl")
include("TensorPrototype/Tensors.jl")
include("TensorPrototype/Operations.jl")
include("TensorPrototype/Trigonometry.jl")
include("TensorPrototype/Differentation.jl")
include("TreeVisitor/Traversal.jl")
include("TreeVisitor/UpdateNodes.jl")
include("TensorPrototype/Assignment.jl")
include("GemJL/Gem.jl")
include("GemJL/ToGem.jl")

end
