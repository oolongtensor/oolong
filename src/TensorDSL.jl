__precompile__()
module TensorDSL

using PyCall

const gem = PyNULL()
const tsfc = PyNULL()
const isinst = PyNULL()
const gemtoop2knl = PyNULL()
const executeop2knl = PyNULL()

# https://github.com/JuliaPy/PyCall.jl/blob/master/README.md#using-pycall-from-julia-modules
function __init__()
    try
        copy!(tsfc, pyimport("tsfc"))
    catch LoadError
        println("Cannot import firedrake. Check that you are in the correct virtual environment.")
        return
    end
    copy!(gem, tsfc.fem.gem)
    copy!(isinst, pybuiltin("isinstance"))
    py"""
    from firedrake.parameters import parameters
    import tsfc
    gem = tsfc.fem.gem
    from gem import indices as make_indices
    impero_utils = tsfc.fem.gem.impero_utils
    import tsfc.loopy as tsfcloopy
    generate_loopy = tsfcloopy.generate
    import numpy as np
    import loopy
    from pyop2 import op2

    # Copy pasted and modified from https://github.com/firedrakeproject/firedrake/blob/2d0351fa769da4fa2d807355526e9400b778fb66/firedrake/slate/slac/compiler.py#L605
    def gem_to_loopy(gem_expr, gem_variables=None):
        '''Converts the gem expression dag into imperoc first, and then further into loopy.
        :return slate_loopy: loopy kernel for slate operations.
        '''
        if gem_variables is None:
            gem_variables = []
        # Creation of return variables for outer loopy
        shape = gem_expr.shape
        idx = make_indices(len(shape))
        indexed_gem_expr = gem.Indexed(gem_expr, idx)
        args = [loopy.GlobalArg("output", shape=shape)] + [
            loopy.GlobalArg(var.name, dtype=np.float64, shape=var.shape) for var in gem_variables]
        ret_vars = [gem.Indexed(gem.Variable("output", shape), idx)]
        preprocessed_gem_expr = impero_utils.preprocess_gem([indexed_gem_expr])
        # glue assignments to return variable
        assignments = list(zip(ret_vars, preprocessed_gem_expr))
        # Part A: slate to impero_c
        impero_c = impero_utils.compile_gem(assignments, (), remove_zeros=False)
        # Part B: impero_c to loopy
        return generate_loopy(impero_c, args, parameters["form_compiler"]["scalar_type"], "gem_loopy", [])

    def loopy_to_op2knl(knl):
        code = loopy.generate_code_v2(knl).device_code()
        # Include-statement makes sin, cos and tan work.
        code = "#include <math.h>\n" + code.replace('void gem_loopy', 'static void gem_loopy')
        return op2.Kernel(code, "gem_loopy", ldargs=["-llapack"])

    def execute_op2knl(op2knl, shape, variables=None):
        if variables is None:
            variables = []
        s = op2.Set(1)
        vardata = [op2.Dat(s ** var.shape, var)(op2.READ) for var in variables]
        zero_mat = op2.Dat(s ** shape, np.zeros(shape))
        op2.par_loop(op2knl, zero_mat.dataset.set, zero_mat(op2.WRITE), *vardata)
        return zero_mat.data

    def gem_to_op2knl(gem_expr, gem_variables=None):
        return loopy_to_op2knl(gem_to_loopy(gem_expr, gem_variables))

    def execute_gem(gem_expr, gem_variables=None, variables=None):
        return execute_op2knl(gem_to_op2knl(gem_expr, gem_variables), gem_expr.shape, variables)

    """
    copy!(gemtoop2knl, py"gem_to_op2knl")
    copy!(executeop2knl, py"execute_op2knl")

end

export
Node,

AbstractVectorSpace, VectorSpace, DualVectorSpace, RnSpace, dual, dim,

Index, FreeIndex, FixedIndex, Indices, toindex,

AbstractTensor, TerminalTensor, Scalar, VariableTensor,
Tensor, DeltaTensor, ZeroTensor, ConstantTensor,

Operation, IndexSumOperation, AddOperation, OuterProductOperation, ⊗,
IndexingOperation, ComponentTensorOperation, componenttensor,
DivisionOperation, trace, PowerOperation,

SineOperation, CosineOperation, TangentOperation, ArcsineOperation,
ArccosineOperation, ArctangentOperation,

differentiate, divergence,

Assignment, assign, RootNode,

updatechildren, updatevectorspace,

togem, execute, Kernel,

gem, isinst, gemtoop2knl, executeop2knl, findvariables

TensorDSL
include("TensorPrototype/Node.jl")
include("TensorPrototype/VectorSpace.jl")
include("TensorPrototype/Indices.jl")
include("TensorPrototype/Tensors.jl")
include("TensorPrototype/Operations.jl")
include("TensorPrototype/Trigonometry.jl")
include("TreeVisitor/Traversal.jl")
include("TensorPrototype/Differentation.jl")
include("TreeVisitor/UpdateNodes.jl")
include("TensorPrototype/Assignment.jl")
include("Gem/togem.jl")
include("Gem/GenerateCode.jl")

end
