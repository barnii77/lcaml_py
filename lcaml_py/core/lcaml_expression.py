import itertools

from . import lcaml_parser as lcaml_parser
from . import parser_types as parser_types
from . import interpreter_vm as interpreter_vm_mod
from . import extern_python as extern_python
from . import pyffi

from .resolvable import Resolvable
from .ast_related import AstRelated
from .lcaml_lexer import Syntax
from .token_type import Token, TokenKind
from .lcaml_utils import (
    PhantomType,
    split_at_context_end,
    EqualsAny,
    get_unique_name,
    indent,
    expect_only_expression,
)
from .interpreter_types import Object, DType
from .operation_kind import OperationKind
from typing import List, Dict, Optional, Set, Iterable, Any, Tuple, Union

COMPILE_WITH_CONTEXT_LEAKING = True
JIT_BY_DEFAULT = False
SUPPRESS_JIT = False
JIT_OPT_LEVEL = 2
DEBUG_PRINT_OPTIMIZED_LLVM_IR = False
DEBUG_PRINT_UNOPTIMIZED_LLVM_IR = False
C_CALL_ERR_CODES = {ZeroDivisionError: 1}
C_CALL_ERR_EXCEPTIONS = {v: k for k, v in C_CALL_ERR_CODES.items()}


# lazy initialization of llvmlite (brython cannot catch the import error, so this is needed)
def initialize_llvmlite():
    if SUPPRESS_JIT:
        return

    g = globals()
    try:
        import llvmlite.ir as llvm_ir
        import llvmlite.binding as llvm_binding
        from ctypes import CFUNCTYPE, c_double, c_int64, c_bool, POINTER, byref as c_byref

        llvm_binding.initialize()
        llvm_binding.initialize_native_target()
        llvm_binding.initialize_native_asmprinter()
        target = llvm_binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        # And an execution engine with an empty backing module
        backing_mod = llvm_binding.parse_assembly("")
        LLVM_EXECUTION_ENGINE = llvm_binding.create_mcjit_compiler(
            backing_mod, target_machine
        )
    except Exception:
        pass
    else:
        g["llvm_ir"] = llvm_ir
        g["llvm_binding"] = llvm_binding
        g["CFUNCTYPE"] = CFUNCTYPE
        g["c_double"] = c_double
        g["c_int64"] = c_int64
        g["c_bool"] = c_bool
        g["POINTER"] = POINTER
        g["c_byref"] = c_byref
        g["target"] = target
        g["target_machine"] = target_machine
        g["LLVM_EXECUTION_ENGINE"] = LLVM_EXECUTION_ENGINE


def compile_llvm_module(ir_mod, main_func_name) -> int:
    """
    Compile the LLVM module object with the given engine.
    From the compiled module object, the main function is extracted
    and a function pointer to it is returned as an int.
    """
    binding_mod = llvm_binding.parse_assembly(str(ir_mod))
    binding_mod.verify()

    if DEBUG_PRINT_UNOPTIMIZED_LLVM_IR:
        print(str(binding_mod))

    try:
        pto = llvm_binding.create_pipeline_tuning_options(JIT_OPT_LEVEL)
        pb = llvm_binding.create_pass_builder(target_machine, pto)
        mpm = pb.getModulePassManager()
        mpm.run(binding_mod, pb)
    except AttributeError as e:
        if not hasattr(e, "__lcaml_traceback_info"):
            setattr(e, "__lcaml_traceback_info", [])
        getattr(e, "__lcaml_traceback_info").append("Check if you installed the dev version of llvmlite from conda according to the `Setup` section of README.md")
        raise e

    if DEBUG_PRINT_OPTIMIZED_LLVM_IR:
        print(str(binding_mod))

    LLVM_EXECUTION_ENGINE.add_module(binding_mod)
    LLVM_EXECUTION_ENGINE.finalize_object()
    LLVM_EXECUTION_ENGINE.run_static_constructors()
    return LLVM_EXECUTION_ENGINE.get_function_address(main_func_name)


TokenStream = List[Token]
Context = Dict[str, Object]


SYMBOL_TO_OPKIND = {
    "+": OperationKind.ADD,
    "-": OperationKind.SUB,
    "**": OperationKind.POW,
    "*": OperationKind.MUL,
    "/": OperationKind.DIV,
    "//": OperationKind.IDIV,
    "%": OperationKind.MOD,
    "!": OperationKind.NOT,
    "==": OperationKind.EQ,
    "!=": OperationKind.NEQ,
    "<": OperationKind.LT,
    ">": OperationKind.GT,
    "~": OperationKind.FLIP,
    "<=": OperationKind.LTE,
    ">=": OperationKind.GTE,
    "||": OperationKind.OR,
    "&&": OperationKind.AND,
    "|": OperationKind.BITOR,
    "&": OperationKind.BITAND,
    "^": OperationKind.BITXOR,
    "<<": OperationKind.LSH,
    ">>": OperationKind.RSH,
}

OPKIND_TO_PYTHON_SYMBOL = {
    OperationKind.ADD: "+",
    OperationKind.SUB: "-",
    OperationKind.POW: "**",
    OperationKind.MUL: "*",
    OperationKind.DIV: "/",
    OperationKind.IDIV: "//",
    OperationKind.MOD: "%",
    OperationKind.NOT: "not",
    OperationKind.EQ: "==",
    OperationKind.NEQ: "!=",
    OperationKind.LT: "<",
    OperationKind.GT: ">",
    OperationKind.FLIP: "~",
    OperationKind.LTE: "<=",
    OperationKind.GTE: ">=",
    OperationKind.OR: "or",
    OperationKind.AND: "and",
    OperationKind.BITOR: "|",
    OperationKind.BITAND: "&",
    OperationKind.BITXOR: "^",
    OperationKind.LSH: "<<",
    OperationKind.RSH: ">>",
}


class JitCompError(Exception):
    pass


class Gettable:
    def get(self, ident: "parser_types.AstIdentifier") -> "Object":
        raise NotImplementedError()


class FunctionParseState:
    ExpectCommaOrEndOfArgs = 0
    ExpectArgumentOrEndOfArgs = 1


class Function(AstRelated, Resolvable):
    """

    Attributes:
        ast: (lcaml_parser.Ast) the AST that makes up the function
        arguments: the arguments of the function
        bounds: the bounds of the function (functions can bind to values defined when they are defined, then these values go out of scope but the function is still bound to them)

    """

    def __init__(
        self,
        body,
        arguments: List["parser_types.AstIdentifier"],
        bounds: Union[Iterable[str], Dict[str, "Object"]],
        syntax: Syntax = Syntax(),
        _file="<unknown>",
        force_jit=False,
    ):
        self.body = body
        self.arguments = arguments
        self._syntax = syntax
        self.jit_cache = {}
        self._file = _file
        self.force_jit = force_jit
        if isinstance(bounds, dict):
            self.bounds: Dict[str, Object] = bounds
        else:
            self.bounds: Dict[str, Object] = {ident: None for ident in bounds}

    def __str__(self):
        return "Function(" + str(self.body) + ", " + str(self.arguments) + ")"

    def to_python(self):
        name = get_unique_name()
        args = [expect_only_expression(arg.to_python()) for arg in self.arguments]
        body_pre_insert, body_block, body_post_insert = self.body.to_python()
        if COMPILE_WITH_CONTEXT_LEAKING:
            body_block = (
                "\n".join(
                    f'_ad7aaf167f237a94dc2c3ad2["{arg}"] = {arg}'
                    for arg in args + [Syntax._this_intrinsic]
                )
                + "\n"
                + body_block
            )
        function_def = (
            f"def {name}(_ad7aaf167f237a94dc2c3ad2, "
            + ", ".join(args)
            + f", {Syntax._this_intrinsic}):\n{indent(body_block)}\n{name}_self_referral_list = [0]"
        )
        value = (
            "lambda _ad7aaf167f237a94dc2c3ad2, "
            + ", ".join(arg for arg in args)
            + f": {name}("
            + (
                "_ad7aaf167f237a94dc2c3ad2.copy(), "
                if COMPILE_WITH_CONTEXT_LEAKING
                else "{}, "
            )
            + ", ".join(args)
            + f", {name}_self_referral_list[0])"
        )
        populate_self_referral_list = f"{name}_self_referral_list[0] = {value}"
        return (
            body_pre_insert + "\n" + function_def + "\n" + body_post_insert,
            "(" + value + ")",
            populate_self_referral_list,
        )

    @staticmethod
    def _get_type(expr, dtypes: Dict["str", "DType.ty"]) -> "DType.ty":
        if isinstance(expr, Constant):
            return expr.value.type
        elif isinstance(expr, Variable):
            if expr.identifier.name in dtypes:
                return dtypes[expr.identifier.name]
            raise JitCompError(f"usage of undefined variable `{expr.identifier.name}`")
        elif isinstance(expr, Expression):
            return Function._get_type(expr.expression, dtypes)
        elif isinstance(expr, Operation):
            if expr.is_unary:
                dtype_right = Function._get_type(expr.right, dtypes)
                dtype_out = DType._operation_result_rules.get(expr.operation, {}).get(dtype_right)
                if dtype_out is None:
                    raise JitCompError(
                        f"unsupported operation {OPKIND_TO_PYTHON_SYMBOL[expr.operation]} for {DType.name(dtype_right)} (right of unary op)"
                    )
                return dtype_right
            else:
                dtype_left, dtype_right = Function._get_type(expr.left, dtypes), Function._get_type(expr.right, dtypes)
                dtype_out = DType._operation_result_rules.get(expr.operation, {}).get(dtype_left, {}).get(dtype_right)
                if dtype_out is None:
                    raise JitCompError(
                        f"unsupported operation {OPKIND_TO_PYTHON_SYMBOL[expr.operation]} for {DType.name(dtype_left)} (left) and {DType.name(dtype_right)} (right)"
                    )
                return dtype_out
        else:
            raise JitCompError(f"expression of type {type(expr)} cannot be type analyzed")

    @staticmethod
    def _get_ret_type(ast, context, dtypes: Dict["str", "DType.ty"]) -> Optional["DType.ty"]:
        if isinstance(ast, lcaml_parser.Ast):
            types = []
            for stmt in ast.statements:
                types.append(Function._get_ret_type(stmt, context, dtypes))
                # if I generate instr after a terminator instr, that is invalid LLVM IR => stop
                if stmt.type == parser_types.AstStatementType.RETURN:
                    break
            types = [ty for ty in types if ty is not None]
            if types:
                if not all(types[0] == ty for ty in types):
                    raise JitCompError("different branches have different return types")
                return types[0]
        elif isinstance(ast, lcaml_parser.AstStatement):
            if ast.type in (
                parser_types.AstStatementType.ASSIGNMENT,
                parser_types.AstStatementType.RETURN,
                parser_types.AstStatementType.EXPRESSION,
                parser_types.AstStatementType.CONTROL_FLOW,
                parser_types.AstStatementType.WHILE_LOOP,
            ):
                return Function._get_ret_type(ast.value, context, dtypes)
            else:
                raise ValueError("invalid type attribute encountered on AstStatement")
        elif isinstance(ast, parser_types.AstAssignment):
            dtype = Function._get_type(ast.value, dtypes)
            if ast.identifier.name not in dtypes:
                dtypes[ast.identifier.name] = dtype
            if dtypes[ast.identifier.name] != dtype:
                raise JitCompError("different dtypes assigned into same var")
        elif isinstance(ast, parser_types.AstReturn):
            dtype = Function._get_type(ast.value, dtypes)
            return dtype
        elif isinstance(ast, parser_types.AstControlFlow):
            types = []
            for branch in ast.branches:
                types.append(Function._get_ret_type(branch.body, context, dtypes))
            types = [ty for ty in types if ty is not None]
            if not all(types[0] == ty for ty in types):
                raise JitCompError("different if condition branches have different return types")
            if types:
                return types[0]
        elif isinstance(ast, parser_types.AstWhileLoop):
            return Function._get_ret_type(ast.body, context, dtypes)

    @staticmethod
    def _get_llvm_type(t: "DType.ty"):
        if t == DType.INT:
            return llvm_ir.IntType(64)
        elif t == DType.FLOAT:
            return llvm_ir.DoubleType()
        elif t == DType.BOOL:
            return llvm_ir.IntType(1)
        elif t == DType.UNIT:
            return llvm_ir.VoidType()
        else:
            raise JitCompError(f"cannot convert type {DType.name(t)} to llvm type")

    def _compatibility_cast(
        self,
        builder: "llvm_ir.IRBuilder",
        a: "llvm_ir.Value",
        b: "llvm_ir.Value",
        dtype_a: "DType.ty",
        dtype_b: "DType.ty",
        a_llty: "llvm_ir.Type",
        b_llty: "llvm_ir.Type",
    ) -> tuple["llvm_ir.Value", "llvm_ir.Value", "llvm_ir.Type", "DType.ty"]:
        def jitc_err():
            raise JitCompError("cannot upcast")

        hierarchy = [DType.BOOL, DType.INT, DType.FLOAT]
        if dtype_a not in hierarchy:
            raise JitCompError(f"A dtype not supported in the jit compiler was encountered: {DType.name(dtype_a)}")
        if dtype_b not in hierarchy:
            raise JitCompError(f"A dtype not supported in the jit compiler was encountered: {DType.name(dtype_b)}")
        upcast_fns = [lambda x: builder.zext(x, llvm_ir.IntType(64)), lambda x: builder.sitofp(x, llvm_ir.DoubleType()), lambda _: jitc_err()]
        h_a, h_b = hierarchy.index(dtype_a), hierarchy.index(dtype_b)
        out_llty = a_llty if h_a > h_b else b_llty
        out_dtype = dtype_a if h_a > h_b else dtype_b
        while True:
            if h_a < h_b:
                upf = upcast_fns[h_a]
                a = upf(a)
                h_a += 1
            elif h_a > h_b:
                upf = upcast_fns[h_b]
                b = upf(b)
                h_b += 1
            else:
                break
        if h_a > len(hierarchy) or h_b > len(hierarchy):
            raise RuntimeError("lcaml (jitc) internal error: unreachable")
        return a, b, out_llty, out_dtype

    @staticmethod
    def _value_of_type(dtype, value: Any = 0):
        if dtype == DType.FLOAT:
            return float(value)
        elif dtype == DType.INT:
            return int(value)
        elif dtype == DType.BOOL:
            return bool(value)
        elif dtype == DType.UNIT:
            return None
        else:
            raise JitCompError("unsupported dtype for jit compiler")
    
    @staticmethod
    def _insert_zero_check(builder: "llvm_ir.builder.IRBuilder", llvalue, dtype, func_ret_type):
        assert dtype != DType.UNIT
        err_out_param = builder.function.args[-1]
        llty = Function._get_llvm_type(dtype)
        zero = llvm_ir.Constant(llty, Function._value_of_type(dtype))
        div_by_zero_err_code = llvm_ir.Constant(llvm_ir.IntType(64), C_CALL_ERR_CODES[ZeroDivisionError])
        ret_val = (
            llvm_ir.Constant(Function._get_llvm_type(func_ret_type), Function._value_of_type(func_ret_type))
            if func_ret_type != DType.UNIT
            else None
        )
        bb_is_zero = builder.append_basic_block("is_zero")
        bb_is_not_zero = builder.append_basic_block("is_not_zero")

        if dtype == DType.FLOAT:
            cond = builder.fcmp_unordered("!=", llvalue, zero)
        else:
            cond = builder.icmp_signed("!=", llvalue, zero)
        branch_inst = builder.cbranch(cond, bb_is_not_zero, bb_is_zero)
        branch_inst.set_weights([99, 1])

        builder.position_at_end(bb_is_zero)
        builder.store(div_by_zero_err_code, err_out_param)
        if ret_val is not None:
            builder.ret(ret_val)
        else:
            builder.ret_void()
        builder.position_at_end(bb_is_not_zero)

    def _jit_compile(
        self,
        ast,
        builder: "llvm_ir.builder.IRBuilder",
        dtypes: Dict[str, "DType.ty"],
        vars: Dict[str, "llvm_ir.Value"],
        context: "Context",
        bb_allocas: Optional["llvm_ir.Block"] = None,
    ) -> Union[tuple["llvm_ir.Value", "llvm_ir.Type"], tuple[None, None]]:
        if isinstance(ast, lcaml_parser.Ast):
            for stmt in ast.statements:
                self._jit_compile(stmt, builder, dtypes, vars, context, bb_allocas)
                # if I generate instr after a terminator instr, that is invalid LLVM IR => stop
                if stmt.type == parser_types.AstStatementType.RETURN:
                    break
        elif isinstance(ast, lcaml_parser.AstStatement):
            if ast.type in (
                parser_types.AstStatementType.ASSIGNMENT,
                parser_types.AstStatementType.RETURN,
                parser_types.AstStatementType.EXPRESSION,
                parser_types.AstStatementType.CONTROL_FLOW,
                parser_types.AstStatementType.WHILE_LOOP,
            ):
                self._jit_compile(ast.value, builder, dtypes, vars, context, bb_allocas)
            else:
                raise ValueError("invalid type attribute encountered on AstStatement")
        elif isinstance(ast, parser_types.AstAssignment):
            dtype = self._get_type(ast.value, dtypes)
            if ast.identifier.name not in dtypes:
                dtypes[ast.identifier.name] = dtype
            if dtypes[ast.identifier.name] != dtype:
                raise JitCompError("different dtypes assigned into same var")
            value, _ = self._jit_compile(ast.value, builder, dtypes, vars, context, bb_allocas)
            if ast.identifier.name not in vars:
                if bb_allocas:
                    with builder.goto_block(bb_allocas):
                        vars[ast.identifier.name] = builder.alloca(
                            self._get_llvm_type(dtype),
                            name=ast.identifier.name
                        )
                else:
                    vars[ast.identifier.name] = builder.alloca(
                        self._get_llvm_type(dtype),
                        name=ast.identifier.name
                    )
            builder.store(
                value,
                vars[ast.identifier.name],
            )
        elif isinstance(ast, parser_types.AstReturn):
            value, _ = self._jit_compile(ast.value, builder, dtypes, vars, context, bb_allocas)
            builder.ret(value)
        elif isinstance(ast, parser_types.AstExpressionStatement):
            self._jit_compile(ast.expression, builder, dtypes, vars, context, bb_allocas)
        elif isinstance(ast, parser_types.AstControlFlow):
            bbs = [
                (
                    builder.append_basic_block(f"cond{i}"),
                    builder.append_basic_block(f"branch{i}"),
                )
                for i in range(len(ast.branches))
            ]
            post_if = builder.append_basic_block("post_if")
            cond_block_chain = iter([bb_cond for bb_cond, _ in bbs] + [post_if])
            builder.branch(next(cond_block_chain))
            for (bb_cond, bb_body), branch in zip(bbs, ast.branches):
                # build cond
                builder.position_at_end(bb_cond)
                cond, cond_llty = self._jit_compile(
                    branch.condition, builder, dtypes, vars, context, bb_allocas
                )
                # switch instr with default case cond.true and value eq 0 case cond.false (optimized into br instr by llvm)
                sw = builder.switch(cond, bb_body)
                sw.add_case(llvm_ir.Constant(cond_llty, 0), next(cond_block_chain))

                # build body
                builder.position_at_end(bb_body)
                self._jit_compile(branch.body, builder, dtypes, vars, context, bb_allocas)
                builder.branch(post_if)

            builder.position_at_end(post_if)
        elif isinstance(ast, parser_types.AstWhileLoop):
            bb_cond = builder.append_basic_block("cond")
            bb_body = builder.append_basic_block("body")
            bb_post_while = builder.append_basic_block("post_while")
            builder.branch(bb_cond)

            builder.position_at_end(bb_cond)
            cond, cond_llty = self._jit_compile(ast.condition, builder, dtypes, vars, context, bb_allocas)
            sw = builder.switch(cond, bb_body)
            sw.add_case(llvm_ir.Constant(cond_llty, 0), bb_post_while)

            builder.position_at_end(bb_body)
            self._jit_compile(ast.body, builder, dtypes, vars, context, bb_allocas)
            builder.branch(bb_cond)
            builder.position_at_end(bb_post_while)
        elif isinstance(ast, Function):
            raise JitCompError(
                "higher-order functions don't support being jit-compiled"
            )
        elif isinstance(ast, FunctionCall):
            # jit compile inner function
            types = tuple(self._get_type(arg, dtypes) for arg in ast.arguments)
            f_res = ast.function_resolvable
            if isinstance(f_res, Variable):
                fn = context.get(f_res.identifier.name)
                if not fn or not isinstance(fn.value, Function):
                    raise JitCompError(
                        "use of undefined name '"
                        + str(f_res.identifier.name)
                        + "': must execute without jit-compilation"
                    )
                fn = fn.value
            elif isinstance(f_res, Function):
                fn = f_res
            else:
                raise JitCompError(
                    "jit-compiling complex function calling expressions like `(func_list.(2*n + 1))(4)` is not supported"
                )
            fn, ret_ty = fn.jit_compile(builder.module, types, context)
            args = [
                self._jit_compile(arg, builder, dtypes, vars, context, bb_allocas)[0]
                for arg in ast.arguments
            ]
            return builder.call(fn, args), self._get_llvm_type(ret_ty)
        elif isinstance(ast, Constant):
            ty = self._get_llvm_type(ast.value.type)
            return llvm_ir.Constant(ty, ast.value.value), ty
        elif isinstance(ast, Variable):
            if ast.identifier.name not in vars:
                raise JitCompError(
                    "usage of potentially unbound variable '"
                    + ast.identifier.name
                    + "'"
                )
            return builder.load(vars[ast.identifier.name], ast.identifier.name + ".load"), self._get_llvm_type(dtypes[ast.identifier.name])
        elif isinstance(ast, Operation):
            if ast.is_unary:
                common_dtype = self._get_type(ast.right, dtypes)
                b, common_llty = self._jit_compile(ast.right, builder, dtypes, vars, context, bb_allocas)
            else:
                ty_a, ty_b = self._get_type(ast.left, dtypes), self._get_type(
                    ast.right, dtypes
                )
                (a, a_llty), (b, b_llty) = self._jit_compile(
                    ast.left, builder, dtypes, vars, context, bb_allocas
                ), self._jit_compile(ast.right, builder, dtypes, vars, context, bb_allocas)
                a, b, common_llty, common_dtype = self._compatibility_cast(builder, a, b, ty_a, ty_b, a_llty, b_llty)

            if ast.is_unary and ast.operation not in OperationKind._unary:
                raise RuntimeError("invalid op type for unary op")

            if common_dtype in (DType.BOOL, DType.INT):
                if ast.operation == OperationKind.ADD:
                    return builder.add(a, b), common_llty
                elif ast.operation == OperationKind.SUB:
                    return builder.sub(a, b), common_llty
                elif ast.operation == OperationKind.MUL:
                    return builder.mul(a, b), common_llty
                elif ast.operation == OperationKind.DIV:
                    a = builder.sitofp(a, llvm_ir.DoubleType())
                    b = builder.sitofp(b, llvm_ir.DoubleType())
                    self._insert_zero_check(builder, b, DType.FLOAT, dtypes["->"])
                    return builder.fdiv(a, b), llvm_ir.DoubleType()
                elif ast.operation == OperationKind.IDIV:
                    self._insert_zero_check(builder, b, common_dtype, dtypes["->"])
                    div_out = builder.sdiv(a, b)
                    zero = llvm_ir.Constant(common_llty, 0)
                    r = builder.zext(builder.xor(
                        builder.icmp_signed('<', a, zero),
                        builder.icmp_signed('<', b, zero)
                    ), common_llty)
                    return builder.sub(div_out, r), common_llty
                elif ast.operation == OperationKind.MOD:
                    self._insert_zero_check(builder, b, common_dtype, dtypes["->"])
                    return builder.srem(a, b), common_llty
                elif ast.operation == OperationKind.NOT:
                    return builder.icmp_signed(
                        "==", b, llvm_ir.Constant(common_llty, 0)
                    ), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.EQ:
                    return builder.icmp_signed("==", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.NEQ:
                    return builder.icmp_signed("!=", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.LT:
                    return builder.icmp_signed("<", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.GT:
                    return builder.icmp_signed(">", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.FLIP:
                    return builder.not_(b), common_llty
                elif ast.operation == OperationKind.LTE:
                    return builder.icmp_signed("<=", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.GTE:
                    return builder.icmp_signed(">=", a, b), llvm_ir.IntType(1)
                elif (
                    ast.operation == OperationKind.OR
                    or ast.operation == OperationKind.BITOR
                ):
                    return builder.or_(a, b), common_llty
                elif (
                    ast.operation == OperationKind.AND
                    or ast.operation == OperationKind.BITAND
                ):
                    return builder.and_(a, b), common_llty
                elif ast.operation == OperationKind.BITXOR:
                    return builder.xor(a, b), common_llty
                elif ast.operation == OperationKind.LSH:
                    return builder.shl(a, b), common_llty
                elif ast.operation == OperationKind.RSH:
                    return builder.lshr(a, b), common_llty
                elif ast.operation == OperationKind.POW:
                    raise JitCompError("pow jit unsupported at the moment")
                else:
                    raise JitCompError("unsupported operation")

            else:
                if ast.operation == OperationKind.ADD:
                    return builder.fadd(a, b), common_llty
                elif ast.operation == OperationKind.SUB:
                    return builder.fsub(a, b), common_llty
                elif ast.operation == OperationKind.MUL:
                    return builder.fmul(a, b), common_llty
                elif ast.operation == OperationKind.DIV:
                    self._insert_zero_check(builder, b, common_dtype, dtypes["->"])
                    return builder.fdiv(a, b), common_llty
                elif ast.operation == OperationKind.IDIV:
                    self._insert_zero_check(builder, b, common_dtype, dtypes["->"])
                    div_out = builder.fdiv(a, b)
                    r = builder.zext(builder.fcmp_unordered(
                        '<', div_out, llvm_ir.Constant(llvm_ir.DoubleType(), 0.0)
                    ), llvm_ir.IntType(64))
                    floored = builder.sub(builder.fptosi(div_out, llvm_ir.IntType(64)), r)
                    return floored, llvm_ir.IntType(64)
                elif ast.operation == OperationKind.MOD:
                    self._insert_zero_check(builder, b, common_dtype, dtypes["->"])
                    return builder.frem(a, b), common_llty
                elif ast.operation == OperationKind.NOT:
                    return builder.fcmp_unordered(
                        "==", b, llvm_ir.Constant(llvm_ir.DoubleType(), 0.0)
                    ), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.EQ:
                    return builder.fcmp_unordered("==", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.NEQ:
                    return builder.fcmp_unordered("!=", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.LT:
                    return builder.fcmp_unordered("<", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.GT:
                    return builder.fcmp_unordered(">", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.FLIP:
                    raise JitCompError("Cannot bitflip floating-point number")
                elif ast.operation == OperationKind.LTE:
                    return builder.fcmp_unordered("<=", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.GTE:
                    return builder.fcmp_unordered(">=", a, b), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.OR:
                    return builder.or_(
                        builder.fcmp_unordered(
                            "!=", a, llvm_ir.Constant(llvm_ir.DoubleType(), 0.0)
                        ),
                        builder.fcmp_unordered(
                            "!=", b, llvm_ir.Constant(llvm_ir.DoubleType(), 0.0)
                        ),
                    ), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.AND:
                    return builder.and_(
                        builder.fcmp_unordered(
                            "!=", a, llvm_ir.Constant(llvm_ir.DoubleType(), 0.0)
                        ),
                        builder.fcmp_unordered(
                            "!=", b, llvm_ir.Constant(llvm_ir.DoubleType(), 0.0)
                        ),
                    ), llvm_ir.IntType(1)
                elif ast.operation == OperationKind.BITOR:
                    raise JitCompError("cannot bitwise-or floating-point numbers")
                elif ast.operation == OperationKind.BITAND:
                    raise JitCompError("cannot bitwise-and floating-point numbers")
                elif ast.operation == OperationKind.BITXOR:
                    raise JitCompError("cannot bitwise-xor floating-point numbers")
                elif ast.operation == OperationKind.LSH:
                    raise JitCompError("cannot left shift floating-point numbers")
                elif ast.operation == OperationKind.RSH:
                    raise JitCompError("cannot right shift floating-point numbers")
                elif ast.operation == OperationKind.POW:
                    raise JitCompError("pow jit unsupported at the moment")
                else:
                    raise JitCompError("unsupported operation")

        elif isinstance(ast, Expression):
            return self._jit_compile(ast.expression, builder, dtypes, vars, context, bb_allocas)
        else:
            raise RuntimeError(f"unknown ast node type {type(ast)}")
        return None, None

    def jit_compile(
        self,
        module: "llvm_ir.Module",
        input_types: Tuple["DType.ty", ...],
        context: "Context",
        is_main: bool = False,
    ) -> tuple["llvm_ir.Function", "DType.ty"]:
        if llvm_ir is None or llvm_binding is None:
            raise JitCompError("llvmlite not installed")

        allowed_ext_bounds = self.bounds.copy()
        statically_def_vars = self.get_statically_defined_vars()
        for ident in itertools.chain(
            map(lambda i: i.name, self.arguments),
            statically_def_vars
        ):
            if ident in allowed_ext_bounds:
                allowed_ext_bounds.pop(ident)

        illegal_arg_types = (
            DType.FUNCTION,
            DType.STRUCT_TYPE,
            DType.EXTERN_PYTHON,
            DType.PY_OBJ,
            DType.TABLE,
            DType.STRING,
            DType.LIST,  # TODO maybe support this
            DType.UNIT,  # this too?
        )
        # TODO I should try to support function bounds
        # illegal_bound_types = (
        #     DType.FUNCTION,
        #     DType.STRUCT_TYPE,
        #     DType.EXTERN_PYTHON,
        #     DType.PY_OBJ,
        #     DType.TABLE,
        #     DType.STRING,
        #     DType.LIST,
        # )
        illegal_bound_types = (PhantomType(),)
        # pyffi makes this complicated or maybe impossible; for now, I will assume here that pyffi functions do not modify the ast of lcaml functions.
        # However, this may not be true. TODO can I come up with some way to avoid this problem that does not require me to hash the entire ast recursively?
        for k, v in allowed_ext_bounds.items():
            if v is None:
                raise JitCompError(
                    f"cannot jit-compile functions with unresolved bounds: found unresolved bound {k}"
                )
            elif v.type in illegal_bound_types:
                raise JitCompError(
                    f"jit-compiling functions with bounds of type {DType.name(v.type)} is unsupported: found bound {k} of illegal type"
                )

        allowed_bounds_dtypes = {
            n: obj.type for n, obj in allowed_ext_bounds.items()
        }
        dtypes = {
            arg: dtype
            for arg, dtype in itertools.chain(
                zip((arg.name for arg in self.arguments), input_types),
                allowed_bounds_dtypes.items(),
            )
        }
        for dt in dtypes.values():
            if dt in illegal_arg_types:
                raise JitCompError(
                    f"compiling functions with non-primitive input parameter types is unsupported: got input of type {DType.name(dt)}"
                )

        arg_llvm_types = [
            self._get_llvm_type(dtype)
            for dtype in input_types
        ] + [llvm_ir.PointerType(llvm_ir.IntType(64))]
        ret_type = Function._get_ret_type(self.body, context, dtypes)
        must_insert_ret_void = False
        if ret_type is None:
            must_insert_ret_void = True
            ret_type = DType.UNIT
        dtypes["->"] = ret_type
        ret_llvm_type = self._get_llvm_type(ret_type)
        func_ty = llvm_ir.FunctionType(ret_llvm_type, arg_llvm_types)
        func = llvm_ir.Function(module, func_ty, "main" if is_main else "")
        bb_allocas = func.append_basic_block("allocas")
        builder = llvm_ir.IRBuilder(bb_allocas)

        vars = {
            arg.name: builder.alloca(func.args[i].type, name=arg.name)
            for i, arg in enumerate(self.arguments)
        }

        bb_entry = builder.append_basic_block("entry")
        builder.position_at_end(bb_entry)

        for v, alloca in zip(func.args, vars.values()):
            builder.store(v, alloca)
        try:
            self._jit_compile(self.body, builder, dtypes, vars, context, bb_allocas)
            if must_insert_ret_void:
                builder.ret_void()
            builder.position_at_end(bb_allocas)
            builder.branch(bb_entry)
        except JitCompError as e:
            # try to decouple the failed function from the module
            module.globals.pop(func.name)
            raise e

        return func, ret_type

    def jit_compile_main(
        self,
        input_types: Tuple["DType.ty"],
        context: "Context",
    ) -> tuple[int, "DType.ty"]:
        if input_types not in self.jit_cache:
            try:
                ir_mod = llvm_ir.Module()
                main_func, ret_type = self.jit_compile(
                    ir_mod, input_types, context, is_main=True
                )
                # convert the IR mod to a binding mod (yes, emitting and re-parsing LLVM IR is the recommended way of doing that as per documentation)
                func_ptr = compile_llvm_module(ir_mod, main_func.name)
            except Exception as e:
                self.jit_cache[input_types] = e
            else:
                self.jit_cache[input_types] = func_ptr, ret_type

        ret = self.jit_cache[input_types]
        if isinstance(ret, Exception):
            raise ret
        return ret

    def get_statically_defined_vars(self) -> List[str]:
        # TODO I could do advanced CF analysis here, but for now, I'll just return those vars defined at the top of the file
        out = []
        for ast in self.body.statements:
            if ast.type == parser_types.AstStatementType.ASSIGNMENT:
                out.append(ast.value.identifier.name)
            else:
                break
        return out

    def resolve(self, context: Context) -> Object:
        # use context to resolve bounds
        intersecting_keys = self.bounds.keys() & context.keys()
        for key in intersecting_keys:
            self.bounds[key] = context[key]
        ret = Object(DType.FUNCTION, self)
        if self._syntax._this_intrinsic in self.bounds:
            self.bounds[self._syntax._this_intrinsic] = ret
        return ret

    @classmethod
    def from_stream(
        cls,
        stream: TokenStream,
        syntax: Syntax = Syntax(),
    ):
        """
        Builds function from stream

        Args:
            stream: stream to build from

        Returns:
            Function: function built from stream

        Raises:
            ValueError: Syntax error or interpreter bug
        """
        # check for function argument token
        # this token contains a bunch of function arguments
        # token.value = "|x y z|"
        # first, extract arguments
        first_token = stream.pop(0)
        if first_token.type != TokenKind.FUNCTION_ARGS:
            raise ValueError(
                f"Expected function arguments, got {first_token.type} instead"
            )

        identifiers_raw = map(str.strip, syntax._extract_fn_args(first_token.value))
        arguments = list(
            map(
                lambda raw_id: parser_types.AstIdentifier(
                    Token(TokenKind.IDENTIFIER, raw_id)
                ),
                identifiers_raw,
            )
        )
        # consume function body
        first_body_token = stream.pop(0)
        if first_body_token.type != TokenKind.LCURLY:
            raise ValueError(
                f"Expected lcurly for function body, got {first_body_token.type}"
            )

        body_stream, remaining_stream = split_at_context_end(
            stream, Token(TokenKind.RCURLY, PhantomType())
        )
        remaining_stream.pop(0)  # remove RCURLY
        body, symbols_used = lcaml_parser.Ast.from_stream(body_stream, syntax)

        source_file = getattr(stream, "__file") if hasattr(stream, "__file") else "<unknown>"
        return (
            cls(
                body,
                arguments,
                map(lambda variable: variable.identifier.name, symbols_used),
                syntax,
                source_file,
            ),
            remaining_stream,
            symbols_used,
        )


class Operation(AstRelated, Resolvable):
    def __init__(self, left: Optional[Resolvable], operation: Token, right: Resolvable):
        optype = SYMBOL_TO_OPKIND.get(operation.value)
        if optype is None:
            raise ValueError(f"Unknown operation {operation.value}")
        self.left = left
        self.operation = optype
        self.right = right
        self.is_unary = optype in [OperationKind.FLIP, OperationKind.NOT]

    def __str__(self):
        return (
            "Operation("
            + str(self.left)
            + ", "
            + str(self.operation)
            + ", "
            + str(self.right)
            + ")"
        )

    def to_python(self):
        if self.is_unary:
            pre_insert, expr, post_insert = self.right.to_python()
            return (
                pre_insert,
                f"{OPKIND_TO_PYTHON_SYMBOL[self.operation]}({expr})",
                post_insert,
            )
        else:
            assert self.left is not None
            pre_inserts, exprs, post_inserts = zip(
                *[value.to_python() for value in (self.left, self.right)]
            )
            pre_insert = "\n".join(pre_inserts)
            post_insert = "\n".join(post_inserts)
            left, right = exprs
            return (
                pre_insert,
                f"({left} {OPKIND_TO_PYTHON_SYMBOL[self.operation]} {right})",
                post_insert,
            )

    def resolve(self, context: Context) -> Object:
        """
        This function evaluates the operation for values of left and right
        """
        if self.left is None:
            left: Object = None  # typehint so the lsp doesnt spam me with warnings
        else:
            left: Object = self.left.resolve(context)

        right = self.right.resolve(context)
        if left is None and self.operation not in OperationKind._unary:
            raise ValueError("Left operand must not be None")

        if self.operation == OperationKind.ADD:
            return left.add(right)
        elif self.operation == OperationKind.SUB:
            return left.sub(right)
        elif self.operation == OperationKind.MUL:
            return left.mul(right)
        elif self.operation == OperationKind.DIV:
            return left.div(right)
        elif self.operation == OperationKind.IDIV:
            return left.idiv(right)
        elif self.operation == OperationKind.POW:
            return left.pow(right)
        elif self.operation == OperationKind.MOD:
            return left.mod(right)
        elif self.operation == OperationKind.NOT:
            return right.bool_not()
        elif self.operation == OperationKind.EQ:
            return left.eq(right)
        elif self.operation == OperationKind.NEQ:
            return left.neq(right)
        elif self.operation == OperationKind.LT:
            return left.lt(right)
        elif self.operation == OperationKind.GT:
            return left.gt(right)
        elif self.operation == OperationKind.FLIP:
            return right.flip()
        elif self.operation == OperationKind.LTE:
            return left.lte(right)
        elif self.operation == OperationKind.GTE:
            return left.gte(right)
        elif self.operation == OperationKind.OR:
            return left.bool_or(right)
        elif self.operation == OperationKind.AND:
            return left.bool_and(right)
        elif self.operation == OperationKind.BITOR:
            return left.bitor(right)
        elif self.operation == OperationKind.BITAND:
            return left.bitand(right)
        elif self.operation == OperationKind.BITXOR:
            return left.bitxor(right)
        elif self.operation == OperationKind.RSH:
            return left.rsh(right)
        elif self.operation == OperationKind.LSH:
            return left.lsh(right)
        else:
            raise ValueError(f"Unknown operation type {self.operation}")


class StructTypeParseState:
    ExpectFieldOrEnd = 0
    ExpectCommaOrEnd = 1


class StructType(AstRelated, Resolvable):
    """

    Attributes:
        fields: the names of the fields of a struct

    """

    def __init__(self, fields: List["parser_types.AstIdentifier"]):
        self.fields = fields

    def __str__(self):
        return f"StructType({self.fields})"

    def to_python(self):
        return (
            "",
            "set(["
            + ", ".join(
                expect_only_expression(field.to_python()) for field in self.fields
            )
            + "])",
            "",
        )

    def resolve(self, context: Context):
        return Object(DType.STRUCT_TYPE, self)

    @classmethod
    def from_stream(cls, stream: TokenStream):
        token = stream.pop(0)
        if token.type != TokenKind.STRUCT:
            raise ValueError(f"expected struct keyword, got {token}")
        token = stream.pop(0)
        if token.type != TokenKind.LCURLY:
            raise ValueError(f"expected lcurly, got {token}")

        state = StructTypeParseState.ExpectFieldOrEnd
        fields: List["parser_types.AstIdentifier"] = []
        while stream:
            token = stream.pop(0)
            if state == StructTypeParseState.ExpectFieldOrEnd:
                if token.type == TokenKind.IDENTIFIER:
                    field = parser_types.AstIdentifier(token)
                    fields.append(field)
                    state = StructTypeParseState.ExpectCommaOrEnd
                elif token.type == TokenKind.RCURLY:
                    break
                else:
                    raise ValueError(f"Expected field, got {token}")
            elif state == StructTypeParseState.ExpectCommaOrEnd:
                if token.type == TokenKind.COMMA:
                    state = StructTypeParseState.ExpectFieldOrEnd
                elif token.type == TokenKind.RCURLY:
                    break
                else:
                    raise ValueError(f"expected comma, got {token}")
            else:
                raise RuntimeError(
                    f"Internal Bug: invalid state reached in parsing of struct definition (state code: {state})"
                )
        else:
            raise ValueError("Unexpected end of tokenstream")

        return cls(fields), stream, set()


class ListParseState:
    ExpectExpressionOrEnd = 0
    ExpectCommaOrEnd = 1


class LList(AstRelated, Resolvable):
    """

    Attributes:
        type: currently not in use
        fields: the fields of the struct instance

    """

    def __init__(self, values: List[Resolvable], type: Optional[Resolvable] = None):
        self.type = type
        self.values = values

    def __str__(self):
        return f"List({self.values})"

    def to_python(self):
        pre_inserts, exprs, post_inserts = zip(
            *[value.to_python() for value in self.values]
        )
        pre_insert = "\n".join(pre_inserts)
        post_insert = "\n".join(post_inserts)
        return pre_insert, "[" + ", ".join(exprs) + "]", post_insert

    def resolve(self, context: Context):
        for i, expression in enumerate(self.values):
            self.values[i] = expression.resolve(context)
        return Object(DType.LIST, self)

    @classmethod
    def from_stream(cls, stream: TokenStream, syntax: Syntax = Syntax()):
        token = stream.pop(0)
        if token.type != TokenKind.LSQUARE:
            raise ValueError(f"expected lsquare, got {token}")
        state = ListParseState.ExpectExpressionOrEnd
        all_symbols_used = set()
        values = []
        # expression after : can either end with a } or a , and both cases should be handled
        _expression_terminating_token = Token(
            EqualsAny(TokenKind.COMMA, TokenKind.RSQUARE), PhantomType()
        )
        while stream:
            token = stream.pop(0)
            if token.type == TokenKind.RSQUARE:
                break
            if state == ListParseState.ExpectExpressionOrEnd:
                stream.insert(0, token)
                expression, stream, symbols_used = Expression.from_stream(
                    stream, syntax, terminating_token=_expression_terminating_token
                )
                all_symbols_used.update(symbols_used)
                values.append(expression)
                state = ListParseState.ExpectCommaOrEnd
            elif state == ListParseState.ExpectCommaOrEnd:
                if token.type == TokenKind.COMMA:
                    state = ListParseState.ExpectExpressionOrEnd
                elif token.type == TokenKind.RSQUARE:
                    break
                else:
                    raise ValueError(f"expected comma or rcurly, got {token}")
                state = ListParseState.ExpectExpressionOrEnd
        return cls(values), stream, all_symbols_used


class TableParseState:
    ExpectFieldOrEnd = 0
    ExpectColon = 1
    ExpectExpression = 2
    ExpectCommaOrEnd = 3


class Table(AstRelated, Resolvable, Gettable):
    """

    Attributes:
        type: currently not in use
        fields: the fields of the struct instance

    """

    def __init__(
        self, fields: Dict[Any, Resolvable], type: Optional[Resolvable] = None
    ):
        self.type = type
        self.fields = fields

    def __str__(self):
        return f"Table({self.fields})"

    def to_python(self):
        keys = self.fields.keys()
        pre_inserts, exprs, post_inserts = zip(
            *[value.to_python() for value in self.fields.values()]
        )
        pre_insert = "\n".join(pre_inserts)
        post_insert = "\n".join(post_inserts)
        return (
            pre_insert,
            "{"
            + ", ".join(f'"{key}": ' + expr for key, expr in zip(keys, exprs))
            + "}",
            post_insert,
        )

    def resolve(self, context: Context):
        for field, expression in self.fields.items():
            self.fields[field] = expression.resolve(context)
        return Object(DType.TABLE, self)

    def get(self, ident: "parser_types.AstIdentifier") -> Object:
        if not isinstance(ident, parser_types.AstIdentifier):
            raise TypeError(f"Expected parser_types.AstIdentifier, got {ident}")
        if ident.name not in self.fields:
            raise ValueError(f"Field {ident.name} not found in struct {self}")
        return self.fields[ident.name]

    @classmethod
    def from_stream(cls, stream: TokenStream, syntax: Syntax = Syntax()):
        token = stream.pop(0)
        if token.type != TokenKind.LCURLY:
            raise ValueError(f"expected lcurly, got {token}")
        state = TableParseState.ExpectFieldOrEnd
        field = None
        all_symbols_used = set()
        assignments = []
        # expression after : can either end with a } or a , and both cases should be handled
        _expression_terminating_token = Token(
            EqualsAny(TokenKind.COMMA, TokenKind.RCURLY), PhantomType()
        )
        while stream:
            token = stream.pop(0)
            # (or does it?) FIXME: this does not work for stacked structures (struct inside struct)
            if token.type == TokenKind.RCURLY:
                break
            if state == TableParseState.ExpectFieldOrEnd:
                if token.type != TokenKind.IDENTIFIER:
                    raise ValueError(f"expected colon, got {token}")
                field = parser_types.AstIdentifier(token)
                state = TableParseState.ExpectColon
            elif state == TableParseState.ExpectColon:
                if token.type != TokenKind.COLON:
                    raise ValueError(f"expected colon, got {token}")
                state = TableParseState.ExpectExpression
            elif state == TableParseState.ExpectExpression:
                if field is None:
                    raise ValueError("Interal Error or Syntax Error: Field is None")
                stream.insert(0, token)
                expression, stream, symbols_used = Expression.from_stream(
                    stream, syntax, terminating_token=_expression_terminating_token
                )
                all_symbols_used.update(symbols_used)
                assignments.append((field, expression))
                field = None
                state = TableParseState.ExpectCommaOrEnd
            elif state == TableParseState.ExpectCommaOrEnd:
                if token.type == TokenKind.COMMA:
                    state = TableParseState.ExpectFieldOrEnd
                elif token.type == TokenKind.RCURLY:
                    break
                else:
                    raise ValueError(f"expected comma or rcurly, got {token}")
                state = TableParseState.ExpectFieldOrEnd
        if state not in (
            TableParseState.ExpectCommaOrEnd,
            TableParseState.ExpectFieldOrEnd,
        ):
            raise ValueError("Unexpected end of tokenstream")
        assignments = dict(assignments)
        assignments = {
            field.name: expression for field, expression in assignments.items()
        }
        return cls(assignments), stream, all_symbols_used


class FieldAccess(AstRelated, Resolvable):
    """
    A class that represents a field access to a value. Can be stacked by having a FieldAccess as the object field of another FieldAccess

    Attributes:
        object: the resolvable (expression, variable, whatever) to access field from
        field: the field to look for and return

    """

    def __init__(self, object: "Resolvable", field: "parser_types.AstIdentifier"):
        """

        Args:
            object: the resolvable (expression, variable, whatever) to access field from
            field: the field to look for and return

        """
        self.object = object
        self.field = field

    def __str__(self):
        return "AstFieldAccess(" + str(self.object) + ", " + str(self.field) + ")"

    def to_python(self) -> tuple[str, str, str]:
        return (
            "",
            f'({expect_only_expression(self.object.to_python())}["{expect_only_expression(self.field.to_python())}"])',
            "",
        )

    def resolve(self, context: Context) -> Object:
        obj = self.object.resolve(context)
        if isinstance(
            obj, (Gettable, Object)
        ):  # Object is fine if wrapped value is gettable
            return obj.get(self.field)
        else:
            raise TypeError(
                f"Internal Error: Cannot access field {self.field} on non-struct/table {obj}"
            )


class FunctionCall(AstRelated, Resolvable):
    """

    Attributes:
        function_resolvable: (Resolvable[Function]) function to call
        arguments: (List[Expression]) arguments to call function with

    """

    def __init__(
        self,
        fuction_resolvable: Resolvable,
        arguments: List["Expression"],
        syntax: Syntax = Syntax(),
    ):
        """
        Resolved by spawning a new interpreter_vm_mod.InterpreterVM

        Args:
            function_container: Object[Function]
            arguments: List[Expression]
        """
        self.function_resolvable = fuction_resolvable
        self.arguments = arguments
        self._syntax = syntax

    def __str__(self):
        return (
            "FunctionCall("
            + str(self.function_resolvable)
            + ", "
            + str(self.arguments)
            + ")"
        )

    def to_python(self) -> tuple[str, str, str]:
        f_pre_insert, f_expr, f_post_insert = self.function_resolvable.to_python()
        arg_pre_inserts, arg_exprs, arg_post_inserts = zip(
            *[arg.to_python() for arg in self.arguments]
        )
        pre_insert = "\n".join((*arg_pre_inserts, f_pre_insert))
        post_insert = "\n".join((f_post_insert, *arg_post_inserts))

        f_expr = (
            f"({f_expr} if _4a11dbf5131539804348ceb5({f_expr}, _8f43c264c756af91f5eff200) else _8f43c264c756af91f5eff200({f_expr}))("
            + (
                "_ad7aaf167f237a94dc2c3ad2, "
                if COMPILE_WITH_CONTEXT_LEAKING
                else "{}, "
            )
            + ", ".join(arg_expr for arg_expr in arg_exprs)
            + ")"
        )
        return (
            pre_insert,
            "(" + f_expr + ")",
            post_insert,
        )

    @staticmethod
    def to_c_type(ty: "DType.ty"):
        if ty == DType.FLOAT:
            return c_double
        elif ty == DType.INT:
            return c_int64
        elif ty == DType.BOOL:
            return c_bool
        else:
            raise TypeError("cannot convert to ctype")

    @staticmethod
    def c_call(func, args, out_type):
        # Run the function via ctypes
        cfunc = CFUNCTYPE(
            None if out_type == DType.UNIT else FunctionCall.to_c_type(out_type),
            *([FunctionCall.to_c_type(arg.type) for arg in args] + [POINTER(c_int64)]),
        )(func)
        err_out_param = c_int64()
        res = cfunc(*(pyffi._lcaml_to_python(arg) for arg in args), c_byref(err_out_param))
        ec = err_out_param.value
        exc = C_CALL_ERR_EXCEPTIONS.get(ec)
        if exc is not None:
            e = exc()
            if not hasattr(e, "__lcaml_traceback_info"):
                setattr(e, "__lcaml_traceback_info", [])
            getattr(e, "__lcaml_traceback_info").append(
                "Exception was raised inside of JIT-compiled function and no traceback could be generated. To get a full one, try disabling the JIT compiler."
            )
            raise e
        return pyffi._python_to_lcaml(res)

    def resolve(self, context: Context) -> Optional[Object]:
        """
        Resolves the value of the function call by spawning a new interpreter_vm_mod.InterpreterVM

        Args:
            context: context to resolve function in

        Returns:
            Object: return value of function call

        Raises:
            TypeError: Cannot call non-function
        """
        function = self.function_resolvable.resolve(context)
        if isinstance(function, Object):
            function = function.value

        # resolve args and make arg locals
        resolved_args = [arg.resolve(context) for arg in self.arguments]

        if isinstance(function, Function):
            arg_locals = zip((arg.name for arg in function.arguments), resolved_args)

            if len(resolved_args) < len(function.arguments):
                # not all args provided -> return curried function
                # remove first n elements (curried away)
                remaining_args = function.arguments[len(resolved_args) :]
                # add curried args to bounds
                bounds = function.bounds.copy()
                bounds.update(arg_locals)
                result = Function(function.body, remaining_args, bounds, self._syntax, function._file)
                return Object(DType.FUNCTION, result)
            else:
                # all args provided -> execute function
                # create local context
                local_context = context.copy()
                # functions can bind to global values
                bounds = {k: v for k, v in function.bounds.items() if v is not None}
                # overwrite variables from outer context with local args
                bounds.update(arg_locals)
                # add call-time bounds (arguments) and creation-time bounds
                local_context.update(bounds)

                # try to JIT compile function
                if not SUPPRESS_JIT and (function.force_jit or JIT_BY_DEFAULT) and "llvm_ir" in globals() and "llvm_binding" in globals():
                    try:
                        # TODO maybe try to pass in bounds as parameters?
                        arg_types = tuple(arg.type for arg in resolved_args)
                        func_ptr, ret_type = function.jit_compile_main(
                            arg_types, local_context
                        )
                        result = FunctionCall.c_call(
                            func_ptr, list(bounds.values()), ret_type
                        )
                        return result
                    except JitCompError as e:
                        if function.force_jit:
                            raise e
                elif not SUPPRESS_JIT and function.force_jit:
                    raise RuntimeError("llvmlite not installed: cannot use jit compiler")

                # spawn new interpreter vm
                base_vm_obj = context.get(Syntax._vm_intrinsic)
                if base_vm_obj is None or base_vm_obj.type != DType.PY_OBJ or not isinstance(base_vm_obj.value, interpreter_vm_mod.InterpreterVM):
                    raise RuntimeError(f"{Syntax._vm_intrinsic} does not store exist or does not store the current vm")
                base_vm: "interpreter_vm_mod.InterpreterVM" = base_vm_obj.value
                interpreter_vm = interpreter_vm_mod.InterpreterVM(
                    function.body,
                    local_context,
                    base_vm,
                    base_vm.line_callbacks,
                    base_vm.next_step_callbacks,
                    function._file,
                    _causes_traceback_entry=True,
                    _enable_vm_callbacks=base_vm._enable_vm_callbacks,
                )
                interpreter_vm.execute()
                result = interpreter_vm.return_value
                return result if result is not None else Object(DType.UNIT, None)

        elif isinstance(function, extern_python.ExternPython):
            return function.execute(context.copy(), resolved_args)

        elif hasattr(function, "execute"):
            print(
                "Warning: unregistered object type executed (function not registered as ExternPython, but has necessary API). "
                "Please report to developer."
            )
            return function.__class__.execute(function, context.copy(), resolved_args)

        else:
            raise TypeError("Cannot call non-function")


class Variable(AstRelated, Resolvable):
    def __init__(self, identifier: "parser_types.AstIdentifier"):
        self.identifier = identifier

    def __str__(self):
        return "Variable(" + str(self.identifier) + ")"

    def to_python(self):
        ident = expect_only_expression(self.identifier.to_python())
        return "", f'(_ad7aaf167f237a94dc2c3ad2["{ident}"])', ""

    def resolve(self, context: Context):
        result = context.get(self.identifier.name)
        if result is None:
            raise RuntimeError(f"{self.identifier} is undefined")
        return result


class Constant(AstRelated, Resolvable):
    """

    Attributes:
        value: value of the constant (int, float, str, bool, ...)
        type: type of the constant (DType)

    """

    def __init__(self, token: Token, syntax: Syntax = Syntax()):
        if token.type == TokenKind.UNIT_TYPE:
            value = None
            kind = DType.UNIT
        elif token.type == TokenKind.INTEGER:
            try:
                value = int(token.value)
                kind = DType.INT
            except ValueError:
                raise ValueError(f"Invalid integer {token.value}")
        elif token.type == TokenKind.FLOATING_POINT:
            try:
                value = float(token.value)
                kind = DType.FLOAT
            except ValueError:
                raise ValueError(f"Invalid floating point {token.value}")
        elif token.type == TokenKind.STRING_LITERAL:
            value = token.value
            kind = DType.STRING
        elif token.type == TokenKind.BOOLEAN:
            try:
                value = syntax._true == token.value
                kind = DType.BOOL
            except ValueError:
                raise ValueError(f"Invalid boolean {token.value}")
        else:
            raise ValueError(f"Invalid type for constant: {token.type}")

        self.value = Object(kind, value)

    def __str__(self):
        return "Constant(" + str(self.value) + ")"

    def to_python(self):
        if self.value.type == DType.STRING:
            value = '"' + str(self.value.value) + '"'
        elif self.value.type == DType.UNIT:
            value = "None"
        elif self.value.type == DType.BOOL:
            value = str(self.value.value).title()
        else:
            value = str(self.value.value)
        return "", value, ""

    def resolve(self, context: Context):
        return self.value


class ObjectFakeAst(AstRelated, Resolvable):
    """
    Helper class to wrap an object in an AST node interface so I don't have to rewrite functions that require this API
    """

    def __init__(self, obj: "Object"):
        self.obj = obj

    def __str__(self):
        return "ObjectFakeAst(" + str(self.obj) + ")"

    def to_python(self):
        raise RuntimeError("unreachable")

    def resolve(self, context: Context):
        return self.obj


class ExpressionBuildState:
    EXPECT_RESOLVABLE = 0
    EXPECT_OPERATOR = 1


class Expression(AstRelated, Resolvable):
    """

    Attributes:
        expression: TokenStream of expression

    """

    def __init__(self, expression: Resolvable):
        self.expression = expression

    def __str__(self):
        return "Expression(" + str(self.expression) + ")"

    def to_python(self):
        return self.expression.to_python()

    def resolve(self, context: Context):
        """
        This function resolves the value of the expression.
        """
        return self.expression.resolve(context)

    @classmethod
    def _build_from(cls, stream: TokenStream, syntax: Syntax = Syntax()):
        """
        Build from stream raw (expects stream to not contain any other tokens that do not belong to expression)
        This function does a total of 7 passes across the data:
            1. Make operations without parameters and parse tokens into constants, variables or Exressions
            2. Identify function calls
            (3.x) Fill out operations with their args:
                3.1 . (field access)
                3.2. ! ~ - (unary ops)
                3.3 binary ops according to precedence table (hardcoded in function, really easy to find in source code)

        Args:
            stream: stream (only of Expression) to build from

        Raises:
            ValueError: Syntax error or interpreter bug

        Returns:
            Expression: Expression object built from stream

        """
        all_symbols_used: Set[Variable] = set()
        first_pass_buffer = []
        # first pass across data: make operations without parameters and parse tokens into constants, variables or Exressions
        # NOTE: this pass will not detect function calls
        while stream:
            # NOTE: inplace pop is fine because this is only called on expression_stream
            token = stream.pop(0)

            if token.type == TokenKind.OPERATOR:
                op = Operation(None, token, None)
                first_pass_buffer.append(op)
            elif token.type == TokenKind.DOT:
                fa = FieldAccess(None, None)  # reuse logic from Operation handling
                first_pass_buffer.append(fa)
            elif token.type in TokenKind._builtin_types:
                first_pass_buffer.append(Constant(token, syntax))
            elif token.type == TokenKind.IDENTIFIER:
                first_pass_buffer.append(Variable(parser_types.AstIdentifier(token)))
            elif token.type == TokenKind.LPAREN:
                expression, stream, symbols_used = cls.from_stream(
                    stream, syntax, Token(TokenKind.RPAREN, PhantomType())
                )
                first_pass_buffer.append(expression)
                stream.pop(0)  # remove RPAREN
                all_symbols_used.update(symbols_used)
            else:
                raise ValueError(f"Unexpected token {token}")

        # second pass: identify function calls
        FUNCTION_CALL_ALLOWED_TYPES = (Constant, Variable, cls)
        second_pass_buffer = []
        # NOTE: will consume tokens from first_pass_buffer (by popping)
        while first_pass_buffer:
            # can be operation, constant, variable or expression
            thing = first_pass_buffer.pop(0)

            if isinstance(thing, Variable):  # add all symbols to all_symbols_used
                all_symbols_used.add(thing)

            if type(thing) in (Variable, cls):
                # might be function call
                # check the following token(s)
                if not first_pass_buffer:
                    # EOS
                    second_pass_buffer.append(thing)
                    break
                next_thing = first_pass_buffer[0]
                if type(next_thing) in FUNCTION_CALL_ALLOWED_TYPES:
                    # function call
                    function_call = FunctionCall(thing, [], syntax)
                    while (
                        first_pass_buffer
                        and type(first_pass_buffer[0]) in FUNCTION_CALL_ALLOWED_TYPES
                    ):  # NOTE: safe to run because python interpreter will short circuit type(first_pass_buffer[0]) part
                        # consume all arguments
                        next_arg = first_pass_buffer.pop(0)
                        function_call.arguments.append(next_arg)
                    second_pass_buffer.append(function_call)

                else:
                    # not function call
                    second_pass_buffer.append(thing)

            else:
                second_pass_buffer.append(thing)

        this_pass_buffer, prev_pass_buffer = [], second_pass_buffer
        # third set of passes: fill out operations with their args
        # NOTE: this is a sequence of related passes because ! ~ have to be done before * / %
        # NOTE: and * / % have to be done before + -
        # NOTE and + - have to be done before | & which have to be done before == != < > <= >=
        # NOTE: which have to be done before || &&

        while prev_pass_buffer:
            thing = prev_pass_buffer.pop(0)

            if isinstance(thing, FieldAccess):
                # NOTE: binary operands are always on the left and right
                if not prev_pass_buffer:
                    raise ValueError(
                        "Binary operand must have left and right operand, but doesn't have right"
                    )
                if not this_pass_buffer:
                    raise ValueError(
                        "Binary operand must have left and right operand, but doesn't have left"
                    )
                left = this_pass_buffer.pop()
                right = prev_pass_buffer.pop(0)
                # if variable, unwrap to raw identifier
                if isinstance(right, Variable):
                    right = right.identifier
                thing.object = left
                thing.field = right

            this_pass_buffer.append(thing)
        # stage one of third pass (unary): ! ~ -
        this_pass_buffer, prev_pass_buffer = [], this_pass_buffer
        while prev_pass_buffer:
            thing = prev_pass_buffer.pop(0)

            if isinstance(thing, Operation):
                is_unary_minus = thing.operation == OperationKind.SUB and not this_pass_buffer
                if thing.is_unary or is_unary_minus:
                    # NOTE: unary operands are always on the right
                    if not prev_pass_buffer:
                        raise ValueError("Unary operand must have right operand")
                    right = prev_pass_buffer.pop(0)
                    thing.right = right  # thing is unary operation
                    if is_unary_minus:
                        thing.left = Constant(Token(TokenKind.INTEGER, "0"))

            this_pass_buffer.append(thing)

        # all the other passes (binary) are kind of the same
        sorted_pass_operations = (
            (OperationKind.POW,),
            (
                OperationKind.MUL,
                OperationKind.DIV,
                OperationKind.IDIV,
                OperationKind.MOD,
            ),
            (
                OperationKind.ADD,
                OperationKind.SUB,
            ),
            (
                OperationKind.LSH,
                OperationKind.RSH,
            ),
            (
                OperationKind.BITAND,
            ),
            (
                OperationKind.BITXOR,
            ),
            (
                OperationKind.BITOR,
            ),
            (
                OperationKind.EQ,
                OperationKind.NEQ,
                OperationKind.LT,
                OperationKind.GT,
                OperationKind.LTE,
                OperationKind.GTE,
            ),
            (
                OperationKind.AND,
            ),
            (
                OperationKind.OR,
            ),
        )
        for pass_operations in sorted_pass_operations:
            # use prev_pass_buffer to store previous pass
            this_pass_buffer, prev_pass_buffer = [], this_pass_buffer

            while prev_pass_buffer:
                thing = prev_pass_buffer.pop(0)

                if isinstance(thing, Operation):
                    if thing.operation in pass_operations:
                        # NOTE: binary operands are always on the left and right
                        if not prev_pass_buffer:
                            raise ValueError(
                                "Binary operand must have left and right operand, but doesn't have right"
                            )
                        if not this_pass_buffer:
                            raise ValueError(
                                "Binary operand must have left and right operand, but doesn't have left"
                            )
                        left = this_pass_buffer.pop()
                        right = prev_pass_buffer.pop(0)
                        thing.left = left
                        thing.right = right

                this_pass_buffer.append(thing)

        if len(this_pass_buffer) != 1:
            raise ValueError(
                "Error while parsing expression: Syntax error or interpreter bug"
            )
        expression = this_pass_buffer[0]
        return cls(expression), all_symbols_used

    @classmethod
    def from_stream(
        cls,
        stream: TokenStream,
        syntax: Syntax = Syntax(),
        terminating_token: Token = Token(
            TokenKind.SEMICOLON, PhantomType()
        ),  # use PhantomType so it matches anything of that type
    ):
        """

        Args:
            stream: TokenStream to parse
            terminating_token: Token to terminate expression with

        Raises:
            ValueError: Semicolon not found (no end of expression found)

        Returns:
            Expression: AstExpression object built from tokenstream
            Stream: Remaining tokenstream
        """
        if not stream:
            raise ValueError("Empty stream")
        elif stream[0].type == TokenKind.FUNCTION_ARGS:
            # expression is a function (because function is a full expression and cannot be paired with other things in one expression, for that, you need subexpressions)
            function, remaining_stream, symbols_used = Function.from_stream(
                stream, syntax
            )
            return cls(function), remaining_stream, symbols_used
        elif stream[0].type == TokenKind.STRUCT:
            struct_type, remaining_stream, symbols_used = StructType.from_stream(stream)
            return cls(struct_type), remaining_stream, symbols_used
        elif stream[0].type == TokenKind.LCURLY:
            (
                table,
                remaining_stream,
                symbols_used,
            ) = Table.from_stream(stream, syntax)
            return cls(table), remaining_stream, symbols_used
        elif stream[0].type == TokenKind.LSQUARE:
            (
                lst,
                remaining_stream,
                symbols_used,
            ) = LList.from_stream(stream, syntax)
            return cls(lst), remaining_stream, symbols_used
        if terminating_token not in stream:
            raise ValueError(f"Expression must end with a {terminating_token}")

        # exclude terminating and starting token
        # leave terminating token in remaining stream so the parser can be sure syntax is valid
        expression_stream, remaining_stream = split_at_context_end(
            stream, terminating_token
        )
        if not expression_stream:
            raise ValueError("Empty expression")

        expression, symbols_used = cls._build_from(expression_stream, syntax)
        return expression, remaining_stream, symbols_used
