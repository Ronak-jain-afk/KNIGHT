# SPDX-License-Identifier: Apache-2.0
import ast
import math
import operator

from langchain_core.tools import tool

# Allowed operators
BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed math functions and constants
MATH_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in BIN_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow:
            if right > 1000:
                raise ValueError("Exponent too large (max 1000)")
            # Reject before pow() builds an astronomically large integer (H-09).
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                estimated_bits = max(1, left.bit_length()) * right
                if estimated_bits > 10_000:
                    raise ValueError("Result too large (max 10000 bits)")
        return BIN_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return UNARY_OPS[op_type](operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise TypeError("Only direct math function calls allowed")
        fn_name = node.func.id
        if fn_name not in MATH_FUNCTIONS:
            raise ValueError(f"Unknown or disallowed function: {fn_name}")
        fn = MATH_FUNCTIONS[fn_name]
        args = [_eval_node(arg) for arg in node.args]
        return fn(*args)
    elif isinstance(node, ast.Name):
        if node.id in MATH_FUNCTIONS:
            val = MATH_FUNCTIONS[node.id]
            if isinstance(val, (int, float)):
                return val
        raise ValueError(f"Unknown identifier: {node.id}")
    else:
        raise TypeError(f"Disallowed expression syntax: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """Safely calculate a mathematical expression using numbers, basic operators, and math functions."""
    try:
        expr = expression.strip()
        parsed = ast.parse(expr, mode="eval")
        result = _eval_node(parsed)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return f"ERROR: invalid_expression: {exc}"
