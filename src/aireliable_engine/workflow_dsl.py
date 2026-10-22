"""AIReliable 条件 DSL（安全子集）。

设计参考：

核心特性：
    1. **安全**：不调用 Python eval / exec / __import__，全部 AST 自实现求值
    2. **白名单函数**：len / exists / not_exists / contains / startswith / endswith / str / int
    3. **白名单方法**：.startswith() / .endswith() / .contains()
    4. **变量引用**：``${node.output.port}`` / ``${inputs.field}`` / ``${variant}`` / ``${loras}``
    5. **操作符**：``== != > >= < <=``，``and / or / not``
    6. **禁止**：属性访问（除白名单方法）/ 函数定义 / lambda / 任意 Python 内建函数

语法（EBNF 风格）::

    expression  := or_expr
    or_expr     := and_expr ("or" and_expr)*
    and_expr    := not_expr ("and" not_expr)*
    not_expr    := "not" not_expr | comparison
    comparison  := postfix (cmp_op postfix)?
    postfix     := atom ("." method_name "(" args ")")*
    atom        := NUMBER | STRING | "true" | "false" | "null"
                | VAR | func_call | "(" or_expr ")"
    func_call   := IDENT "(" args ")"
    args        := [expression ("," expression)*]

使用示例::

    from aireliable_engine.workflow_dsl import evaluate_condition

    context = {
        "inputs": {"size": "2K", "use_pe": True},
        "variant": "ernie-turbo",
        "loras": [{"name": "style_xfer"}],
        "input_images": {"output": {"images": [1, 2, 3]}},
    }

    # 1. 长度检查
    evaluate_condition('len(${input_images.output.images}) > 0', context)

    # 2. 字符串方法
    evaluate_condition('${variant}.startswith("ernie")', context)

    # 3. 逻辑组合
    evaluate_condition(
        'len(${loras}) > 0 and ${variant}.contains("turbo")',
        context,
    )

    # 4. 存在性检查
    evaluate_condition('exists(${inputs.negative_prompt})', context)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# 异常类
# ===========================================================================

class DSLError(Exception):
    """DSL 错误基类（语法 + 求值）。"""


class DSLParseError(DSLError):
    """DSL 语法错误。

    在 Lexer / Parser 阶段抛出，含错误位置（字符偏移）。
    """

    def __init__(self, message: str, pos: int = -1):
        self.pos = pos
        self.message = message
        if pos >= 0:
            super().__init__(f"{message}（位置 {pos}）")
        else:
            super().__init__(message)


class DSLEvalError(DSLError):
    """DSL 求值错误。

    在 Evaluator 阶段抛出（变量不存在 / 类型不匹配 / 函数参数错误等）。
    """


# ===========================================================================
# Token
# ===========================================================================

@dataclass
class Token:
    """词法 token。

    Attributes:
        type: token 类型（NUMBER / STRING / IDENT / VAR / OP / DOT /
            LPAREN / RPAREN / COMMA / EOF）
        value: token 原始文本
        pos: token 起始字符位置（用于错误定位）
    """

    type: str
    value: str
    pos: int


# ===========================================================================
# Lexer（词法分析器）
# ===========================================================================

# Token 正则模式列表（顺序敏感：长匹配优先）
# None 表示跳过该 token（如空白）
_TOKEN_PATTERNS: list[tuple[str, Optional[str]]] = [
    (r"\s+", None),                              # 空白（跳过）
    (r"\d+\.\d+", "NUMBER"),                     # 浮点
    (r"\d+", "NUMBER"),                          # 整数
    (r'"[^"]*"', "STRING"),                      # 双引号字符串
    (r"'[^']*'", "STRING"),                      # 单引号字符串
    (r"\$\{[^}]+\}", "VAR"),                     # ${...} 变量引用
    (r"==|!=|>=|<=|>|<", "OP"),                  # 比较操作符
    (r"\(", "LPAREN"),
    (r"\)", "RPAREN"),
    (r",", "COMMA"),
    (r"\.", "DOT"),
    (r"[A-Za-z_][A-Za-z0-9_]*", "IDENT"),         # 标识符（含关键字）
]


class Lexer:
    """词法分析器：把表达式字符串切分为 token 列表。

    简化策略：所有字母数字开头的 token 都识别为 IDENT，
    关键字（and/or/not/true/false/null）在 Parser 中按 IDENT 值区分。
    """

    def __init__(self, text: str):
        self.text = text
        self.n = len(text)

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        text = self.text
        i = 0
        n = self.n
        while i < n:
            matched = False
            for pattern, ttype in _TOKEN_PATTERNS:
                m = re.match(pattern, text[i:])
                if m:
                    value = m.group(0)
                    if ttype is not None:
                        tokens.append(Token(ttype, value, i))
                    i += len(value)
                    matched = True
                    break
            if not matched:
                raise DSLParseError(
                    f"非法字符 {text[i]!r}",
                    pos=i,
                )
        tokens.append(Token("EOF", "", i))
        return tokens


# ===========================================================================
# AST 节点
# ===========================================================================

@dataclass
class Literal:
    """字面量节点。

    Attributes:
        value: Python 值（int/float/str/bool/None）
        kind: 字面量类型（"number" / "string" / "bool" / "null"）
    """

    value: Any
    kind: str


@dataclass
class VariableRef:
    """变量引用节点 ``${a.b.c}``。

    Attributes:
        path: 路径段列表，如 ``["a", "b", "c"]``
        raw: 原始文本 ``${a.b.c}``（用于错误信息）
    """

    path: list[str]
    raw: str = ""


@dataclass
class FuncCall:
    """函数调用节点 ``len(x)``。

    Attributes:
        name: 函数名（必须在白名单内）
        args: 参数 AST 节点列表
    """

    name: str
    args: list[Any] = field(default_factory=list)


@dataclass
class MethodCall:
    """方法调用节点 ``base.contains(x)``。

    Attributes:
        base: 被调用对象 AST 节点
        method: 方法名（必须在白名单内）
        args: 参数 AST 节点列表
    """

    base: Any
    method: str
    args: list[Any] = field(default_factory=list)


@dataclass
class BinaryOp:
    """二元操作节点。

    Attributes:
        op: 操作符（``==`` / ``!=`` / ``>`` / ``>=`` / ``<`` / ``<=`` /
            ``and`` / ``or``）
        left: 左操作数 AST 节点
        right: 右操作数 AST 节点
    """

    op: str
    left: Any
    right: Any


@dataclass
class UnaryOp:
    """一元操作节点。

    Attributes:
        op: 操作符（目前仅支持 ``not``）
        operand: 操作数 AST 节点
    """

    op: str
    operand: Any


# ===========================================================================
# Parser（递归下降语法分析器）
# ===========================================================================

# 关键字
_KEYWORDS = frozenset({"and", "or", "not", "true", "false", "null"})

# 白名单函数（调用形式 ``func(args)``）
_ALLOWED_FUNCS = frozenset({
    "len", "exists", "not_exists",
    "contains", "startswith", "endswith",
    "str", "int",
})

# 白名单方法（调用形式 ``base.method(args)``）
_ALLOWED_METHODS = frozenset({"startswith", "endswith", "contains"})

# 比较操作符
_CMP_OPS = frozenset({"==", "!=", ">", ">=", "<", "<="})


class Parser:
    """递归下降语法分析器。

    文法见模块 docstring。出错抛 DSLParseError。
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # ----- token 操作 -----
    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect(self, ttype: str, value: Optional[str] = None) -> Token:
        t = self._current()
        if t.type != ttype or (value is not None and t.value != value):
            expected = f"{ttype}" + (f" {value!r}" if value else "")
            raise DSLParseError(
                f"期望 {expected}, 实际 {t.type} {t.value!r}",
                pos=t.pos,
            )
        return self._advance()

    def _is_keyword(self, kw: str) -> bool:
        t = self._current()
        return t.type == "IDENT" and t.value == kw

    # ----- 入口 -----
    def parse(self) -> Any:
        if self._current().type == "EOF":
            raise DSLParseError("空表达式")
        node = self._or_expr()
        if self._current().type != "EOF":
            t = self._current()
            raise DSLParseError(
                f"未消费的 token {t.value!r}",
                pos=t.pos,
            )
        return node

    # ----- or_expr := and_expr ("or" and_expr)* -----
    def _or_expr(self) -> Any:
        left = self._and_expr()
        while self._is_keyword("or"):
            self._advance()
            right = self._and_expr()
            left = BinaryOp("or", left, right)
        return left

    # ----- and_expr := not_expr ("and" not_expr)* -----
    def _and_expr(self) -> Any:
        left = self._not_expr()
        while self._is_keyword("and"):
            self._advance()
            right = self._not_expr()
            left = BinaryOp("and", left, right)
        return left

    # ----- not_expr := "not" not_expr | comparison -----
    def _not_expr(self) -> Any:
        if self._is_keyword("not"):
            self._advance()
            operand = self._not_expr()
            return UnaryOp("not", operand)
        return self._comparison()

    # ----- comparison := postfix (cmp_op postfix)? -----
    def _comparison(self) -> Any:
        left = self._postfix()
        if self._current().type == "OP":
            op = self._advance().value
            right = self._postfix()
            return BinaryOp(op, left, right)
        return left

    # ----- postfix := atom ("." method_name "(" args ")")* -----
    def _postfix(self) -> Any:
        node = self._atom()
        while self._current().type == "DOT":
            self._advance()  # DOT
            method_token = self._current()
            if method_token.type != "IDENT":
                raise DSLParseError(
                    f"期望方法名, 实际 {method_token.type} {method_token.value!r}",
                    pos=method_token.pos,
                )
            method_name = method_token.value
            if method_name not in _ALLOWED_METHODS:
                allowed = ", ".join(sorted(_ALLOWED_METHODS))
                raise DSLParseError(
                    f"不允许的方法 {method_name!r}（仅支持: {allowed}）",
                    pos=method_token.pos,
                )
            self._advance()  # IDENT
            self._expect("LPAREN")
            args = self._args()
            self._expect("RPAREN")
            node = MethodCall(node, method_name, args)
        return node

    # ----- atom -----
    def _atom(self) -> Any:
        t = self._current()

        if t.type == "NUMBER":
            self._advance()
            value: Any = float(t.value) if "." in t.value else int(t.value)
            return Literal(value, "number")

        if t.type == "STRING":
            self._advance()
            inner = t.value[1:-1]  # 去掉外层引号
            return Literal(inner, "string")

        if t.type == "VAR":
            self._advance()
            inner = t.value[2:-1]  # 去掉 ${ 和 }
            path = [p.strip() for p in inner.split(".") if p.strip()]
            if not path:
                raise DSLParseError(
                    f"变量引用 {t.value!r} 路径为空",
                    pos=t.pos,
                )
            return VariableRef(path, raw=t.value)

        if t.type == "IDENT":
            # 关键字字面量
            if t.value == "true":
                self._advance()
                return Literal(True, "bool")
            if t.value == "false":
                self._advance()
                return Literal(False, "bool")
            if t.value == "null":
                self._advance()
                return Literal(None, "null")
            # 函数调用
            if t.value in _ALLOWED_FUNCS:
                self._advance()
                self._expect("LPAREN")
                args = self._args()
                self._expect("RPAREN")
                return FuncCall(t.value, args)
            raise DSLParseError(
                f"未知标识符 {t.value!r}",
                pos=t.pos,
            )

        if t.type == "LPAREN":
            self._advance()
            node = self._or_expr()
            self._expect("RPAREN")
            return node

        raise DSLParseError(
            f"未预期的 token {t.type} {t.value!r}",
            pos=t.pos,
        )

    # ----- args := [expression ("," expression)*] -----
    def _args(self) -> list[Any]:
        args: list[Any] = []
        if self._current().type == "RPAREN":
            return args
        args.append(self._or_expr())
        while self._current().type == "COMMA":
            self._advance()
            args.append(self._or_expr())
        return args


# ===========================================================================
# Evaluator（AST 求值器）
# ===========================================================================

class Evaluator:
    """AST 求值器。

    递归遍历 AST，按节点类型求值。

    Args:
        context: 变量上下文，按 path 段递归查找。
            例如 ``${a.b.c}`` 在 ``context["a"]["b"]["c"]`` 处查找。
    """

    def __init__(self, context: dict[str, Any]):
        self.context = context

    def eval(self, node: Any) -> Any:
        if isinstance(node, Literal):
            return node.value
        if isinstance(node, VariableRef):
            return self._eval_var(node)
        if isinstance(node, FuncCall):
            return self._eval_func(node)
        if isinstance(node, MethodCall):
            return self._eval_method(node)
        if isinstance(node, BinaryOp):
            return self._eval_binop(node)
        if isinstance(node, UnaryOp):
            return self._eval_unop(node)
        raise DSLEvalError(f"未知 AST 节点类型 {type(node).__name__}")

    # ----- 变量引用 -----
    def _eval_var(self, node: VariableRef) -> Any:
        current: Any = self.context
        for i, seg in enumerate(node.path):
            current = self._access(current, seg, node.raw, depth=i)
        return current

    @staticmethod
    def _access(current: Any, seg: str, raw: str, depth: int = 0) -> Any:
        """按 path 段递归访问。

        支持 dict[key] / list[index] / 对象属性（拒绝 _ 开头私有属性）。
        """
        if current is None:
            raise DSLEvalError(
                f"变量 {raw} 在 path 段 {seg!r} 处为 None（无法继续访问）"
            )
        if isinstance(current, dict):
            if seg not in current:
                raise DSLEvalError(f"变量 {raw} 不存在（key {seg!r} 未找到）")
            return current[seg]
        if isinstance(current, (list, tuple)):
            try:
                idx = int(seg)
            except ValueError:
                raise DSLEvalError(
                    f"变量 {raw} 索引 {seg!r} 不是整数"
                )
            if idx < 0 or idx >= len(current):
                raise DSLEvalError(
                    f"变量 {raw} 索引 {idx} 越界（长度 {len(current)}）"
                )
            return current[idx]
        # 对象属性访问：拒绝私有/保护属性
        if seg.startswith("_"):
            raise DSLEvalError(
                f"禁止访问私有属性 {seg!r}（变量 {raw}）"
            )
        if not hasattr(current, seg):
            raise DSLEvalError(
                f"变量 {raw} 不存在（属性 {seg!r} 未找到于 "
                f"{type(current).__name__}）"
            )
        return getattr(current, seg)

    # ----- 函数调用 -----
    def _eval_func(self, node: FuncCall) -> Any:
        name = node.name

        # exists / not_exists 延迟求值：不预先 eval args
        if name == "exists":
            if len(node.args) != 1:
                raise DSLEvalError("exists() 需要 1 个参数")
            return self._try_exists(node.args[0])
        if name == "not_exists":
            if len(node.args) != 1:
                raise DSLEvalError("not_exists() 需要 1 个参数")
            return not self._try_exists(node.args[0])

        # 其他函数预先 eval args
        args = [self.eval(a) for a in node.args]

        if name == "len":
            if len(args) != 1:
                raise DSLEvalError("len() 需要 1 个参数")
            if args[0] is None:
                raise DSLEvalError("len() 参数不能为 None")
            try:
                return len(args[0])
            except TypeError as e:
                raise DSLEvalError(f"len() 调用失败: {e}")

        if name == "contains":
            if len(args) != 2:
                raise DSLEvalError("contains() 需要 2 个参数")
            container, item = args
            if container is None:
                raise DSLEvalError("contains() container 不能为 None")
            try:
                return item in container
            except TypeError as e:
                raise DSLEvalError(f"contains() 调用失败: {e}")

        if name == "startswith":
            if len(args) != 2:
                raise DSLEvalError("startswith() 需要 2 个参数")
            s, prefix = args
            if not isinstance(s, str):
                raise DSLEvalError(
                    f"startswith() 第 1 个参数需要 str, 实际 {type(s).__name__}"
                )
            if not isinstance(prefix, str):
                raise DSLEvalError(
                    f"startswith() 第 2 个参数需要 str, 实际 {type(prefix).__name__}"
                )
            return s.startswith(prefix)

        if name == "endswith":
            if len(args) != 2:
                raise DSLEvalError("endswith() 需要 2 个参数")
            s, suffix = args
            if not isinstance(s, str):
                raise DSLEvalError(
                    f"endswith() 第 1 个参数需要 str, 实际 {type(s).__name__}"
                )
            if not isinstance(suffix, str):
                raise DSLEvalError(
                    f"endswith() 第 2 个参数需要 str, 实际 {type(suffix).__name__}"
                )
            return s.endswith(suffix)

        if name == "str":
            if len(args) != 1:
                raise DSLEvalError("str() 需要 1 个参数")
            return str(args[0]) if args[0] is not None else ""

        if name == "int":
            if len(args) != 1:
                raise DSLEvalError("int() 需要 1 个参数")
            try:
                return int(args[0])
            except (TypeError, ValueError) as e:
                raise DSLEvalError(f"int() 调用失败: {e}")

        raise DSLEvalError(f"未知函数 {name!r}")

    def _try_exists(self, arg_node: Any) -> bool:
        """检查变量是否存在，不抛错。

        - 变量引用：path 全部能解析 → True；任一段不能 → False
        - 字面量 / 函数调用：求值成功 → True；抛错 → False
        """
        try:
            self.eval(arg_node)
            return True
        except DSLEvalError:
            return False

    # ----- 方法调用 -----
    def _eval_method(self, node: MethodCall) -> Any:
        base = self.eval(node.base)
        args = [self.eval(a) for a in node.args]
        method = node.method

        if method == "contains":
            if len(args) != 1:
                raise DSLEvalError(".contains() 需要 1 个参数")
            if base is None:
                raise DSLEvalError(".contains() base 不能为 None")
            try:
                return args[0] in base
            except TypeError as e:
                raise DSLEvalError(f".contains() 调用失败: {e}")

        if method == "startswith":
            if len(args) != 1:
                raise DSLEvalError(".startswith() 需要 1 个参数")
            if not isinstance(base, str):
                raise DSLEvalError(
                    f".startswith() 需要 str, 实际 {type(base).__name__}"
                )
            if not isinstance(args[0], str):
                raise DSLEvalError(
                    f".startswith() 参数需要 str, 实际 {type(args[0]).__name__}"
                )
            return base.startswith(args[0])

        if method == "endswith":
            if len(args) != 1:
                raise DSLEvalError(".endswith() 需要 1 个参数")
            if not isinstance(base, str):
                raise DSLEvalError(
                    f".endswith() 需要 str, 实际 {type(base).__name__}"
                )
            if not isinstance(args[0], str):
                raise DSLEvalError(
                    f".endswith() 参数需要 str, 实际 {type(args[0]).__name__}"
                )
            return base.endswith(args[0])

        raise DSLEvalError(f"不允许的方法 {method!r}")

    # ----- 二元操作 -----
    def _eval_binop(self, node: BinaryOp) -> Any:
        op = node.op

        # 逻辑短路
        if op == "and":
            left = self.eval(node.left)
            if not left:
                return False
            return bool(self.eval(node.right))
        if op == "or":
            left = self.eval(node.left)
            if left:
                return True
            return bool(self.eval(node.right))

        # 比较
        left = self.eval(node.left)
        right = self.eval(node.right)
        try:
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
        except TypeError as e:
            raise DSLEvalError(
                f"操作符 {op!r} 应用于 {type(left).__name__} 和 "
                f"{type(right).__name__} 失败: {e}"
            )

        raise DSLEvalError(f"未知操作符 {op!r}")

    # ----- 一元操作 -----
    def _eval_unop(self, node: UnaryOp) -> Any:
        if node.op == "not":
            return not bool(self.eval(node.operand))
        raise DSLEvalError(f"未知一元操作符 {node.op!r}")


# ===========================================================================
# 公开 API
# ===========================================================================

def evaluate_condition(expression: str, context: dict[str, Any]) -> bool:
    """评估条件表达式，返回 bool。

    Args:
        expression: DSL 表达式字符串，如
            ``'len(${input_images.output.images}) > 0'``
        context: 变量上下文，按 path 段递归查找。
            通常包含 inputs / variant / loras / 节点输出。

    Returns:
        表达式求值结果，强制转换为 bool。

    Raises:
        DSLParseError: 表达式语法错误
        DSLEvalError: 求值错误（变量不存在 / 类型不匹配等）

    Example::

        context = {
            "inputs": {"size": "2K"},
            "variant": "ernie-turbo",
            "input_images": {"output": {"images": [1, 2, 3]}},
        }
        assert evaluate_condition(
            'len(${input_images.output.images}) > 0 and ${variant}.startswith("ernie")',
            context,
        )
    """
    tokens = Lexer(expression).tokenize()
    ast = Parser(tokens).parse()
    result = Evaluator(context).eval(ast)
    return bool(result)


def validate_expression(expression: str) -> list[str]:
    """仅语法校验，返回错误信息列表。

    不实际求值，只检查语法合法性。

    Args:
        expression: DSL 表达式字符串

    Returns:
        错误信息列表（空列表表示语法合法）
    """
    try:
        tokens = Lexer(expression).tokenize()
        Parser(tokens).parse()
        return []
    except DSLError as e:
        return [str(e)]


def parse_expression(expression: str) -> Any:
    """解析表达式为 AST（用于调试 / 工具可视化）。

    Args:
        expression: DSL 表达式字符串

    Returns:
        AST 根节点

    Raises:
        DSLParseError: 语法错误
    """
    tokens = Lexer(expression).tokenize()
    return Parser(tokens).parse()


def eval_ast(ast_node: Any, context: dict[str, Any]) -> Any:
    """对已解析的 AST 求值。

    Args:
        ast_node: AST 根节点（来自 parse_expression）
        context: 变量上下文

    Returns:
        求值结果（不强制 bool 转换）

    Raises:
        DSLEvalError: 求值错误
    """
    return Evaluator(context).eval(ast_node)


__all__ = [
    # 异常
    "DSLError",
    "DSLParseError",
    "DSLEvalError",
    # AST 节点
    "Literal",
    "VariableRef",
    "FuncCall",
    "MethodCall",
    "BinaryOp",
    "UnaryOp",
    # Lexer / Parser / Evaluator
    "Token",
    "Lexer",
    "Parser",
    "Evaluator",
    # 公开 API
    "evaluate_condition",
    "validate_expression",
    "parse_expression",
    "eval_ast",
]
