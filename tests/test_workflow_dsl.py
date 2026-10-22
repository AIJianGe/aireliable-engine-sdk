"""条件 DSL 单元测试。

覆盖任务 4.2 验收标准：
    - 词法分析器（tokenizer）
    - 语法分析器（递归下降 parser）
    - AST 求值器
    - 内置函数白名单
    - 所有语法分支

"""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import pytest

from aireliable_engine.workflow_dsl import (
    BinaryOp,
    DSLEvalError,
    DSLError,
    DSLParseError,
    Evaluator,
    FuncCall,
    Lexer,
    Literal,
    MethodCall,
    Parser,
    Token,
    UnaryOp,
    VariableRef,
    eval_ast,
    evaluate_condition,
    parse_expression,
    validate_expression,
)


# ===========================================================================
# 测试用 context
# ===========================================================================


@pytest.fixture
def basic_ctx() -> dict[str, Any]:
    """基础测试上下文。"""
    return {
        "inputs": {
            "size": "2K",
            "use_pe": True,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "negative_prompt": "",
            "tags": ["a", "b", "c"],
            "empty_list": [],
            "none_value": None,
        },
        "variant": "ernie-turbo",
        "loras": [{"name": "style_xfer"}, {"name": "color_xfer"}],
        "input_images": {
            "output": {
                "images": [1, 2, 3],
            },
        },
        "node_a": {
            "output": {
                "port": "value",
            },
        },
    }


# ===========================================================================
# Lexer 测试
# ===========================================================================


class TestLexer:
    """Lexer 词法分析测试。"""

    def test_tokenize_integer(self):
        tokens = Lexer("42").tokenize()
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == "42"
        assert tokens[-1].type == "EOF"

    def test_tokenize_float(self):
        tokens = Lexer("3.14").tokenize()
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == "3.14"

    def test_tokenize_double_quoted_string(self):
        tokens = Lexer('"hello"').tokenize()
        assert tokens[0].type == "STRING"
        assert tokens[0].value == '"hello"'

    def test_tokenize_single_quoted_string(self):
        tokens = Lexer("'world'").tokenize()
        assert tokens[0].type == "STRING"
        assert tokens[0].value == "'world'"

    def test_tokenize_var_simple(self):
        tokens = Lexer("${variant}").tokenize()
        assert tokens[0].type == "VAR"
        assert tokens[0].value == "${variant}"

    def test_tokenize_var_with_path(self):
        tokens = Lexer("${node_a.output.port}").tokenize()
        assert tokens[0].type == "VAR"
        assert tokens[0].value == "${node_a.output.port}"

    def test_tokenize_operators(self):
        for op in ["==", "!=", ">", ">=", "<", "<="]:
            tokens = Lexer(op).tokenize()
            assert tokens[0].type == "OP"
            assert tokens[0].value == op

    def test_tokenize_punctuation(self):
        tokens = Lexer("a(b,c).d").tokenize()
        types = [t.type for t in tokens if t.type != "EOF"]
        assert types == ["IDENT", "LPAREN", "IDENT", "COMMA", "IDENT", "RPAREN", "DOT", "IDENT"]

    def test_tokenize_keywords_as_ident(self):
        """关键字在 Lexer 阶段仍是 IDENT。"""
        for kw in ["and", "or", "not", "true", "false", "null"]:
            tokens = Lexer(kw).tokenize()
            assert tokens[0].type == "IDENT"
            assert tokens[0].value == kw

    def test_tokenize_whitespace_skipped(self):
        tokens = Lexer("  42  ").tokenize()
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == "42"

    def test_tokenize_newlines_skipped(self):
        tokens = Lexer("42\n  3").tokenize()
        types = [t.type for t in tokens if t.type != "EOF"]
        assert types == ["NUMBER", "NUMBER"]

    def test_tokenize_invalid_char_raises(self):
        with pytest.raises(DSLParseError):
            Lexer("@").tokenize()

    def test_tokenize_invalid_char_with_pos(self):
        with pytest.raises(DSLParseError) as exc_info:
            Lexer("42 @").tokenize()
        assert exc_info.value.pos == 3

    def test_tokenize_empty_string(self):
        tokens = Lexer("").tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == "EOF"

    def test_tokenize_complex_expression(self):
        tokens = Lexer('len(${x}) > 0 and ${y}.startswith("abc")').tokenize()
        types = [t.type for t in tokens if t.type != "EOF"]
        assert types == [
            "IDENT", "LPAREN", "VAR", "RPAREN",
            "OP", "NUMBER",
            "IDENT", "VAR", "DOT", "IDENT", "LPAREN", "STRING", "RPAREN",
        ]

    def test_tokenize_op_long_match_priority(self):
        """长操作符优先于短操作符（>= 而非 > + =）。"""
        tokens = Lexer(">=").tokenize()
        assert tokens[0].type == "OP"
        assert tokens[0].value == ">="

    def test_tokenize_number_float_priority(self):
        """浮点优先于整数。"""
        tokens = Lexer("3.14").tokenize()
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == "3.14"


# ===========================================================================
# AST 节点测试
# ===========================================================================


class TestASTNodes:
    """AST 节点是 dataclass。"""

    def test_literal_is_dataclass(self):
        assert is_dataclass(Literal)

    def test_variable_ref_is_dataclass(self):
        assert is_dataclass(VariableRef)

    def test_func_call_is_dataclass(self):
        assert is_dataclass(FuncCall)

    def test_method_call_is_dataclass(self):
        assert is_dataclass(MethodCall)

    def test_binary_op_is_dataclass(self):
        assert is_dataclass(BinaryOp)

    def test_unary_op_is_dataclass(self):
        assert is_dataclass(UnaryOp)


# ===========================================================================
# Parser 测试
# ===========================================================================


class TestParser:
    """Parser 语法分析测试。"""

    def test_parse_number_integer(self):
        ast = parse_expression("42")
        assert isinstance(ast, Literal)
        assert ast.value == 42
        assert ast.kind == "number"

    def test_parse_number_float(self):
        ast = parse_expression("3.14")
        assert isinstance(ast, Literal)
        assert ast.value == 3.14
        assert ast.kind == "number"

    def test_parse_string_double_quoted(self):
        ast = parse_expression('"hello"')
        assert isinstance(ast, Literal)
        assert ast.value == "hello"
        assert ast.kind == "string"

    def test_parse_string_single_quoted(self):
        ast = parse_expression("'world'")
        assert isinstance(ast, Literal)
        assert ast.value == "world"
        assert ast.kind == "string"

    def test_parse_true(self):
        ast = parse_expression("true")
        assert isinstance(ast, Literal)
        assert ast.value is True
        assert ast.kind == "bool"

    def test_parse_false(self):
        ast = parse_expression("false")
        assert isinstance(ast, Literal)
        assert ast.value is False
        assert ast.kind == "bool"

    def test_parse_null(self):
        ast = parse_expression("null")
        assert isinstance(ast, Literal)
        assert ast.value is None
        assert ast.kind == "null"

    def test_parse_var_simple(self):
        ast = parse_expression("${variant}")
        assert isinstance(ast, VariableRef)
        assert ast.path == ["variant"]
        assert ast.raw == "${variant}"

    def test_parse_var_with_path(self):
        ast = parse_expression("${node_a.output.port}")
        assert isinstance(ast, VariableRef)
        assert ast.path == ["node_a", "output", "port"]

    def test_parse_func_call(self):
        ast = parse_expression("len(${x})")
        assert isinstance(ast, FuncCall)
        assert ast.name == "len"
        assert len(ast.args) == 1

    def test_parse_func_call_no_args(self):
        ast = parse_expression("len()")
        assert isinstance(ast, FuncCall)
        assert ast.name == "len"
        assert len(ast.args) == 0

    def test_parse_func_call_multiple_args(self):
        ast = parse_expression('contains(${x}, "a")')
        assert isinstance(ast, FuncCall)
        assert ast.name == "contains"
        assert len(ast.args) == 2

    def test_parse_method_call(self):
        ast = parse_expression('${x}.startswith("abc")')
        assert isinstance(ast, MethodCall)
        assert ast.method == "startswith"
        assert len(ast.args) == 1

    def test_parse_binary_op_comparison(self):
        ast = parse_expression("1 > 2")
        assert isinstance(ast, BinaryOp)
        assert ast.op == ">"

    def test_parse_binary_op_equality(self):
        ast = parse_expression('${x} == "abc"')
        assert isinstance(ast, BinaryOp)
        assert ast.op == "=="

    def test_parse_binary_op_and(self):
        ast = parse_expression("true and false")
        assert isinstance(ast, BinaryOp)
        assert ast.op == "and"

    def test_parse_binary_op_or(self):
        ast = parse_expression("true or false")
        assert isinstance(ast, BinaryOp)
        assert ast.op == "or"

    def test_parse_unary_not(self):
        ast = parse_expression("not true")
        assert isinstance(ast, UnaryOp)
        assert ast.op == "not"

    def test_parse_paren_group(self):
        ast = parse_expression("(true)")
        assert isinstance(ast, Literal)
        assert ast.value is True

    def test_parse_nested_paren(self):
        ast = parse_expression("((true))")
        assert isinstance(ast, Literal)
        assert ast.value is True

    def test_parse_precedence_and_over_or(self):
        """and 比 or 优先级高。"""
        ast = parse_expression("true or false and false")
        assert isinstance(ast, BinaryOp)
        assert ast.op == "or"
        # 右边是 and
        assert isinstance(ast.right, BinaryOp)
        assert ast.right.op == "and"

    def test_parse_precedence_comparison_over_and(self):
        """比较比 and 优先级高。"""
        ast = parse_expression("1 > 0 and 2 > 1")
        assert isinstance(ast, BinaryOp)
        assert ast.op == "and"
        assert isinstance(ast.left, BinaryOp)
        assert ast.left.op == ">"
        assert isinstance(ast.right, BinaryOp)
        assert ast.right.op == ">"

    def test_parse_not_over_comparison(self):
        """not 比较比较优先级高。"""
        ast = parse_expression("not 1 > 0")
        # 等价于 not (1 > 0)
        assert isinstance(ast, UnaryOp)
        assert ast.op == "not"
        assert isinstance(ast.operand, BinaryOp)

    def test_parse_method_chain_syntactically_allowed(self):
        """方法链语法上允许（白名单方法可串联），但求值时类型不匹配会抛错。"""
        # 语法上合法：${x}.startswith("a") -> bool，再调 .endswith 应抛 DSLEvalError
        ast = parse_expression('${x}.startswith("a").endswith("b")')
        assert isinstance(ast, MethodCall)
        # 内层是 ${x}.startswith("a")
        assert isinstance(ast.base, MethodCall)
        assert ast.base.method == "startswith"
        # 求值时抛错（startswith 返回 bool，bool 无 endswith 方法）
        with pytest.raises(DSLEvalError):
            eval_ast(ast, {"x": "abcde"})


class TestParserErrors:
    """Parser 错误情况。"""

    def test_empty_expression_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("")

    def test_whitespace_only_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("   ")

    def test_unconsumed_token_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("1 2")

    def test_unclosed_paren_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("(1")

    def test_unmatched_close_paren_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("1)")

    def test_unknown_identifier_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("abc")

    def test_disallowed_function_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("eval(1)")

    def test_disallowed_method_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression('${x}.upper()')

    def test_disallowed_dunder_method_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression('${x}.__class__()')

    def test_missing_func_args_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("len")  # len 后面没有括号

    def test_missing_comma_in_args_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression('contains(${x} "a")')

    def test_missing_method_paren_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression('${x}.startswith')

    def test_empty_var_path_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("${}")

    def test_invalid_char_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("@")

    def test_double_op_raises(self):
        with pytest.raises(DSLParseError):
            parse_expression("1 >> 2")


# ===========================================================================
# Evaluator 测试 - 字面量与变量
# ===========================================================================


class TestEvaluatorLiterals:
    """Evaluator 字面量求值。"""

    def test_eval_integer(self, basic_ctx):
        ast = parse_expression("42")
        assert Evaluator(basic_ctx).eval(ast) == 42

    def test_eval_float(self, basic_ctx):
        ast = parse_expression("3.14")
        assert Evaluator(basic_ctx).eval(ast) == 3.14

    def test_eval_string(self, basic_ctx):
        ast = parse_expression('"hello"')
        assert Evaluator(basic_ctx).eval(ast) == "hello"

    def test_eval_true(self, basic_ctx):
        ast = parse_expression("true")
        assert Evaluator(basic_ctx).eval(ast) is True

    def test_eval_false(self, basic_ctx):
        ast = parse_expression("false")
        assert Evaluator(basic_ctx).eval(ast) is False

    def test_eval_null(self, basic_ctx):
        ast = parse_expression("null")
        assert Evaluator(basic_ctx).eval(ast) is None


class TestEvaluatorVariables:
    """Evaluator 变量引用求值。"""

    def test_eval_var_dict_path(self, basic_ctx):
        ast = parse_expression("${inputs.size}")
        assert Evaluator(basic_ctx).eval(ast) == "2K"

    def test_eval_var_nested_dict_path(self, basic_ctx):
        ast = parse_expression("${node_a.output.port}")
        assert Evaluator(basic_ctx).eval(ast) == "value"

    def test_eval_var_list_index(self, basic_ctx):
        ast = parse_expression("${inputs.tags.0}")
        assert Evaluator(basic_ctx).eval(ast) == "a"

    def test_eval_var_list_index_out_of_range(self, basic_ctx):
        ast = parse_expression("${inputs.tags.10}")
        with pytest.raises(DSLEvalError):
            Evaluator(basic_ctx).eval(ast)

    def test_eval_var_missing_key_raises(self, basic_ctx):
        ast = parse_expression("${inputs.nonexistent}")
        with pytest.raises(DSLEvalError):
            Evaluator(basic_ctx).eval(ast)

    def test_eval_var_missing_top_key_raises(self, basic_ctx):
        ast = parse_expression("${nonexistent}")
        with pytest.raises(DSLEvalError):
            Evaluator(basic_ctx).eval(ast)

    def test_eval_var_none_value(self, basic_ctx):
        ast = parse_expression("${inputs.none_value}")
        assert Evaluator(basic_ctx).eval(ast) is None

    def test_eval_var_access_on_none_raises(self, basic_ctx):
        """None 上访问 path 抛错。"""
        ast = parse_expression("${inputs.none_value.sub}")
        with pytest.raises(DSLEvalError):
            Evaluator(basic_ctx).eval(ast)


# ===========================================================================
# Evaluator 测试 - 函数调用
# ===========================================================================


class TestEvaluatorFunctions:
    """Evaluator 白名单函数测试。"""

    def test_len_list(self, basic_ctx):
        assert evaluate_condition("len(${inputs.tags}) == 3", basic_ctx)

    def test_len_dict(self, basic_ctx):
        # dict 的 len 是 key 数量
        assert evaluate_condition("len(${inputs}) > 0", basic_ctx)

    def test_len_string(self, basic_ctx):
        assert evaluate_condition('len(${variant}) == 11', basic_ctx)

    def test_len_none_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition("len(${inputs.none_value})", basic_ctx)

    def test_len_wrong_arg_count_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition("len()", basic_ctx)

        with pytest.raises(DSLEvalError):
            evaluate_condition("len(${x}, ${y})", basic_ctx)

    def test_exists_present(self, basic_ctx):
        assert evaluate_condition("exists(${inputs.size})", basic_ctx) is True

    def test_exists_missing(self, basic_ctx):
        assert evaluate_condition("exists(${inputs.nonexistent})", basic_ctx) is False

    def test_exists_none_value_is_present(self, basic_ctx):
        """key 存在但 value 是 None，exists 返回 True。"""
        assert evaluate_condition("exists(${inputs.none_value})", basic_ctx) is True

    def test_exists_wrong_arg_count_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition("exists()", basic_ctx)

    def test_not_exists_present(self, basic_ctx):
        assert evaluate_condition("not_exists(${inputs.size})", basic_ctx) is False

    def test_not_exists_missing(self, basic_ctx):
        assert evaluate_condition("not_exists(${inputs.nonexistent})", basic_ctx) is True

    def test_contains_list_item_present(self, basic_ctx):
        assert evaluate_condition('contains(${inputs.tags}, "a")', basic_ctx)

    def test_contains_list_item_absent(self, basic_ctx):
        assert not evaluate_condition('contains(${inputs.tags}, "z")', basic_ctx)

    def test_contains_string_substring(self, basic_ctx):
        assert evaluate_condition('contains(${variant}, "turbo")', basic_ctx)

    def test_contains_none_container_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition('contains(${inputs.none_value}, "a")', basic_ctx)

    def test_contains_wrong_arg_count_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition('contains(${x})', basic_ctx)

    def test_startswith_func(self, basic_ctx):
        assert evaluate_condition('startswith(${variant}, "ernie")', basic_ctx)

    def test_startswith_func_no_match(self, basic_ctx):
        assert not evaluate_condition('startswith(${variant}, "turbo")', basic_ctx)

    def test_startswith_non_string_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition('startswith(${inputs.tags}, "a")', basic_ctx)

    def test_endswith_func(self, basic_ctx):
        assert evaluate_condition('endswith(${variant}, "turbo")', basic_ctx)

    def test_endswith_func_no_match(self, basic_ctx):
        assert not evaluate_condition('endswith(${variant}, "ernie")', basic_ctx)

    def test_str_func_converts_int(self, basic_ctx):
        ast = parse_expression("str(42)")
        assert Evaluator(basic_ctx).eval(ast) == "42"

    def test_str_func_converts_none(self, basic_ctx):
        ast = parse_expression("str(null)")
        assert Evaluator(basic_ctx).eval(ast) == ""

    def test_int_func_valid(self, basic_ctx):
        ast = parse_expression('int("42")')
        assert Evaluator(basic_ctx).eval(ast) == 42

    def test_int_func_invalid_raises(self, basic_ctx):
        ast = parse_expression('int("abc")')
        with pytest.raises(DSLEvalError):
            Evaluator(basic_ctx).eval(ast)


# ===========================================================================
# Evaluator 测试 - 方法调用
# ===========================================================================


class TestEvaluatorMethods:
    """Evaluator 白名单方法测试。"""

    def test_method_startswith(self, basic_ctx):
        assert evaluate_condition('${variant}.startswith("ernie")', basic_ctx)

    def test_method_startswith_no_match(self, basic_ctx):
        assert not evaluate_condition('${variant}.startswith("turbo")', basic_ctx)

    def test_method_endswith(self, basic_ctx):
        assert evaluate_condition('${variant}.endswith("turbo")', basic_ctx)

    def test_method_endswith_no_match(self, basic_ctx):
        assert not evaluate_condition('${variant}.endswith("ernie")', basic_ctx)

    def test_method_contains_string(self, basic_ctx):
        assert evaluate_condition('${variant}.contains("turbo")', basic_ctx)

    def test_method_contains_list(self, basic_ctx):
        assert evaluate_condition('${inputs.tags}.contains("a")', basic_ctx)

    def test_method_contains_no_match(self, basic_ctx):
        assert not evaluate_condition('${inputs.tags}.contains("z")', basic_ctx)

    def test_method_startswith_non_string_base_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition('${inputs.tags}.startswith("a")', basic_ctx)

    def test_method_contains_none_base_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition('${inputs.none_value}.contains("a")', basic_ctx)

    def test_method_wrong_arg_count_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition('${variant}.startswith()', basic_ctx)

        with pytest.raises(DSLEvalError):
            evaluate_condition('${variant}.startswith("a", "b")', basic_ctx)


# ===========================================================================
# Evaluator 测试 - 操作符
# ===========================================================================


class TestEvaluatorOperators:
    """Evaluator 操作符测试。"""

    def test_eq_numbers(self, basic_ctx):
        assert evaluate_condition("1 == 1", basic_ctx)

    def test_eq_strings(self, basic_ctx):
        assert evaluate_condition('"abc" == "abc"', basic_ctx)

    def test_neq(self, basic_ctx):
        assert evaluate_condition("1 != 2", basic_ctx)

    def test_gt(self, basic_ctx):
        assert evaluate_condition("2 > 1", basic_ctx)

    def test_ge(self, basic_ctx):
        assert evaluate_condition("1 >= 1", basic_ctx)

    def test_lt(self, basic_ctx):
        assert evaluate_condition("1 < 2", basic_ctx)

    def test_le(self, basic_ctx):
        assert evaluate_condition("1 <= 1", basic_ctx)

    def test_eq_var_literal(self, basic_ctx):
        assert evaluate_condition('${inputs.size} == "2K"', basic_ctx)

    def test_eq_var_bool_true(self, basic_ctx):
        assert evaluate_condition("${inputs.use_pe} == true", basic_ctx)

    def test_eq_var_bool_false(self, basic_ctx):
        assert not evaluate_condition("${inputs.use_pe} == false", basic_ctx)

    def test_gt_var_numbers(self, basic_ctx):
        assert evaluate_condition("${inputs.num_inference_steps} > 20", basic_ctx)

    def test_ge_var_numbers(self, basic_ctx):
        assert evaluate_condition("${inputs.guidance_scale} >= 7.5", basic_ctx)

    def test_compare_incompatible_types_raises(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition('"abc" > 1', basic_ctx)


class TestEvaluatorLogic:
    """Evaluator 逻辑操作符测试。"""

    def test_and_both_true(self, basic_ctx):
        assert evaluate_condition("true and true", basic_ctx)

    def test_and_short_circuit_false_left(self, basic_ctx):
        # 左边 false 时不应求右边（即使右边报错也应短路）
        # 这里用不存在的变量测试短路
        assert not evaluate_condition(
            "false and ${nonexistent}", basic_ctx
        )

    def test_and_short_circuit_false_right(self, basic_ctx):
        assert not evaluate_condition(
            "true and false", basic_ctx
        )

    def test_or_both_false(self, basic_ctx):
        assert not evaluate_condition("false or false", basic_ctx)

    def test_or_short_circuit_true_left(self, basic_ctx):
        # 左边 true 时不应求右边
        assert evaluate_condition(
            "true or ${nonexistent}", basic_ctx
        )

    def test_not_true(self, basic_ctx):
        assert not evaluate_condition("not true", basic_ctx)

    def test_not_false(self, basic_ctx):
        assert evaluate_condition("not false", basic_ctx)

    def test_not_var(self, basic_ctx):
        # use_pe=True, not use_pe == False (即 not True) -> False
        assert not evaluate_condition("not ${inputs.use_pe}", basic_ctx)

    def test_complex_and_or(self, basic_ctx):
        # true or false and false  -> true or (false and false) -> true
        assert evaluate_condition("true or false and false", basic_ctx)

    def test_paren_overrides_precedence(self, basic_ctx):
        # (true or false) and false -> false
        assert not evaluate_condition("(true or false) and false", basic_ctx)

    def test_nested_not(self, basic_ctx):
        # not not true -> true
        assert evaluate_condition("not not true", basic_ctx)


# ===========================================================================
# evaluate_condition 集成测试
# ===========================================================================


class TestEvaluateCondition:
    """evaluate_condition 公开 API 测试。"""

    def test_returns_bool_true(self, basic_ctx):
        result = evaluate_condition("1 > 0", basic_ctx)
        assert result is True
        assert isinstance(result, bool)

    def test_returns_bool_false(self, basic_ctx):
        result = evaluate_condition("1 < 0", basic_ctx)
        assert result is False
        assert isinstance(result, bool)

    def test_returns_bool_from_literal(self, basic_ctx):
        result = evaluate_condition("true", basic_ctx)
        assert result is True

    def test_coerces_non_bool_to_bool(self, basic_ctx):
        """非 bool 结果强制转 bool。"""
        # len(${x}) == 3 已经是 bool
        # 但直接 len(${x}) 返回 int，会被强制转 bool
        result = evaluate_condition("len(${inputs.tags})", basic_ctx)
        assert result is True  # len=3 -> bool(3) -> True

    def test_zero_coerces_to_false(self, basic_ctx):
        result = evaluate_condition("len(${inputs.empty_list})", basic_ctx)
        assert result is False  # len=0 -> bool(0) -> False

    def test_complex_condition_1(self, basic_ctx):
        """长度检查 + 字符串匹配。"""
        expr = 'len(${input_images.output.images}) > 0 and ${variant}.startswith("ernie")'
        assert evaluate_condition(expr, basic_ctx)

    def test_complex_condition_2(self, basic_ctx):
        """存在性 + 布尔字段。"""
        expr = 'exists(${inputs.use_pe}) and ${inputs.use_pe} == true'
        assert evaluate_condition(expr, basic_ctx)

    def test_complex_condition_3(self, basic_ctx):
        """数组包含。"""
        expr = 'contains(${inputs.tags}, "a") or contains(${inputs.tags}, "z")'
        assert evaluate_condition(expr, basic_ctx)

    def test_complex_condition_4(self, basic_ctx):
        """not_exists + 空字符串判断。"""
        expr = 'not_exists(${inputs.negative_prompt}) or ${inputs.negative_prompt} == ""'
        # negative_prompt 存在但是空字符串
        assert evaluate_condition(expr, basic_ctx)

    def test_raises_on_parse_error(self, basic_ctx):
        with pytest.raises(DSLParseError):
            evaluate_condition("1 +", basic_ctx)

    def test_raises_on_eval_error(self, basic_ctx):
        with pytest.raises(DSLEvalError):
            evaluate_condition("${nonexistent}", basic_ctx)

    def test_dsl_error_is_base(self):
        """DSLError 是基类。"""
        assert issubclass(DSLParseError, DSLError)
        assert issubclass(DSLEvalError, DSLError)


# ===========================================================================
# validate_expression 测试
# ===========================================================================


class TestValidateExpression:
    """validate_expression 公开 API 测试。"""

    def test_valid_simple(self):
        assert validate_expression("1 > 0") == []

    def test_valid_complex(self):
        assert validate_expression(
            'len(${x}) > 0 and ${y}.startswith("abc")'
        ) == []

    def test_invalid_syntax(self):
        errors = validate_expression("1 +")
        assert len(errors) == 1

    def test_invalid_empty(self):
        errors = validate_expression("")
        assert len(errors) == 1

    def test_invalid_unknown_func(self):
        errors = validate_expression("foo(1)")
        assert len(errors) == 1

    def test_invalid_disallowed_method(self):
        errors = validate_expression('${x}.upper()')
        assert len(errors) == 1

    def test_does_not_eval(self):
        """validate 不实际求值，所以变量不存在不会报错。"""
        # 仅语法检查，不需要 context
        assert validate_expression("${nonexistent}") == []


# ===========================================================================
# parse_expression / eval_ast 测试
# ===========================================================================


class TestParseAndEvalAst:
    """parse_expression + eval_ast 组合测试（AST 复用）。"""

    def test_parse_returns_ast(self):
        ast = parse_expression("1 > 0")
        assert isinstance(ast, BinaryOp)

    def test_eval_ast_evaluates(self, basic_ctx):
        ast = parse_expression("${inputs.size} == \"2K\"")
        assert eval_ast(ast, basic_ctx) is True

    def test_ast_reuse_with_different_context(self):
        """同一 AST 可在多个 context 上求值。"""
        ast = parse_expression("${x} > 0")
        assert eval_ast(ast, {"x": 5}) is True
        assert eval_ast(ast, {"x": -1}) is False
        assert eval_ast(ast, {"x": 0}) is False

    def test_ast_with_list_context(self):
        ast = parse_expression("${items.0} == \"first\"")
        assert eval_ast(ast, {"items": ["first", "second"]}) is True


# ===========================================================================
# 安全性测试
# ===========================================================================


class TestSecurity:
    """DSL 安全性测试。"""

    def test_eval_func_rejected(self):
        with pytest.raises(DSLParseError):
            parse_expression("eval('1+1')")

    def test_exec_func_rejected(self):
        with pytest.raises(DSLParseError):
            parse_expression("exec('print(1)')")

    def test_import_rejected(self):
        with pytest.raises(DSLParseError):
            parse_expression("__import__('os')")

    def test_dunder_method_rejected(self):
        with pytest.raises(DSLParseError):
            parse_expression('${x}.__class__')

    def test_dunder_method_call_rejected(self):
        with pytest.raises(DSLParseError):
            parse_expression('${x}.__class__()')

    def test_private_attr_in_var_rejected(self, basic_ctx):
        """${...} 内部访问私有属性被拒。"""
        # 构造一个有私有属性的对象
        class _Obj:
            def __init__(self):
                self._private = "secret"
                self.public = "ok"

        ctx = {"obj": _Obj()}
        with pytest.raises(DSLEvalError):
            evaluate_condition("${obj._private} == \"secret\"", ctx)

    def test_public_attr_in_var_allowed(self):
        """${...} 内部访问公有属性允许。"""
        class _Obj:
            def __init__(self):
                self.public = "ok"

        ctx = {"obj": _Obj()}
        assert evaluate_condition('${obj.public} == "ok"', ctx)

    def test_no_python_builtins_accessible(self):
        """Python 内建函数不可访问（仅白名单）。"""
        # 尝试访问 open / __import__ / getattr 等
        for bad_func in ["open", "getattr", "setattr", "delattr", "globals", "locals", "vars"]:
            with pytest.raises(DSLParseError):
                parse_expression(f"{bad_func}(1)")


# ===========================================================================
# 边界情况测试
# ===========================================================================


class TestEdgeCases:
    """边界情况测试。"""

    def test_deeply_nested_parens(self, basic_ctx):
        assert evaluate_condition("((((true))))", basic_ctx)

    def test_long_path(self, basic_ctx):
        ctx = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        assert evaluate_condition('${a.b.c.d.e} == "deep"', ctx)

    def test_var_with_spaces_in_path(self):
        """${...} 内 path 段两侧的空格被 strip。"""
        ctx = {"a": {"b": "value"}}
        # 注意：${ a . b } 这样的写法，正则匹配的是 [^}]+，会包含空格
        # Parser 用 split('.') 后 strip
        assert evaluate_condition('${ a . b } == "value"', ctx)

    def test_integer_comparison(self, basic_ctx):
        assert evaluate_condition("${inputs.num_inference_steps} == 30", basic_ctx)

    def test_float_comparison(self, basic_ctx):
        assert evaluate_condition("${inputs.guidance_scale} == 7.5", basic_ctx)

    def test_zero_is_false(self, basic_ctx):
        assert not evaluate_condition("0", basic_ctx)

    def test_nonzero_is_true(self, basic_ctx):
        assert evaluate_condition("1", basic_ctx)

    def test_empty_string_is_false(self, basic_ctx):
        assert not evaluate_condition('""', basic_ctx)

    def test_nonempty_string_is_true(self, basic_ctx):
        assert evaluate_condition('"abc"', basic_ctx)

    def test_mixed_types_eq(self, basic_ctx):
        """不同类型 == 比较返回 False。"""
        assert not evaluate_condition("1 == \"1\"", basic_ctx)

    def test_var_returns_dict_truthy(self, basic_ctx):
        """dict 非空时为 True。"""
        assert evaluate_condition("${inputs}", basic_ctx)

    def test_var_returns_empty_list_falsy(self, basic_ctx):
        """空 list 为 False。"""
        assert not evaluate_condition("${inputs.empty_list}", basic_ctx)


# ===========================================================================
# ===========================================================================


class TestWorkflowScenarios:
    """模拟真实工作流场景的条件表达式。"""

    @pytest.fixture
    def workflow_ctx(self) -> dict:
        """工作流上下文，模拟 WorkflowExecutor 构建的 context。"""
        return {
            "inputs": {
                "prompt": "cyberpunk city",
                "negative_prompt": "blurry",
                "size": "2K",
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "use_pe": True,
            },
            "variant": "ernie-turbo-v2",
            "loras": [{"name": "style_xfer"}, {"name": "color_xfer"}],
            "input_images": {"output": {"images": ["img1", "img2"]}},
            "main_model": {"output": {"model": "pipe_obj", "vae": "vae_obj"}},
            "sampler": {"output": {"latent": "latent_obj"}},
        }

    def test_scenario_images_input_exists(self, workflow_ctx):
        """场景：判断是否有多图输入。"""
        expr = 'len(${input_images.output.images}) > 0'
        assert evaluate_condition(expr, workflow_ctx)

    def test_scenario_high_res_upscale(self, workflow_ctx):
        """场景：高分辨率放大。"""
        expr = '${inputs.size}.endswith("2K")'
        assert evaluate_condition(expr, workflow_ctx)

    def test_scenario_pe_encode(self, workflow_ctx):
        """场景：是否启用 PE 编码。"""
        expr = '${inputs.use_pe} == true'
        assert evaluate_condition(expr, workflow_ctx)

    def test_scenario_lora_apply(self, workflow_ctx):
        """场景：是否应用 LoRA。"""
        expr = 'len(${loras}) > 0'
        assert evaluate_condition(expr, workflow_ctx)

    def test_scenario_lora_specific_name(self, workflow_ctx):
        """场景：LoRA 名称匹配。"""
        # 检查 loras 列表中是否包含 name=style_xfer 的项
        # 注意：DSL 不支持复杂对象比较，只能检查 loras.names
        # 这里需要调用方提供 loras.names 字段
        workflow_ctx["loras_names"] = ["style_xfer", "color_xfer"]
        expr = 'contains(${loras_names}, "style_xfer")'
        assert evaluate_condition(expr, workflow_ctx)

    def test_scenario_variant_turbo(self, workflow_ctx):
        """场景：variant 是 turbo 系列。"""
        expr = '${variant}.contains("turbo")'
        assert evaluate_condition(expr, workflow_ctx)

    def test_scenario_combined(self, workflow_ctx):
        """场景：组合条件。"""
        expr = (
            'len(${input_images.output.images}) > 0 '
            'and ${inputs.use_pe} == true '
            'and ${variant}.startswith("ernie")'
        )
        assert evaluate_condition(expr, workflow_ctx)

    def test_scenario_skip_node_when_no_images(self, workflow_ctx):
        """场景：无图输入时跳过 IPAdapter。"""
        # 修改 context 模拟无图
        workflow_ctx["input_images"]["output"]["images"] = []
        expr = 'len(${input_images.output.images}) > 0'
        assert not evaluate_condition(expr, workflow_ctx)

    def test_scenario_skip_optional_negative_prompt(self, workflow_ctx):
        """场景：negative_prompt 不存在或为空时跳过。"""
        # 存在且非空
        expr = 'exists(${inputs.negative_prompt}) and ${inputs.negative_prompt} != ""'
        assert evaluate_condition(expr, workflow_ctx)

        # 删除字段
        del workflow_ctx["inputs"]["negative_prompt"]
        expr = 'not_exists(${inputs.negative_prompt}) or ${inputs.negative_prompt} == ""'
        assert evaluate_condition(expr, workflow_ctx)


# ===========================================================================
# Token 测试
# ===========================================================================


class TestToken:
    """Token dataclass 测试。"""

    def test_token_is_dataclass(self):
        assert is_dataclass(Token)

    def test_token_fields(self):
        t = Token(type="NUMBER", value="42", pos=0)
        assert t.type == "NUMBER"
        assert t.value == "42"
        assert t.pos == 0

    def test_token_equality(self):
        t1 = Token("NUMBER", "42", 0)
        t2 = Token("NUMBER", "42", 0)
        assert t1 == t2
