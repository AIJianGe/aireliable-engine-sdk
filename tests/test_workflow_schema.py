"""WorkflowSchema 单元测试（v0.2 阶段 4 任务 4.5）。

覆盖任务 4.5 验收标准：
    - yaml 加载时严格校验
    - 错误信息清晰

参考：
    - v0.2-tasks.md 任务 4.5
    - PR/architecture.md §12.4 工作流 graph 结构
    - PR/development-guide.md §29 工作流 yaml 规范
"""

from __future__ import annotations

import pytest
import yaml as yaml_lib

from aireliable_engine.workflow_schema import (
    BranchDef,
    NodeDef,
    WorkflowSchema,
    WorkflowSchemaError,
    validate_workflow_dict,
    validate_workflow_yaml,
)


# ===========================================================================
# fixtures
# ===========================================================================

@pytest.fixture
def basic_workflow_dict() -> dict:
    """最小合法工作流 dict。"""
    return {
        "name": "test_workflow",
        "description": "测试工作流",
        "nodes": [
            {
                "id": "input_prompt",
                "type": "TextInput",
                "pos": [100, 100],
                "params": {"field": "prompt"},
            },
            {
                "id": "main_model",
                "type": "ModelLoader",
                "params": {
                    "variant": "${variant}",
                    "cache_dir": "models/image/checkpoints/ernie-image/",
                },
            },
            {
                "id": "clip_encode",
                "type": "CLIPTextEncode",
                "inputs": {
                    "text": "input_prompt.output.text",
                    "clip": "main_model.output.clip",
                },
            },
        ],
        "outputs": {
            "embeds": "clip_encode.output.embeds",
        },
    }


@pytest.fixture
def basic_workflow_yaml(basic_workflow_dict) -> str:
    """最小合法工作流 yaml 字符串。"""
    return yaml_lib.safe_dump(basic_workflow_dict, allow_unicode=True, sort_keys=False)


@pytest.fixture
def ernie_image_workflow_yaml() -> str:
    """ERNIE-Image 完整工作流示例（来自 PR/architecture.md §12.4.1）。"""
    return """
name: ernie_image_default
description: ERNIE-Image 默认文生图工作流

nodes:
  - id: input_images
    type: ImagesInput
    pos: [100, 100]
    params: {}

  - id: input_prompt
    type: TextInput
    pos: [100, 250]
    params:
      field: prompt

  - id: main_model
    type: ModelLoader
    pos: [400, 100]
    params:
      variant: "${variant}"
      cache_dir: "models/image/checkpoints/ernie-image/"

  - id: clip_text_encode
    type: CLIPTextEncode
    pos: [400, 250]
    inputs:
      text: input_prompt.output.text
      clip: main_model.output.clip

  - id: ipadapter_apply
    type: IPAdapterApply
    pos: [700, 100]
    inputs:
      images: input_images.output.images
      model: main_model.output.model
    condition: "len(${input_images.output.images}) > 0"

  - id: sampler
    type: Sampler
    pos: [1000, 100]
    inputs:
      model: main_model.output.model
      cond: clip_text_encode.output.embeds

  - id: vae_decode
    type: VAEDecode
    pos: [1300, 100]
    inputs:
      latent: sampler.output.latent
      vae: main_model.output.vae

  - id: save
    type: ImageSave
    pos: [1600, 100]
    inputs:
      images: vae_decode.output.images

outputs:
  images: save.output.images
"""


# ===========================================================================
# NodeDef 测试
# ===========================================================================

class TestNodeDef:
    """节点定义校验测试。"""

    def test_minimal_node(self):
        """最小节点定义（仅 id + type）。"""
        node = NodeDef(id="n1", type="Echo")
        assert node.id == "n1"
        assert node.type == "Echo"
        assert node.pos is None
        assert node.params == {}
        assert node.inputs == {}
        assert node.condition is None

    def test_full_node(self):
        """完整节点定义。"""
        node = NodeDef(
            id="main_model",
            type="ModelLoader",
            pos=[100, 200],
            params={"variant": "${variant}", "cache_dir": "/tmp"},
            inputs={"text": "input_prompt.output.text"},
            condition="len(${inputs.images}) > 0",
        )
        assert node.pos == [100, 200]
        assert node.params["variant"] == "${variant}"
        assert node.inputs["text"] == "input_prompt.output.text"
        assert node.condition == "len(${inputs.images}) > 0"

    def test_id_underscore_prefix(self):
        """id 以下划线开头合法。"""
        node = NodeDef(id="_internal_node", type="Echo")
        assert node.id == "_internal_node"

    def test_id_with_numbers(self):
        """id 含数字合法（非开头）。"""
        node = NodeDef(id="node_123", type="Echo")
        assert node.id == "node_123"

    def test_id_empty_string_raises(self):
        """id 空字符串报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "", "type": "Echo"}],
                "outputs": {},
            })

    def test_id_starts_with_digit_raises(self):
        """id 以数字开头报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "1node", "type": "Echo"}],
                "outputs": {},
            })

    def test_id_with_special_chars_raises(self):
        """id 含特殊字符报错。"""
        for bad_id in ["n-d", "n.d", "n d", "n@d", "n#d"]:
            with pytest.raises(WorkflowSchemaError):
                validate_workflow_dict({
                    "name": "w",
                    "nodes": [{"id": bad_id, "type": "Echo"}],
                    "outputs": {},
                })

    def test_type_empty_raises(self):
        """type 空字符串报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": ""}],
                "outputs": {},
            })

    def test_pos_wrong_length_raises(self):
        """pos 长度不为 2 报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "pos": [100]}],
                "outputs": {},
            })
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "pos": [100, 200, 300]}],
                "outputs": {},
            })

    def test_pos_non_int_raises(self):
        """pos 含非 int 报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "pos": ["100", 200]}],
                "outputs": {},
            })

    def test_pos_bool_rejected(self):
        """pos 不接受 bool（虽然 bool 是 int 子类）。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "pos": [True, 200]}],
                "outputs": {},
            })

    def test_condition_empty_string_raises(self):
        """condition 空字符串报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "condition": ""}],
                "outputs": {},
            })

    def test_condition_whitespace_only_raises(self):
        """condition 仅空白报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "condition": "   "}],
                "outputs": {},
            })

    def test_params_non_dict_raises(self):
        """params 非 dict 报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "params": "not a dict"}],
                "outputs": {},
            })

    def test_inputs_non_dict_raises(self):
        """inputs 非 dict 报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "inputs": "not a dict"}],
                "outputs": {},
            })

    def test_extra_field_rejected(self):
        """未知字段被拒绝（extra='forbid'）。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "unknown_field": "value"}],
                "outputs": {},
            })

    def test_inputs_accept_arbitrary_values(self):
        """inputs 接受任意类型值（str 引用 / dict / list / int / None）。"""
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [{
                "id": "n1",
                "type": "Echo",
                "inputs": {
                    "str_ref": "node1.output.text",
                    "var_ref": "${inputs.text}",
                    "literal_str": "hello",
                    "literal_int": 42,
                    "literal_list": [1, 2, 3],
                    "literal_dict": {"k": "v"},
                    "literal_null": None,
                },
            }],
            "outputs": {},
        })
        assert schema.nodes[0].inputs["literal_int"] == 42
        assert schema.nodes[0].inputs["literal_list"] == [1, 2, 3]


# ===========================================================================
# WorkflowSchema 顶层校验
# ===========================================================================

class TestWorkflowSchemaTopLevel:
    """工作流顶层字段校验。"""

    def test_minimal_valid(self, basic_workflow_dict):
        """最小合法工作流。"""
        schema = validate_workflow_dict(basic_workflow_dict)
        assert schema.name == "test_workflow"
        assert schema.description == "测试工作流"
        assert len(schema.nodes) == 3
        assert schema.outputs["embeds"] == "clip_encode.output.embeds"

    def test_missing_name_raises(self, basic_workflow_dict):
        """缺少 name 报错。"""
        del basic_workflow_dict["name"]
        with pytest.raises(WorkflowSchemaError) as exc_info:
            validate_workflow_dict(basic_workflow_dict)
        assert "name" in str(exc_info.value).lower() or "字段" in str(exc_info.value)

    def test_empty_name_raises(self):
        """name 空字符串报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "",
                "nodes": [{"id": "n1", "type": "Echo"}],
                "outputs": {},
            })

    def test_missing_nodes_raises(self):
        """缺少 nodes 报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "outputs": {},
            })

    def test_empty_nodes_raises(self):
        """nodes 空列表报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [],
                "outputs": {},
            })

    def test_missing_outputs_defaults_empty(self, basic_workflow_dict):
        """缺少 outputs 默认为空 dict。"""
        del basic_workflow_dict["outputs"]
        schema = validate_workflow_dict(basic_workflow_dict)
        assert schema.outputs == {}

    def test_missing_description_defaults_empty(self, basic_workflow_dict):
        """缺少 description 默认为空字符串。"""
        del basic_workflow_dict["description"]
        schema = validate_workflow_dict(basic_workflow_dict)
        assert schema.description == ""

    def test_description_non_str_raises(self, basic_workflow_dict):
        """description 非 str 报错。"""
        basic_workflow_dict["description"] = 123
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(basic_workflow_dict)

    def test_extra_top_field_rejected(self, basic_workflow_dict):
        """顶层未知字段被拒绝。"""
        basic_workflow_dict["unknown_top_field"] = "value"
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(basic_workflow_dict)


# ===========================================================================
# 节点 id 唯一性
# ===========================================================================

class TestNodeIdUniqueness:
    """节点 id 唯一性校验。"""

    def test_duplicate_ids_raises(self):
        """重复 id 报错。"""
        with pytest.raises(WorkflowSchemaError) as exc_info:
            validate_workflow_dict({
                "name": "w",
                "nodes": [
                    {"id": "dup", "type": "Echo"},
                    {"id": "dup", "type": "Echo"},
                ],
                "outputs": {},
            })
        assert "重复" in str(exc_info.value) or "duplicate" in str(exc_info.value).lower()

    def test_three_duplicate_raises(self):
        """3 个重复 id 报错（不只第一个）。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({
                "name": "w",
                "nodes": [
                    {"id": "n", "type": "Echo"},
                    {"id": "n", "type": "Echo"},
                    {"id": "n", "type": "Echo"},
                ],
                "outputs": {},
            })

    def test_unique_ids_ok(self):
        """不同 id 通过。"""
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [
                {"id": "n1", "type": "Echo"},
                {"id": "n2", "type": "Echo"},
                {"id": "n3", "type": "Echo"},
            ],
            "outputs": {},
        })
        assert len(schema.nodes) == 3


# ===========================================================================
# 分支定义校验
# ===========================================================================

class TestBranchDef:
    """分支定义校验。"""

    def test_valid_branch(self, basic_workflow_dict):
        """合法分支。"""
        basic_workflow_dict["branches"] = [
            {
                "name": "use_pe",
                "condition": "${inputs.use_pe} == true",
                "nodes": [{"id": "pe_encode", "type": "PEEncode"}],
                "outputs": {"text": "pe_encode.output.text"},
            },
        ]
        schema = validate_workflow_dict(basic_workflow_dict)
        assert len(schema.branches) == 1
        assert schema.branches[0].name == "use_pe"
        assert len(schema.branches[0].nodes) == 1

    def test_branch_name_empty_raises(self, basic_workflow_dict):
        """分支 name 空字符串报错。"""
        basic_workflow_dict["branches"] = [
            {"name": "", "condition": "true"},
        ]
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(basic_workflow_dict)

    def test_branch_name_with_special_chars_raises(self, basic_workflow_dict):
        """分支 name 含特殊字符报错。"""
        basic_workflow_dict["branches"] = [
            {"name": "branch-1", "condition": "true"},
        ]
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(basic_workflow_dict)

    def test_branch_condition_empty_raises(self, basic_workflow_dict):
        """分支 condition 空字符串报错。"""
        basic_workflow_dict["branches"] = [
            {"name": "b", "condition": ""},
        ]
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(basic_workflow_dict)

    def test_duplicate_branch_names_raises(self, basic_workflow_dict):
        """重复分支名报错。"""
        basic_workflow_dict["branches"] = [
            {"name": "branch_a", "condition": "true"},
            {"name": "branch_a", "condition": "false"},
        ]
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(basic_workflow_dict)

    def test_branch_extra_field_rejected(self, basic_workflow_dict):
        """分支未知字段被拒绝。"""
        basic_workflow_dict["branches"] = [
            {"name": "b", "condition": "true", "unknown": "value"},
        ]
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(basic_workflow_dict)

    def test_branch_node_id_can_duplicate_with_top_level(self, basic_workflow_dict):
        """分支内节点 id 与顶层节点重复允许（分支独立作用域）。

        当前 schema 层不跨作用域校验，留给 WorkflowExecutor。
        """
        basic_workflow_dict["branches"] = [
            {
                "name": "b",
                "condition": "true",
                "nodes": [{"id": "input_prompt", "type": "Echo"}],
            },
        ]
        # schema 层不报错（跨作用域校验由执行器负责）
        schema = validate_workflow_dict(basic_workflow_dict)
        assert len(schema.branches[0].nodes) == 1


# ===========================================================================
# 包装格式 / 非包装格式
# ===========================================================================

class TestWorkflowWrapperFormat:
    """{'workflow': {...}} 包装格式与直接格式兼容测试。"""

    def test_direct_format(self, basic_workflow_dict):
        """直接格式：顶层就是工作流字段。"""
        schema = validate_workflow_dict(basic_workflow_dict)
        assert schema.name == "test_workflow"

    def test_wrapped_format(self, basic_workflow_dict):
        """包装格式：{'workflow': {...}}。"""
        wrapped = {"workflow": basic_workflow_dict}
        schema = validate_workflow_dict(wrapped)
        assert schema.name == "test_workflow"

    def test_wrapped_with_extra_top_field_raises(self, basic_workflow_dict):
        """包装格式 + 顶层额外字段报错。"""
        wrapped = {"workflow": basic_workflow_dict, "extra": "value"}
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(wrapped)

    def test_wrapped_with_non_dict_inner_raises(self):
        """包装格式但内部不是 dict 报错。"""
        wrapped = {"workflow": "not a dict"}
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(wrapped)

    def test_non_dict_input_raises(self):
        """非 dict 输入报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict("not a dict")  # type: ignore

    def test_none_input_raises(self):
        """None 输入报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict(None)  # type: ignore

    def test_list_input_raises(self):
        """list 输入报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict([1, 2, 3])  # type: ignore


# ===========================================================================
# yaml 字符串校验
# ===========================================================================

class TestWorkflowYaml:
    """yaml 字符串解析测试。"""

    def test_valid_yaml_direct(self, basic_workflow_yaml):
        """直接格式 yaml。"""
        schema = validate_workflow_yaml(basic_workflow_yaml)
        assert schema.name == "test_workflow"

    def test_valid_yaml_wrapped(self, basic_workflow_dict):
        """包装格式 yaml。"""
        wrapped = {"workflow": basic_workflow_dict}
        yaml_text = yaml_lib.safe_dump(wrapped, allow_unicode=True, sort_keys=False)
        schema = validate_workflow_yaml(yaml_text)
        assert schema.name == "test_workflow"

    def test_ernie_image_full_workflow(self, ernie_image_workflow_yaml):
        """完整 ERNIE-Image 工作流通过校验。"""
        schema = validate_workflow_yaml(ernie_image_workflow_yaml)
        assert schema.name == "ernie_image_default"
        assert len(schema.nodes) == 8
        assert "images" in schema.outputs
        # 检查 condition 字段
        ipadapter = next(n for n in schema.nodes if n.id == "ipadapter_apply")
        assert ipadapter.condition is not None
        assert "len(" in ipadapter.condition

    def test_empty_yaml_raises(self):
        """空 yaml 报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_yaml("")

    def test_whitespace_only_yaml_raises(self):
        """仅空白 yaml 报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_yaml("   \n  \t  \n")

    def test_non_string_input_raises(self):
        """非字符串输入报错。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_yaml(123)  # type: ignore
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_yaml(None)  # type: ignore
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_yaml({"name": "x"})  # type: ignore

    def test_invalid_yaml_syntax_raises(self):
        """非法 yaml 语法报错。"""
        bad_yaml = "name: [unclosed"
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_yaml(bad_yaml)

    def test_yaml_with_comments(self):
        """含注释的 yaml 正常解析。"""
        yaml_text = """
# 工作流注释
name: test  # 行尾注释

nodes:
  - id: n1  # 节点注释
    type: Echo
outputs: {}
"""
        schema = validate_workflow_yaml(yaml_text)
        assert schema.name == "test"

    def test_yaml_unicode(self):
        """yaml 含 Unicode 字符（name / description 用中文，节点 id 仍用 ASCII）。"""
        yaml_text = """
name: 测试工作流
description: 包含中文描述
nodes:
  - id: node_1
    type: Echo
outputs: {}
"""
        schema = validate_workflow_yaml(yaml_text)
        assert schema.name == "测试工作流"
        assert schema.description == "包含中文描述"


# ===========================================================================
# 错误信息清晰度
# ===========================================================================

class TestErrorMessages:
    """错误信息清晰度测试。"""

    def test_error_contains_field_path(self):
        """错误信息包含字段路径。"""
        with pytest.raises(WorkflowSchemaError) as exc_info:
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "1bad", "type": "Echo"}],
                "outputs": {},
            })
        err = exc_info.value
        # 错误信息包含 id 相关路径
        assert "id" in str(err) or "格式" in str(err)

    def test_error_for_missing_required_field(self):
        """缺少必填字段的错误信息包含字段名。"""
        with pytest.raises(WorkflowSchemaError) as exc_info:
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"type": "Echo"}],  # 缺 id
                "outputs": {},
            })
        assert "id" in str(exc_info.value).lower() or "field" in str(exc_info.value).lower()

    def test_error_for_wrong_type_value(self):
        """类型错误的错误信息包含字段名和实际类型。"""
        with pytest.raises(WorkflowSchemaError) as exc_info:
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "n1", "type": "Echo", "pos": "wrong"}],
                "outputs": {},
            })
        err_msg = str(exc_info.value)
        assert "pos" in err_msg.lower()

    def test_error_aggregates_multiple_errors(self):
        """多个错误时聚合显示。"""
        with pytest.raises(WorkflowSchemaError) as exc_info:
            validate_workflow_dict({
                "name": "w",
                "nodes": [
                    {"id": "1bad", "type": ""},  # 两个错误：id 格式 + type 空
                ],
                "outputs": {},
            })
        err_msg = str(exc_info.value)
        # 至少包含一处错误信息
        assert err_msg

    def test_error_is_value_error_subclass(self):
        """WorkflowSchemaError 是 ValueError 子类。"""
        with pytest.raises(WorkflowSchemaError):
            validate_workflow_dict({"name": "w", "nodes": []})
        # ValueError 兼容
        with pytest.raises(ValueError):
            validate_workflow_dict({"name": "w", "nodes": []})


# ===========================================================================
# 转换方法
# ===========================================================================

class TestConversionMethods:
    """to_dict / to_yaml_dict / to_yaml 转换测试。"""

    def test_to_dict(self, basic_workflow_dict):
        """to_dict 返回普通 dict（不含 workflow 包装层）。"""
        schema = validate_workflow_dict(basic_workflow_dict)
        d = schema.to_dict()
        assert isinstance(d, dict)
        assert "workflow" not in d
        assert d["name"] == "test_workflow"
        assert len(d["nodes"]) == 3
        assert d["outputs"]["embeds"] == "clip_encode.output.embeds"

    def test_to_yaml_dict(self, basic_workflow_dict):
        """to_yaml_dict 返回 {'workflow': {...}} 包装格式。"""
        schema = validate_workflow_dict(basic_workflow_dict)
        d = schema.to_yaml_dict()
        assert isinstance(d, dict)
        assert "workflow" in d
        assert d["workflow"]["name"] == "test_workflow"

    def test_to_yaml_string(self, basic_workflow_dict):
        """to_yaml 返回 yaml 字符串。"""
        schema = validate_workflow_dict(basic_workflow_dict)
        yaml_text = schema.to_yaml()
        assert isinstance(yaml_text, str)
        # 重新解析应该等价
        re_schema = validate_workflow_yaml(yaml_text)
        assert re_schema.name == schema.name
        assert len(re_schema.nodes) == len(schema.nodes)

    def test_to_dict_roundtrip(self, basic_workflow_dict):
        """dict -> schema -> dict 往返一致（除默认值外）。"""
        schema = validate_workflow_dict(basic_workflow_dict)
        d = schema.to_dict()
        # 用转换后的 dict 再次校验
        schema2 = validate_workflow_dict(d)
        assert schema2.name == schema.name
        assert len(schema2.nodes) == len(schema.nodes)
        assert schema2.outputs == schema.outputs

    def test_to_yaml_roundtrip(self, basic_workflow_yaml):
        """yaml -> schema -> yaml -> schema 往返一致。"""
        schema1 = validate_workflow_yaml(basic_workflow_yaml)
        yaml_text = schema1.to_yaml()
        schema2 = validate_workflow_yaml(yaml_text)
        assert schema2.name == schema1.name
        assert len(schema2.nodes) == len(schema1.nodes)
        assert schema2.outputs == schema1.outputs


# ===========================================================================
# 边界场景
# ===========================================================================

class TestEdgeCases:
    """边界场景测试。"""

    def test_empty_outputs_ok(self):
        """空 outputs dict 合法。"""
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [{"id": "n1", "type": "Echo"}],
            "outputs": {},
        })
        assert schema.outputs == {}

    def test_outputs_with_literal_values(self):
        """outputs 支持字面值（非 str）。"""
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [{"id": "n1", "type": "Echo"}],
            "outputs": {
                "int_val": 42,
                "list_val": [1, 2, 3],
                "dict_val": {"k": "v"},
                "null_val": None,
                "bool_val": True,
            },
        })
        assert schema.outputs["int_val"] == 42
        assert schema.outputs["list_val"] == [1, 2, 3]
        assert schema.outputs["null_val"] is None
        assert schema.outputs["bool_val"] is True

    def test_params_with_arbitrary_values(self):
        """params 支持任意 JSON 兼容值。"""
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [{
                "id": "n1",
                "type": "Echo",
                "params": {
                    "str_val": "hello",
                    "int_val": 42,
                    "float_val": 3.14,
                    "bool_val": True,
                    "null_val": None,
                    "list_val": [1, "two", [3]],
                    "dict_val": {"nested": {"deep": "value"}},
                    "var_ref": "${inputs.field}",
                },
            }],
            "outputs": {},
        })
        p = schema.nodes[0].params
        assert p["str_val"] == "hello"
        assert p["int_val"] == 42
        assert p["list_val"] == [1, "two", [3]]
        assert p["dict_val"]["nested"]["deep"] == "value"

    def test_pos_zero_values(self):
        """pos 接受 0 值。"""
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [{"id": "n1", "type": "Echo", "pos": [0, 0]}],
            "outputs": {},
        })
        assert schema.nodes[0].pos == [0, 0]

    def test_negative_pos_values(self):
        """pos 接受负值。"""
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [{"id": "n1", "type": "Echo", "pos": [-100, -200]}],
            "outputs": {},
        })
        assert schema.nodes[0].pos == [-100, -200]

    def test_large_workflow(self):
        """大型工作流（100+ 节点）通过。"""
        nodes = [{"id": f"n{i}", "type": "Echo"} for i in range(100)]
        schema = validate_workflow_dict({
            "name": "large",
            "nodes": nodes,
            "outputs": {"r": "n99.output.value"},
        })
        assert len(schema.nodes) == 100

    def test_node_id_max_length(self):
        """节点 id 接受长名称。"""
        long_id = "a" * 100
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [{"id": long_id, "type": "Echo"}],
            "outputs": {},
        })
        assert schema.nodes[0].id == long_id

    def test_node_id_unicode_in_type(self):
        """节点 type 含 Unicode 字符（不限制）。"""
        schema = validate_workflow_dict({
            "name": "w",
            "nodes": [{"id": "n1", "type": "中文节点"}],
            "outputs": {},
        })
        assert schema.nodes[0].type == "中文节点"


# ===========================================================================
# 完整场景
# ===========================================================================

class TestScenarios:
    """真实工作流场景测试。"""

    def test_ernie_image_workflow_with_branches(self, ernie_image_workflow_yaml):
        """ERNIE-Image 工作流含 PE 分支。"""
        # 加载基础工作流
        schema = validate_workflow_yaml(ernie_image_workflow_yaml)
        # 添加 PE 分支
        schema.branches = [
            BranchDef(
                name="with_pe",
                condition="${inputs.use_pe} == true",
                nodes=[NodeDef(id="pe_encode", type="PEEncode")],
                outputs={"text": "pe_encode.output.text"},
            ),
            BranchDef(
                name="without_pe",
                condition="${inputs.use_pe} == false",
                nodes=[],
                outputs={},
            ),
        ]
        # 转换并重新校验
        d = schema.to_dict()
        schema2 = validate_workflow_dict(d)
        assert len(schema2.branches) == 2
        assert schema2.branches[0].name == "with_pe"
        assert schema2.branches[1].name == "without_pe"

    def test_lora_injection_workflow(self):
        """LoRA 注入工作流。"""
        yaml_text = """
name: lora_workflow
description: 带可选 LoRA 的工作流
nodes:
  - id: main_model
    type: ModelLoader
    params:
      variant: "${variant}"
  - id: lora_loader
    type: LoRALoader
    params:
      loras: "${loras}"
    condition: "len(${loras}) > 0"
  - id: apply_lora
    type: ApplyLoRA
    inputs:
      model: main_model.output.model
      loras: lora_loader.output.loras
    condition: "len(${loras}) > 0"
  - id: sampler
    type: Sampler
    inputs:
      model: main_model.output.model
outputs:
  result: sampler.output.image
"""
        schema = validate_workflow_yaml(yaml_text)
        assert len(schema.nodes) == 4
        assert schema.nodes[1].condition is not None

    def test_multi_image_input_workflow(self):
        """多图输入工作流。"""
        yaml_text = """
name: multi_image
nodes:
  - id: images_input
    type: ImagesInput
    params: {}
  - id: ipadapter
    type: IPAdapterApply
    inputs:
      images: images_input.output.images
      model: main_model.output.model
    condition: "len(${images_input.output.images}) > 0"
  - id: main_model
    type: ModelLoader
    params:
      variant: "${variant}"
outputs:
  images: ipadapter.output.images
"""
        schema = validate_workflow_yaml(yaml_text)
        assert len(schema.nodes) == 3

    def test_conditional_node_chain(self):
        """条件节点链。"""
        yaml_text = """
name: conditional_chain
nodes:
  - id: check_input
    type: CheckInput
    params:
      field: "${inputs.use_high_quality}"
  - id: high_quality_path
    type: HighQualityProcessor
    condition: "${inputs.use_high_quality} == true"
  - id: low_quality_path
    type: LowQualityProcessor
    condition: "${inputs.use_high_quality} == false"
  - id: merge
    type: MergeResults
    inputs:
      high: high_quality_path.output.result
      low: low_quality_path.output.result
outputs:
  final: merge.output.result
"""
        schema = validate_workflow_yaml(yaml_text)
        assert len(schema.nodes) == 4
        conditions = [n.condition for n in schema.nodes if n.condition]
        assert len(conditions) == 2

    def test_workflow_with_default_values(self):
        """工作流含默认值语法。"""
        yaml_text = """
name: with_defaults
nodes:
  - id: config_loader
    type: ConfigLoader
    params:
      steps: "${inputs.steps or 50}"
      guidance: "${inputs.guidance or 7.5}"
      seed: "${inputs.seed or -1}"
outputs:
  config: config_loader.output.config
"""
        schema = validate_workflow_yaml(yaml_text)
        params = schema.nodes[0].params
        assert "or 50" in params["steps"]
        assert "or 7.5" in params["guidance"]


# ===========================================================================
# WorkflowSchemaError 属性
# ===========================================================================

class TestWorkflowSchemaError:
    """WorkflowSchemaError 异常类测试。"""

    def test_error_has_field_path(self):
        """错误对象包含 field_path 属性。"""
        with pytest.raises(WorkflowSchemaError) as exc_info:
            validate_workflow_dict({
                "name": "w",
                "nodes": [{"id": "1bad", "type": "Echo"}],
                "outputs": {},
            })
        err = exc_info.value
        assert hasattr(err, "field_path")
        assert isinstance(err.field_path, str)

    def test_error_str_includes_field_path(self):
        """错误字符串包含字段路径。"""
        err = WorkflowSchemaError("test error", field_path="nodes[0].id")
        s = str(err)
        assert "nodes[0].id" in s
        assert "test error" in s

    def test_error_without_field_path(self):
        """无字段路径的错误字符串。"""
        err = WorkflowSchemaError("test error")
        s = str(err)
        assert "test error" in s
        assert "[" not in s  # 无路径前缀

    def test_error_is_value_error(self):
        """WorkflowSchemaError 是 ValueError 子类。"""
        err = WorkflowSchemaError("test")
        assert isinstance(err, ValueError)

    def test_error_raised_in_try_except(self):
        """WorkflowSchemaError 可被 try/except ValueError 捕获。"""
        with pytest.raises(ValueError):
            try:
                validate_workflow_dict({"name": "w", "nodes": []})
            except WorkflowSchemaError:
                # 验证 WorkflowSchemaError 可作为 ValueError 被 except 捕获
                raise ValueError("re-raised as ValueError")
