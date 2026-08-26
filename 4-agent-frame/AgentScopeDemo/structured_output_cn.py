# -*- coding: utf-8 -*-
"""三国狼人杀游戏的结构化输出模型

结构化输出 (Structured Output) 是本项目的核心设计模式：
    通过 Pydantic BaseModel 定义智能体的返回格式，AgentScope 会将模型字段描述
    注入到提示词中，约束 LLM 返回可解析的 JSON，程序再通过 metadata 提取字段值。

模型分为两类：
    1. 静态类 (如 DiscussionModelCN, WitchActionModelCN)
       - 字段固定，不依赖运行时状态
       - 直接作为 structured_model 参数传入

    2. 工厂函数 (如 get_vote_model_cn, get_seer_model_cn, get_hunter_model_cn)
       - 需要动态生成 Literal 类型，限定智能体只能选择当前存活的玩家
       - 接收存活玩家列表，返回动态生成的 Pydantic 模型类
       - 例如：投票时只能投存活玩家，Literal 类型会在运行时限定可选值
"""
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from agentscope.agent import AgentBase


# ---- 静态模型：字段固定，不依赖运行时状态 ----

class DiscussionModelCN(BaseModel):
    """讨论阶段输出格式

    用于狼人和白天讨论阶段，智能体每次发言时返回：
    - reach_agreement: 是否同意当前讨论的结论
    - confidence_level: 对自己推理的信心程度
    - key_evidence: 支持观点的关键证据

    使用场景：main_cn.py 中 werewolf_phase 的讨论循环、day_phase 的发言阶段
    """

    reach_agreement: bool = Field(
        description="是否已达成一致意见",
    )
    confidence_level: int = Field(
        description="对当前推理的信心程度(1-10)",
        ge=1, le=10
    )
    key_evidence: Optional[str] = Field(
        description="支持你观点的关键证据",
        default=None
    )


# ---- 工厂函数：动态生成模型，用于需要限定可选玩家的场景 ----
# 为什么需要工厂函数？
# 因为投票、查验、开枪等操作需要限定智能体只能选择「当前存活的玩家」，
# 而 Literal 类型的可选值必须在类定义时确定，所以每次调用时动态生成。

def get_vote_model_cn(agents: list[AgentBase]) -> type[BaseModel]:
    """动态生成投票模型

    工厂函数：每次调用时根据当前存活玩家动态生成 Pydantic 模型类。
    关键设计：vote 字段的类型是 Literal[存活玩家名字...]，
    这样 LLM 只能从存活玩家中选择投票目标，避免选择已死亡的玩家。

    Args:
        agents: 当前存活的玩家智能体列表

    Returns:
        动态生成的 VoteModelCN 类

    使用场景：main_cn.py 中 day_phase 的投票阶段
    """

    class VoteModelCN(BaseModel):
        """动态投票模型

        vote 字段使用 Literal 类型，可选值在运行时由存活玩家名字确定。
        例如存活玩家是 ["刘备", "曹操", "关羽"]，则 vote 只能是这三个名字之一。
        """

        # Literal 动态限定：tuple(_.name for _ in agents) 生成存活玩家名字元组
        vote: Literal[tuple(_.name for _ in agents)] = Field(
            description="你要投票淘汰的玩家姓名",
        )
        reason: str = Field(
            description="投票理由，简要说明为什么选择此人",
        )
        suspicion_level: int = Field(
            description="对被投票者的怀疑程度(1-10)",
            ge=1, le=10
        )

    return VoteModelCN


class WitchActionModelCN(BaseModel):
    """女巫行动模型

    女巫每晚需要决定两件事：
    1. 是否使用解药救被杀的人（use_antidote）
    2. 是否使用毒药毒杀某人（use_poison + target_name）

    使用场景：main_cn.py 中 witch_phase
    """

    use_antidote: bool = Field(
        description="是否使用解药救人",
        default=False
    )
    use_poison: bool = Field(
        description="是否使用毒药杀人",
        default=False
    )
    target_name: Optional[str] = Field(
        description="目标玩家姓名（救人或毒杀的对象）",
        default=None
    )
    action_reason: Optional[str] = Field(
        description="行动理由",
        default=None
    )


def get_seer_model_cn(agents: list[AgentBase]) -> type[BaseModel]:
    """动态生成预言家查验模型

    与投票模型类似，预言家也只能选择存活玩家作为查验目标。

    Args:
        agents: 当前存活的玩家智能体列表

    Returns:
        动态生成的 SeerModelCN 类

    使用场景：main_cn.py 中 seer_phase
    """

    class SeerModelCN(BaseModel):
        """动态预言家查验模型，target 限定为存活玩家"""

        target: Literal[tuple(_.name for _ in agents)] = Field(
            description="要查验的玩家姓名",
        )
        check_reason: str = Field(
            description="查验此人的原因",
        )
        priority_level: int = Field(
            description="查验优先级(1-10)",
            ge=1, le=10
        )

    return SeerModelCN


def get_hunter_model_cn(agents: list[AgentBase]) -> type[BaseModel]:
    """动态生成猎人开枪模型

    猎人被投票出局时，可以选择开枪带走一名存活玩家。
    shoot=True 时必须指定 target。

    Args:
        agents: 当前存活的玩家智能体列表

    Returns:
        动态生成的 HunterModelCN 类

    使用场景：main_cn.py 中 hunter_phase
    """

    class HunterModelCN(BaseModel):
        """动态猎人开枪模型，target 限定为存活玩家"""

        shoot: bool = Field(
            description="是否使用开枪技能",
        )
        target: Optional[Literal[tuple(_.name for _ in agents)]] = Field(
            description="开枪目标玩家姓名",
            default=None
        )
        shoot_reason: Optional[str] = Field(
            description="开枪理由",
            default=None
        )

    return HunterModelCN


# ---- 静态模型：字段固定 ----

class WerewolfKillModelCN(BaseModel):
    """狼人击杀模型

    狼人讨论结束后，每只狼独立投票选择击杀目标。
    包含击杀策略和团队配合计划，让狼人的决策更有策略性。

    使用场景：main_cn.py 中 werewolf_phase 的投票击杀阶段
    """

    target: str = Field(
        description="要击杀的玩家姓名",
    )
    kill_strategy: str = Field(
        description="击杀策略说明",
    )
    team_coordination: Optional[str] = Field(
        description="与狼队友的配合计划",
        default=None
    )


class GameAnalysisModelCN(BaseModel):
    """游戏分析模型（当前未使用，预留扩展）

    设计用于让智能体在每轮结束后进行全局分析：
    - 怀疑的狼人名单
    - 信任的玩家名单
    - 关键线索汇总
    - 下一步策略

    注意：此模型当前未在 main_cn.py 中使用，可作为后续扩展的预留。
    """

    suspected_werewolves: List[str] = Field(
        description="怀疑的狼人名单",
        default_factory=list
    )
    trusted_players: List[str] = Field(
        description="信任的玩家名单",
        default_factory=list
    )
    key_clues: List[str] = Field(
        description="关键线索列表",
        default_factory=list
    )
    next_strategy: str = Field(
        description="下一步策略",
    )