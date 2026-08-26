# -*- coding: utf-8 -*-
"""
三国狼人杀 - 基于AgentScope的中文版狼人杀游戏
融合三国演义角色和传统狼人杀玩法

整体架构：
    ThreeKingdomsWerewolfGame 是游戏主类，负责：
    1. 创建并管理所有玩家智能体（ReActAgent）
    2. 按「夜晚 → 白天」循环驱动游戏流程
    3. 每个阶段通过 AgentScope 的 pipeline / MsgHub 协调多智能体交互
    4. 通过结构化输出（Pydantic Model）约束智能体返回可解析的行动结果
"""
import asyncio
import os
import random
from typing import List, Dict, Optional

from dotenv import load_dotenv

# 从 .env 文件加载 API 密钥等环境变量（必须在 import agentscope 之前完成）
load_dotenv()

# ---- AgentScope 核心组件 ----
# ReActAgent: 具备「推理-行动」循环能力的智能体，支持 observe（被动接收消息）和 __call__（主动行动）
from agentscope.agent import ReActAgent
# OpenAIChatModel: 兼容 OpenAI API 格式的模型客户端，支持第三方模型服务
from agentscope.model import OpenAIChatModel
# MsgHub: 消息广播中心，在 with 块内将智能体的发言自动广播给组内所有成员
# sequential_pipeline: 让多个智能体按顺序依次执行（用于白天轮流发言）
# fanout_pipeline: 让多个智能体并行执行同一消息（用于投票、击杀等同时行动的场景）
from agentscope.pipeline import MsgHub, sequential_pipeline, fanout_pipeline
# OpenAIMultiAgentFormatter: 多智能体对话格式器，将多轮多方对话整理为模型可理解的上下文
from agentscope.formatter import OpenAIMultiAgentFormatter

# ---- 本地模块导入 ----
from prompt_cn import ChinesePrompts              # 各角色的系统提示词模板
from game_roles import GameRoles                  # 角色定义（阵营、技能、性格）
from structured_output_cn import (                # 结构化输出模型（约束智能体返回格式）
    DiscussionModelCN,                            #   讨论阶段的输出格式
    get_vote_model_cn,                            #   投票模型工厂函数（动态生成 Literal 类型）
    WitchActionModelCN,                           #   女巫行动模型
    get_seer_model_cn,                            #   预言家查验模型工厂函数
    get_hunter_model_cn,                          #   猎人开枪模型工厂函数
    WerewolfKillModelCN                           #   狼人击杀模型
)
from utils_cn import (                            # 工具函数与常量
    check_winning_cn,                             #   胜利条件判定
    majority_vote_cn,                             #   多数投票统计
    get_chinese_name,                             #   获取中文角色名
    format_player_list,                           #   格式化玩家列表
    GameModerator,                                #   游戏主持人（继承 AgentBase）
    MAX_GAME_ROUND,                               #   最大游戏轮数（10）
    MAX_DISCUSSION_ROUND,                         #   每轮白天最大讨论轮数（3）
)


class ThreeKingdomsWerewolfGame:
    """三国狼人杀游戏主类

    游戏状态管理：
        - players: 所有玩家的智能体字典 {名字: ReActAgent}
        - roles: 玩家身份映射 {名字: 角色名}，用于预言家查验和胜利判定
        - alive_players / werewolves / villagers / seer / witch / hunter: 各阵营存活智能体列表
        - witch_has_antidote / witch_has_poison: 女巫道具的一次性使用状态

    游戏流程：
        setup_game() → 循环 [夜晚阶段 → 检查胜利 → 白天阶段 → 检查胜利]
    """

    def __init__(self):
        # 玩家智能体存储
        self.players: Dict[str, ReActAgent] = {}    # {角色名: 智能体} 全量映射
        self.roles: Dict[str, str] = {}             # {角色名: 身份} 如 {"刘备": "狼人"}
        self.moderator = GameModerator()            # 游戏主持人，负责发布公告
        self.alive_players: List[ReActAgent] = []   # 当前存活的所有玩家

        # 按阵营分类的存活玩家列表
        self.werewolves: List[ReActAgent] = []      # 狼人阵营
        self.villagers: List[ReActAgent] = []       # 普通村民
        self.seer: List[ReActAgent] = []            # 预言家（只有1个）
        self.witch: List[ReActAgent] = []           # 女巫（只有1个）
        self.hunter: List[ReActAgent] = []          # 猎人（只有1个）

        # 女巫道具状态（各只能使用一次）
        self.witch_has_antidote = True              # 解药：救活被杀玩家
        self.witch_has_poison = True                # 毒药：毒杀一名玩家

    async def create_player(self, role: str, character: str) -> ReActAgent:
        """创建具有三国背景的玩家智能体

        流程：
        1. 用角色名 + 系统提示词 + 模型配置 创建一个 ReActAgent
        2. 通过 observe() 让智能体被动接收自己的身份信息（不触发回复）
        3. 注册到 players 字典并返回

        Args:
            role: 游戏身份，如 "狼人"、"预言家"、"女巫"、"猎人"、"村民"
            character: 三国角色名，如 "刘备"、"曹操" 等

        Returns:
            创建好的 ReActAgent 实例
        """
        name = get_chinese_name(character)
        self.roles[name] = role  # 记录身份映射，后续用于胜利判定和预言家查验

        # 创建 ReActAgent 智能体
        # - sys_prompt: 系统提示词，定义角色行为规则（包含三国性格 + 游戏策略）
        # - model: OpenAIChatModel 客户端，通过 client_args 传递 base_url 以支持第三方 API
        # - formatter: OpenAIMultiAgentFormatter 多智能体对话格式器，管理多轮对话上下文
        agent = ReActAgent(
            name=name,
            sys_prompt=ChinesePrompts.get_role_prompt(role, character),
            model=OpenAIChatModel(
                model_name=os.getenv("LLM_MODEL_ID", "gpt-4o"),
                api_key=os.getenv("LLM_API_KEY"),
                client_args={
                    # base_url 需通过 client_args 传入底层 OpenAI 客户端（agentscope 1.0.2 API）
                    "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                },
            ),
            formatter=OpenAIMultiAgentFormatter(),
        )

        # 通过 observe() 让智能体知道自己的身份
        # observe 是「被动接收」，只将消息加入上下文，不触发智能体回复
        await agent.observe(
            await self.moderator.announce(
                f"【{name}】你在这场三国狼人杀中扮演{GameRoles.get_role_desc(role)}，"
                f"你的角色是{character}。{GameRoles.get_role_ability(role)}"
            )
        )

        self.players[name] = agent
        return agent

    async def setup_game(self, player_count: int = 6):
        """设置游戏：分配角色、创建智能体、分组

        流程：
        1. 根据人数获取角色配置（如6人局：2狼人+1预言家+1女巫+2村民）
        2. 从9位三国名将中随机抽取 player_count 位
        3. 逐一创建智能体，并按身份分配到对应阵营列表
        4. 发布游戏开始公告

        Args:
            player_count: 玩家人数，默认6人局
        """
        print("🎮 开始设置三国狼人杀游戏...")

        # 获取角色身份列表（如 ["狼人", "狼人", "预言家", "女巫", "村民", "村民"]）
        roles = GameRoles.get_standard_setup(player_count)
        # 从9位三国名将中随机抽取，保证每局角色组合不同
        characters = random.sample([
            "刘备", "关羽", "张飞", "诸葛亮", "赵云",
            "曹操", "司马懿", "周瑜", "孙权"
        ], player_count)

        # 逐一创建玩家智能体，并按身份分组
        for i, (role, character) in enumerate(zip(roles, characters)):
            agent = await self.create_player(role, character)
            self.alive_players.append(agent)

            # 按身份分配到对应阵营列表（用于后续分阶段处理）
            if role == "狼人":
                self.werewolves.append(agent)
            elif role == "预言家":
                self.seer.append(agent)
            elif role == "女巫":
                self.witch.append(agent)
            elif role == "猎人":
                self.hunter.append(agent)
            else:
                self.villagers.append(agent)

        # 向所有智能体广播游戏开始公告
        await self.moderator.announce(
            f"三国狼人杀游戏开始！参与者：{format_player_list(self.alive_players)}"
        )

        print(f"✅ 游戏设置完成，共{len(self.alive_players)}名玩家")

    async def werewolf_phase(self, round_num: int):
        """狼人阶段：讨论 + 投票选择击杀目标

        流程：
        1. 创建 MsgHub（消息广播中心），让狼人之间的发言自动互相可见
        2. 进行 MAX_DISCUSSION_ROUND 轮讨论，每轮每只狼发言一次
        3. 关闭自动广播后，用 fanout_pipeline 让每只狼独立投票选择击杀目标
        4. 统计投票，多数决确定最终击杀目标

        关键概念：
        - MsgHub(enable_auto_broadcast=True): 智能体发言后，其他成员自动收到该消息
        - fanout_pipeline: 多个智能体并行处理同一消息，各自独立返回结果
        - structured_model: 约束智能体返回 Pydantic 模型格式，便于程序解析

        Args:
            round_num: 当前轮数

        Returns:
            被击杀玩家的名字，若无狼人则返回 None
        """
        if not self.werewolves:
            return None

        await self.moderator.announce(f"🐺 狼人请睁眼，选择今晚要击杀的目标...")

        # MsgHub 创建一个消息广播环境
        # announcement 参数作为进入 MsgHub 时的首条系统消息
        # enable_auto_broadcast=True 表示每只狼的发言会自动广播给其他狼
        async with MsgHub(
                self.werewolves,
                enable_auto_broadcast=True,
                announcement=await self.moderator.announce(
                    f"狼人们，请讨论今晚的击杀目标。存活玩家：{format_player_list(self.alive_players)}"
                ),
        ) as werewolves_hub:
            # 讨论阶段：每只狼轮流发言 MAX_DISCUSSION_ROUND 轮
            # 调用 agent(structured_model=...) 会触发智能体推理并返回结构化结果
            for _ in range(MAX_DISCUSSION_ROUND):
                for wolf in self.werewolves:
                    await wolf(structured_model=DiscussionModelCN)

            # 投票阶段：关闭自动广播，避免投票结果互相影响
            werewolves_hub.set_auto_broadcast(False)
            # fanout_pipeline 让每只狼独立投票（并行执行，互不干扰）
            # enable_gather=False 表示不自动收集结果，手动处理返回值
            kill_votes = await fanout_pipeline(
                self.werewolves,
                msg=await self.moderator.announce("请选择击杀目标"),
                structured_model=WerewolfKillModelCN,
                enable_gather=False,
            )

            # 统计投票结果
            votes = {}
            for i, vote_msg in enumerate(kill_votes):
                # 防御性检查：LLM 可能返回异常结果，需验证 metadata 是否存在
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.werewolves[i].name] = vote_msg.metadata.get("target")
                else:
                    # 投票无效时，随机选择一个非狼人目标作为兜底
                    print(f"⚠️ {self.werewolves[i].name} 的击杀投票无效,随机选择目标")
                    import random
                    valid_targets = [p.name for p in self.alive_players if
                                     p.name not in [w.name for w in self.werewolves]]
                    votes[self.werewolves[i].name] = random.choice(valid_targets) if valid_targets else None

            # 多数决：得票最多的玩家被击杀
            killed_player, _ = majority_vote_cn(votes)
            return killed_player

    async def seer_phase(self):
        """预言家阶段：选择一名玩家查验其身份

        流程：
        1. 让预言家智能体通过 structured_model 选择查验目标
        2. 从 self.roles 中查找目标的真实身份
        3. 通过 observe() 将查验结果告知预言家（仅预言家自己知道）

        注意：
        - get_seer_model_cn() 是工厂函数，根据当前存活玩家动态生成 Literal 类型
          确保预言家只能选择存活的玩家作为查验目标
        """
        if not self.seer:
            return

        seer_agent = self.seer[0]
        await self.moderator.announce("🔮 预言家请睁眼，选择要查验的玩家...")

        # 调用智能体并传入 structured_model，返回结果的 metadata 中包含结构化字段
        check_result = await seer_agent(
            structured_model=get_seer_model_cn(self.alive_players)
        )

        # 防御性检查：LLM 可能返回无效结果
        if check_result is None or not hasattr(check_result, 'metadata') or check_result.metadata is None:
            print(f"⚠️ 预言家查验失败,跳过此阶段")
            return

        target_name = check_result.metadata.get("target")
        if not target_name:
            print(f"⚠️ 预言家未选择查验目标,跳过此阶段")
            return

        # 从身份映射中查找目标真实身份（默认"村民"，防止 KeyError）
        target_role = self.roles.get(target_name, "村民")

        # 通过 observe() 将查验结果告知预言家
        # observe 是被动接收，不会触发预言家的额外回复
        result_msg = f"查验结果：{target_name}是{'狼人' if target_role == '狼人' else '好人'}"
        await seer_agent.observe(await self.moderator.announce(result_msg))

    async def witch_phase(self, killed_player: str):
        """女巫阶段：得知今晚被杀玩家，决定是否使用解药/毒药

        流程：
        1. 通过 observe() 告知女巫今晚的死亡信息
        2. 让女巫智能体通过 WitchActionModelCN 决定行动
        3. 根据返回的 metadata 处理解药救人和毒药杀人

        规则：
        - 解药和毒药各只能使用一次（通过 witch_has_antidote/witch_has_poison 控制）
        - 女巫可以看到今晚被杀的人，然后决定是否用解药救
        - 女巫也可以额外使用毒药毒杀一人

        Args:
            killed_player: 今晚被狼人击杀的玩家名（可能为 None 表示无人被杀）

        Returns:
            (final_killed, poisoned_player): 最终死亡玩家和毒杀玩家的元组
        """
        if not self.witch:
            return killed_player, None

        witch_agent = self.witch[0]
        await self.moderator.announce("🧙‍♀️ 女巫请睁眼...")

        # 告知女巫今晚谁被杀了（这是女巫决策的关键信息）
        death_info = f"今晚{killed_player}被狼人击杀" if killed_player else "今晚平安无事"
        await witch_agent.observe(await self.moderator.announce(death_info))

        # 女巫决定行动：是否使用解药/毒药
        witch_action = await witch_agent(structured_model=WitchActionModelCN)

        saved_player = None
        poisoned_player = None

        # 防御性检查：LLM 可能返回无效结果，视为不使用技能
        if witch_action is None or not hasattr(witch_action, 'metadata') or witch_action.metadata is None:
            print(f"⚠️ 女巫行动失败,视为不使用技能")
        else:
            # 处理解药：use_antidote=True 且解药未使用 且确实有人被杀
            if witch_action.metadata.get("use_antidote") and self.witch_has_antidote:
                if killed_player:
                    saved_player = killed_player
                    self.witch_has_antidote = False  # 标记解药已用，不可再用
                    await witch_agent.observe(await self.moderator.announce(f"你使用解药救了{killed_player}"))

            # 处理毒药：use_poison=True 且毒药未使用
            if witch_action.metadata.get("use_poison") and self.witch_has_poison:
                poisoned_player = witch_action.metadata.get("target_name")
                if poisoned_player:
                    self.witch_has_poison = False  # 标记毒药已用，不可再用
                    await witch_agent.observe(await self.moderator.announce(f"你使用毒药毒杀了{poisoned_player}"))

        # 如果被救活了，则今晚无人死亡；否则狼人击杀的目标死亡
        final_killed = killed_player if not saved_player else None

        return final_killed, poisoned_player

    async def hunter_phase(self, shot_by_hunter: str):
        """猎人阶段：被投票出局时可以开枪带走一名玩家

        触发条件：仅当被投票淘汰的玩家恰好是猎人时才触发
        （狼人击杀不触发猎人技能，这是本游戏的规则设定）

        Args:
            shot_by_hunter: 本轮被投票淘汰的玩家名

        Returns:
            被猎人带走的玩家名，若未触发或放弃则返回 None
        """
        if not self.hunter:
            return None

        hunter_agent = self.hunter[0]
        # 只有被投票出局的玩家是猎人本人时，才触发技能
        if hunter_agent.name == shot_by_hunter:
            await self.moderator.announce("🏹 猎人发动技能，可以带走一名玩家...")

            # 猎人选择是否开枪以及目标
            hunter_action = await hunter_agent(
                structured_model=get_hunter_model_cn(self.alive_players)
            )

            # 防御性检查：无效结果视为放弃开枪
            if hunter_action is None or not hasattr(hunter_action, 'metadata') or hunter_action.metadata is None:
                print(f"⚠️ 猎人技能使用失败,视为放弃开枪")
                return None

            # shoot=True 表示选择开枪
            if hunter_action.metadata.get("shoot"):
                target = hunter_action.metadata.get("target")
                if target:
                    await self.moderator.announce(f"猎人{hunter_agent.name}开枪带走了{target}")
                    return target
                else:
                    print(f"⚠️ 猎人选择开枪但未指定目标,视为放弃")
                    return None

        return None

    def update_alive_players(self, dead_players: List[str]):
        """更新存活玩家列表：从所有分组中移除死亡玩家

        为什么需要更新所有列表？
        因为各阶段（werewolf_phase、day_phase 等）使用的是各自的分组列表，
        如果不从所有列表中移除，死亡玩家可能仍参与后续阶段的行动。

        Args:
            dead_players: 本轮死亡的玩家名字列表
        """
        for dead_name in dead_players:
            if dead_name:
                # 从全局存活列表移除
                self.alive_players = [p for p in self.alive_players if p.name != dead_name]
                # 从各阵营分组中移除（死亡玩家不再参与对应阶段的行动）
                self.werewolves = [p for p in self.werewolves if p.name != dead_name]
                self.villagers = [p for p in self.villagers if p.name != dead_name]
                self.seer = [p for p in self.seer if p.name != dead_name]
                self.witch = [p for p in self.witch if p.name != dead_name]
                self.hunter = [p for p in self.hunter if p.name != dead_name]

    async def day_phase(self, round_num: int):
        """白天阶段：全员讨论 + 投票淘汰

        流程：
        1. 创建 MsgHub 让所有存活玩家互相看到发言
        2. sequential_pipeline: 按顺序让每人发言一次（轮流执行）
        3. 关闭自动广播后，fanout_pipeline 让所有人同时投票（并行执行）
        4. 多数决确定淘汰玩家

        与狼人阶段的区别：
        - sequential_pipeline vs 手动循环：sequential_pipeline 是 AgentScope 内置的顺序执行管道
        - 参与者是所有存活玩家，而非仅狼人

        Args:
            round_num: 当前轮数

        Returns:
            被投票淘汰的玩家名
        """
        await self.moderator.day_announcement(round_num)

        # 全员讨论：MsgHub 让所有人的发言互相可见
        async with MsgHub(
                self.alive_players,
                enable_auto_broadcast=True,
                announcement=await self.moderator.announce(
                    f"现在开始自由讨论。存活玩家：{format_player_list(self.alive_players)}"
                ),
        ) as all_hub:
            # sequential_pipeline: 按列表顺序让每个智能体依次发言
            # 每个智能体发言后，由于 MsgHub 的自动广播，其他人都能看到
            await sequential_pipeline(self.alive_players)

            # 投票阶段：关闭自动广播，避免投票结果互相影响
            all_hub.set_auto_broadcast(False)
            # fanout_pipeline: 所有存活玩家并行投票
            # get_vote_model_cn() 动态生成投票模型，Literal 类型限定只能投存活玩家
            vote_msgs = await fanout_pipeline(
                self.alive_players,
                await self.moderator.announce("请投票选择要淘汰的玩家"),
                structured_model=get_vote_model_cn(self.alive_players),
                enable_gather=False,
            )

            # 统计投票结果
            votes = {}
            for i, vote_msg in enumerate(vote_msgs):
                # 防御性检查：无效投票视为弃票
                if vote_msg is not None and hasattr(vote_msg, 'metadata') and vote_msg.metadata is not None:
                    votes[self.alive_players[i].name] = vote_msg.metadata.get("vote")
                else:
                    print(f"⚠️ {self.alive_players[i].name} 的投票无效,视为弃票")
                    votes[self.alive_players[i].name] = None

            # 多数决：得票最多的玩家被淘汰
            voted_out, vote_count = majority_vote_cn(votes)
            await self.moderator.vote_result_announcement(voted_out, vote_count)

            return voted_out

    async def run_game(self):
        """运行游戏主循环

        游戏流程：
            setup_game()  →  初始化所有玩家智能体
            循环（最多 MAX_GAME_ROUND 轮）:
                夜晚阶段:
                    1. werewolf_phase  →  狼人讨论+投票击杀
                    2. seer_phase      →  预言家查验身份
                    3. witch_phase     →  女巫决定救/毒
                    4. 更新死亡玩家 + 公告
                    5. 检查胜利条件
                白天阶段:
                    6. day_phase       →  全员讨论+投票淘汰
                    7. hunter_phase    →  猎人技能（仅被投票出局时触发）
                    8. 更新死亡玩家
                    9. 检查胜利条件
        """
        try:
            # 初始化游戏：创建智能体、分配角色
            await self.setup_game()

            for round_num in range(1, MAX_GAME_ROUND + 1):
                print(f"\n🌙 === 第{round_num}轮游戏开始 ===")

                # ---- 夜晚阶段 ----
                await self.moderator.night_announcement(round_num)

                # 1. 狼人讨论并选择击杀目标
                killed_player = await self.werewolf_phase(round_num)

                # 2. 预言家查验一名玩家身份
                await self.seer_phase()

                # 3. 女巫决定是否使用解药/毒药
                final_killed, poisoned_player = await self.witch_phase(killed_player)

                # 4. 汇总夜晚死亡玩家并更新状态
                night_deaths = [p for p in [final_killed, poisoned_player] if p]
                self.update_alive_players(night_deaths)

                # 5. 公告夜晚结果
                await self.moderator.death_announcement(night_deaths)

                # 6. 检查是否满足胜利条件（狼人全灭 或 狼人数量≥好人）
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                # ---- 白天阶段 ----
                # 7. 全员讨论 + 投票淘汰
                voted_out = await self.day_phase(round_num)

                # 8. 猎人技能（仅当猎人被投票出局时触发）
                hunter_shot = await self.hunter_phase(voted_out)

                # 9. 汇总白天死亡玩家并更新状态
                day_deaths = [p for p in [voted_out, hunter_shot] if p]
                self.update_alive_players(day_deaths)

                # 10. 再次检查胜利条件
                winner = check_winning_cn(self.alive_players, self.roles)
                if winner:
                    await self.moderator.game_over_announcement(winner)
                    return

                print(f"第{round_num}轮结束，存活玩家：{format_player_list(self.alive_players)}")

        except Exception as e:
            print(f"❌ 游戏运行出错：{e}")
            import traceback
            traceback.print_exc()


async def main():
    """程序入口：校验环境变量 → 创建游戏实例 → 启动游戏循环

    使用 asyncio.run() 启动，因为 AgentScope 的 pipeline/MsgHub 都是异步 API。
    """
    # 启动前检查：必须有 API 密钥才能调用 LLM
    if not os.getenv("LLM_API_KEY"):
        print("❌ 请在 .env 文件中设置 LLM_API_KEY")
        return

    print("🎮 欢迎来到三国狼人杀！")

    # 创建并运行游戏
    game = ThreeKingdomsWerewolfGame()
    await game.run_game()


# Python 异步程序标准入口：asyncio.run() 创建事件循环并运行 main()
if __name__ == "__main__":
    asyncio.run(main())