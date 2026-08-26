# -*- coding: utf-8 -*-
"""三国狼人杀游戏工具函数

本模块提供游戏运行所需的基础工具：
- 常量定义：游戏轮数限制、讨论轮数限制、三国名将列表
- 工具函数：角色名获取、玩家列表格式化、投票统计、胜利判定
- GameModerator 类：游戏主持人，继承 AgentBase，负责发布各类游戏公告
- 辅助函数：发言模式分析、可疑度计算、中断处理
"""
import asyncio
import random
from typing import List, Dict, Optional, Any
from collections import Counter

from agentscope.agent import AgentBase   # AgentScope 智能体基类
from agentscope.message import Msg        # AgentScope 消息对象，智能体间通信的基本单元

# ---- 游戏常量 ----
MAX_GAME_ROUND = 10          # 最大游戏轮数（每轮 = 夜晚 + 白天）
MAX_DISCUSSION_ROUND = 3     # 每个讨论阶段的最大轮数（狼人讨论/白天讨论）
# 三国名将候选池：每局从中随机抽取玩家，包含魏蜀吴三方势力
CHINESE_NAMES = [
    "刘备", "关羽", "张飞", "诸葛亮", "赵云",
    "曹操", "司马懿", "典韦", "许褚", "夏侯惇",
    "孙权", "周瑜", "陆逊", "甘宁", "太史慈",
    "吕布", "貂蝉", "董卓", "袁绍", "袁术"
]


def get_chinese_name(character: str = None) -> str:
    """获取中文角色名

    如果传入的角色名在候选池中，直接返回；否则随机选择一个。
    用于将三国角色名与智能体绑定。

    Args:
        character: 指定的三国角色名，可为 None

    Returns:
        确定的中文角色名
    """
    if character and character in CHINESE_NAMES:
        return character
    return random.choice(CHINESE_NAMES)


def format_player_list(players: List[AgentBase], show_roles: bool = False) -> str:
    """将智能体列表格式化为中文可读字符串

    Args:
        players: 智能体列表
        show_roles: 是否显示角色信息（默认 False，仅显示名字）

    Returns:
        用中文顿号分隔的玩家名字符串，如 "刘备、关羽、张飞"
    """
    if not players:
        return "无玩家"

    if show_roles:
        return "、".join([f"{p.name}({getattr(p, 'role', '未知')})" for p in players])
    else:
        return "、".join([p.name for p in players])


def majority_vote_cn(votes: Dict[str, str]) -> tuple[str, int]:
    """多数投票统计：返回得票最多的玩家及其票数

    使用 collections.Counter 统计各目标得票数，取最多的一个。
    用于狼人击杀投票和白天淘汰投票。

    Args:
        votes: {投票者名字: 被投目标名字} 的字典

    Returns:
        (得票最多的玩家名, 得票数)
    """
    if not votes:
        return "无人", 0

    vote_counts = Counter(votes.values())
    most_voted = vote_counts.most_common(1)[0]

    return most_voted[0], most_voted[1]


def check_winning_cn(alive_players: List[AgentBase], roles: Dict[str, str]) -> Optional[str]:
    """检查游戏胜利条件

    胜利规则：
    - 好人胜利：狼人数量 == 0（所有狼人被淘汰）
    - 狼人胜利：狼人数量 >= 好人数量（狼人势力已压倒好人）

    Args:
        alive_players: 当前存活的智能体列表
        roles: {玩家名: 身份} 的映射字典

    Returns:
        胜利信息字符串，若游戏未结束则返回 None
    """
    alive_roles = [roles.get(p.name, "村民") for p in alive_players]
    werewolf_count = alive_roles.count("狼人")
    villager_count = len(alive_roles) - werewolf_count

    if werewolf_count == 0:
        return "好人阵营胜利！所有狼人已被淘汰！"
    elif werewolf_count >= villager_count:
        return "狼人阵营胜利！狼人数量已达到或超过好人！"

    return None


def analyze_speech_pattern(speech: str) -> Dict[str, Any]:
    """分析智能体发言模式（中文关键词匹配）

    通过中文关键词统计，分析发言的：
    - 置信度：包含"确定""肯定"等词的数量
    - 怀疑度：包含"可能""也许"等词的数量
    - 情感倾向：正面词 - 负面词的差值

    注意：此函数当前未被主流程调用，可作为后续扩展的预留。

    Args:
        speech: 智能体的发言文本

    Returns:
        包含 word_count, confidence_keywords, doubt_keywords, emotion_score 的字典
    """
    analysis = {
        "word_count": len(speech),
        "confidence_keywords": 0,
        "doubt_keywords": 0,
        "emotion_score": 0
    }

    # 中文关键词分析
    confidence_words = ["确定", "肯定", "一定", "绝对", "必须", "显然"]
    doubt_words = ["可能", "也许", "或许", "怀疑", "不确定", "感觉"]

    for word in confidence_words:
        analysis["confidence_keywords"] += speech.count(word)

    for word in doubt_words:
        analysis["doubt_keywords"] += speech.count(word)

    # 简单情感分析
    positive_words = ["好", "棒", "赞", "支持", "同意"]
    negative_words = ["坏", "差", "反对", "不行", "错误"]

    for word in positive_words:
        analysis["emotion_score"] += speech.count(word)

    for word in negative_words:
        analysis["emotion_score"] -= speech.count(word)

    return analysis


class GameModerator(AgentBase):
    """游戏主持人（继承 AgentBase）

    职责：发布各类游戏公告，作为系统消息发送给所有智能体。
    继承 AgentBase 使其可以参与 AgentScope 的消息系统，
    通过 Msg 对象与智能体通信。

    公告类型：
    - announce: 通用公告
    - night_announcement: 夜晚开始公告
    - day_announcement: 白天开始公告
    - death_announcement: 死亡公告
    - vote_result_announcement: 投票结果公告
    - game_over_announcement: 游戏结束公告
    """

    def __init__(self) -> None:
        super().__init__()          # 初始化 AgentBase 基类
        self.name = "游戏主持人"    # 主持人名称，显示在消息中
        self.game_log: List[str] = []  # 游戏日志，记录所有公告内容

    async def announce(self, content: str) -> Msg:
        """发布通用游戏公告

        创建一条 system 角色的 Msg 消息，记录到日志并输出到终端。
        其他智能体可以通过 observe() 接收此消息。

        Args:
            content: 公告内容

        Returns:
            创建的 Msg 消息对象
        """
        msg = Msg(
            name=self.name,
            content=f"📢 {content}",
            role="system"
        )
        self.game_log.append(content)
        await self.print(msg)
        return msg

    async def night_announcement(self, round_num: int) -> Msg:
        """夜晚阶段公告"""
        content = f"🌙 第{round_num}夜降临，天黑请闭眼..."
        return await self.announce(content)

    async def day_announcement(self, round_num: int) -> Msg:
        """白天阶段公告"""
        content = f"☀️ 第{round_num}天天亮了，请大家睁眼..."
        return await self.announce(content)

    async def death_announcement(self, dead_players: List[str]) -> Msg:
        """死亡公告"""
        if not dead_players:
            content = "昨夜平安无事，无人死亡。"
        else:
            content = f"昨夜，{format_player_list_str(dead_players)}不幸遇害。"
        return await self.announce(content)

    async def vote_result_announcement(self, voted_out: str, vote_count: int) -> Msg:
        """投票结果公告"""
        content = f"投票结果：{voted_out}以{vote_count}票被淘汰出局。"
        return await self.announce(content)

    async def game_over_announcement(self, winner: str) -> Msg:
        """游戏结束公告"""
        content = f"🎉 游戏结束！{winner}"
        return await self.announce(content)


def format_player_list_str(players: List[str]) -> str:
    """将玩家名字列表格式化为中文顿号分隔的字符串

    与 format_player_list 的区别：
    - 本函数接收字符串列表（玩家名字）
    - format_player_list 接收 AgentBase 列表（智能体对象）

    Args:
        players: 玩家名字列表

    Returns:
        用顿号分隔的字符串，如 "刘备、关羽"
    """
    if not players:
        return "无人"
    return "、".join(players)


def calculate_suspicion_score(player_name: str, game_history: List[Dict]) -> float:
    """计算玩家可疑度分数（0.0 ~ 1.0）

    根据历史事件累计可疑度：
    - 被投票 +0.3
    - 被指控 +0.2
    - 自我辩护 -0.1（降低可疑度）

    注意：此函数当前未被主流程调用，可作为后续扩展的预留。

    Args:
        player_name: 要计算的玩家名
        game_history: 游戏事件历史列表，每个事件包含 type/target/player 等字段

    Returns:
        可疑度分数，范围 [0.0, 1.0]
    """
    score = 0.0

    for event in game_history:
        if event.get("type") == "vote" and event.get("target") == player_name:
            score += 0.3
        elif event.get("type") == "accusation" and event.get("target") == player_name:
            score += 0.2
        elif event.get("type") == "defense" and event.get("player") == player_name:
            score -= 0.1

    return min(max(score, 0.0), 1.0)


async def handle_interrupt(*args: Any, **kwargs: Any) -> Msg:
    """处理游戏中断事件

    当游戏被外部中断时，返回一条系统消息。
    注意：此函数当前未被主流程调用，可作为后续扩展的预留。

    Returns:
        中断通知消息
    """
    return Msg(
        name="系统",
        content="游戏被中断",
        role="system"
    )