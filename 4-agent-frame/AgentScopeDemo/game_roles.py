# -*- coding: utf-8 -*-
"""三国狼人杀角色定义

本模块定义游戏中的所有角色身份和三国角色性格：
- ROLES: 游戏阵营角色（狼人/预言家/女巫/猎人/村民/守护者），包含技能描述和胜利条件
- CHARACTER_TRAITS: 三国名将性格特征，影响智能体的说话风格
- get_standard_setup(): 根据玩家人数生成标准角色配置
"""
from typing import Dict, List


class GameRoles:
    """游戏角色管理类

    管理两类数据：
    1. ROLES - 游戏阵营角色定义（狼人/预言家/女巫等），包含技能、胜利条件、阵营归属
    2. CHARACTER_TRAITS - 三国名将性格特征，用于提示词中塑造智能体的说话风格

    提供类方法用于查询角色信息、判断阵营、生成标准配置。
    """

    # 游戏阵营角色定义
    # 每个角色包含：description(描述)、ability(技能)、win_condition(胜利条件)、team(阵营归属)
    ROLES = {
        "狼人": {
            "description": "狼人",
            "ability": "夜晚可以击杀一名玩家",
            "win_condition": "消灭所有好人或与好人数量相等",
            "team": "狼人阵营"
        },
        "预言家": {
            "description": "预言家",
            "ability": "每晚可以查验一名玩家的身份",
            "win_condition": "消灭所有狼人",
            "team": "好人阵营"
        },
        "女巫": {
            "description": "女巫",
            "ability": "拥有解药和毒药各一瓶，可以救人或杀人",
            "win_condition": "消灭所有狼人",
            "team": "好人阵营"
        },
        "猎人": {
            "description": "猎人",
            "ability": "被投票出局时可以开枪带走一名玩家",
            "win_condition": "消灭所有狼人",
            "team": "好人阵营"
        },
        "村民": {
            "description": "村民",
            "ability": "无特殊技能，依靠推理和投票",
            "win_condition": "消灭所有狼人",
            "team": "好人阵营"
        },
        "守护者": {
            "description": "守护者",
            "ability": "每晚可以守护一名玩家免受狼人攻击",
            "win_condition": "消灭所有狼人",
            "team": "好人阵营"
        }
    }

    # 三国名将性格特征
    # 这些描述会被插入到提示词中，让智能体以特定历史人物的口吻说话
    # 例如：曹操会更犀利，诸葛亮会更谨慎
    CHARACTER_TRAITS = {
        "刘备": "仁德宽厚，善于团结众人，说话温和有礼",
        "关羽": "忠义刚烈，言辞直接，重情重义",
        "张飞": "性格豪爽，说话大声直接，容易冲动",
        "诸葛亮": "智慧超群，分析透彻，言辞谨慎",
        "赵云": "忠勇双全，话语简洁有力",
        "曹操": "雄才大略，善于权谋，话语犀利",
        "司马懿": "深谋远虑，城府极深，言辞含蓄",
        "周瑜": "才华横溢，略显傲气，分析精准",
        "孙权": "年轻有为，善于决断，话语果决"
    }

    @classmethod
    def get_role_desc(cls, role: str) -> str:
        """获取角色描述"""
        return cls.ROLES.get(role, {}).get("description", "未知角色")

    @classmethod
    def get_role_ability(cls, role: str) -> str:
        """获取角色技能"""
        return cls.ROLES.get(role, {}).get("ability", "无特殊技能")

    @classmethod
    def get_character_trait(cls, character: str) -> str:
        """获取角色性格特点"""
        return cls.CHARACTER_TRAITS.get(character, "性格温和，说话得体")

    @classmethod
    def is_werewolf(cls, role: str) -> bool:
        """判断是否为狼人"""
        return role == "狼人"

    @classmethod
    def is_villager_team(cls, role: str) -> bool:
        """判断是否为好人阵营"""
        return cls.ROLES.get(role, {}).get("team") == "好人阵营"

    @classmethod
    def get_standard_setup(cls, player_count: int) -> List[str]:
        """根据玩家人数生成标准角色配置

        标准配置规则：
        - 6人局:  2狼人 + 1预言家 + 1女巫 + 2村民
        - 8人局:  3狼人 + 1预言家 + 1女巫 + 1猎人 + 2村民
        - 9人局:  3狼人 + 1预言家 + 1女巫 + 1猎人 + 1守护者 + 2村民
        - 其他人数: 自动计算，约1/3狼人，依次分配神职，剩余为村民

        Args:
            player_count: 玩家总数

        Returns:
            角色身份列表，如 ["狼人", "狼人", "预言家", "女巫", "村民", "村民"]
        """
        if player_count == 6:
            return ["狼人", "狼人", "预言家", "女巫", "村民", "村民"]
        elif player_count == 8:
            return ["狼人", "狼人", "狼人", "预言家", "女巫", "猎人", "村民", "村民"]
        elif player_count == 9:
            return ["狼人", "狼人", "狼人", "预言家", "女巫", "猎人", "守护者", "村民", "村民"]
        else:
            # 默认配置：约1/3狼人
            werewolf_count = max(1, player_count // 3)
            roles = ["狼人"] * werewolf_count

            # 添加神职
            remaining = player_count - werewolf_count
            if remaining >= 1:
                roles.append("预言家")
                remaining -= 1
            if remaining >= 1:
                roles.append("女巫")
                remaining -= 1
            if remaining >= 1:
                roles.append("猎人")
                remaining -= 1

            # 剩余为村民
            roles.extend(["村民"] * remaining)

            return roles