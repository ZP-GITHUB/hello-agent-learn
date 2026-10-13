"""领域服务层：每个模块对应深度研究工作流里的一段职责。

    planner.py        ③ 规划层：主题 → TodoItem 列表（含三级容错解析）
    search.py         ④ 执行层之检索：dispatch_search / prepare_research_context
    summarizer.py     ④ 执行层之总结：summarize_task / stream_task_summary
    reporter.py       ⑤ 汇总层：把任务总结拼成最终报告
    notes.py           笔记协作指引生成（内联进总结提示词）
    tool_events.py     工具调用事件收集与 SSE 载荷转换（旁路通道）
    text_processing.py 文本清洗：strip_tool_calls
"""

