from typing import Union, Sequence, Any

from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables.base import Runnable

from tool.tools import *  # type: ignore

# TODO 日志系统
TOOL_CALLS = "tool_calls"


def parse_tool_call_ai_message(ai_msg: BaseMessage):
    """ 解析 Agent 返回结果，若存在工具调用则调用工具，组装工具返回结果。若无工具调用则直接回复 Agent 返回答案 """
    assistant_output = ai_msg.additional_kwargs

    # 如果不需要调用工具，直接输出内容
    if TOOL_CALLS not in assistant_output.keys():
        print(f"===== 非调用工具，直接回复 =====")
        return ai_msg.content, False
    else:
        # 进入工具调用循环 => 最终返回工具执行结果
        tool_messages = []
        index = 0
        while len(assistant_output.get(TOOL_CALLS)) > index:
            tool_call = assistant_output.get(TOOL_CALLS)[index]
            tool_call_id = tool_call.get("id")
            function = tool_call.get("function")
            arguments = function.get("arguments")
            func_name = function.get("name")
            print(f"===== 正在调用工具 [{func_name}]，参数：{arguments} =====")
            # 当模型调用工具辅助输出时，需要手动获取模型返回的参数集，通过 eval 手动执行后将消息返回给大模型
            tool_result = eval(f'{func_name}(**{arguments})')

            # 构造工具返回信息
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": tool_result,  # 保持原始工具输出
            }
            tool_messages.append(tool_message)
            print(f"===== 工具返回：{tool_message} =====")
            index += 1
        return tool_messages, True


def tool_call_invoke(agent: Runnable[Union[
    PromptValue, str, Sequence[Union[BaseMessage, list[str], tuple[str, str], str, dict[str, Any]]]], BaseMessage],
                     messages: list):
    """ 调用工具与 Agent 沟通，若 Agent 判定无工具调用时返回最终交互答案 """
    ai_msg = agent.invoke(messages)
    while True:
        tool_call_response, is_call_tool = parse_tool_call_ai_message(ai_msg)
        if is_call_tool:
            # 再次和 Agent 通讯，将工具执行结果返回
            messages.extend(tool_call_response)
            print(f"===== 再次和 Agent 通讯 =====")
            ai_msg = agent.invoke(messages)
        else:
            return tool_call_response
