import json
import requests
import jinja2
import os
import datetime
from typing import List

MAX_ITER_NUM = 10

API_KEY = os.getenv("QWEN_LLM_API_KEY")
URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

TEMPLATE = """Respond to the human as helpfully and accurately as possible. 

{{instruction}}

You have access to the following tools:

{{tools}}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or one of the [{{tool_names}}]

Provide only ONE action per $JSON_BLOB, as shown:

```
{
  "action": $TOOL_NAME,
  "action_input": $ACTION_INPUT
}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "Final response to human"
}
```

Previous conversation history:
{{historic_messages}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.

Question: {{query}}
Thought: {{agent_scratchpad}}"""

INTRODUCTION = "You are a helpful assistant."

TOOLS_DESC = """[
  {
    "type": "function",
    "function": {
      "name": "get_current_time",
      "description": "当你想知道现在的时间时非常有用。",
      "input_schema": {},
      "out_schema": {
        "time": {
          "type": "string",
          "description": "格式为: YYYY-mm-dd HH:MM:SS.ff"
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_location_weather",
      "description": "当你想查询指定城市的天气时非常有用。",
      "input_schema": {
        "city": {
          "type": "string",
          "description": "城市名称",
          "required": true
        }
      },
      "out_schema": {
        "weather_describe": {
          "type": "string",
          "description": "城市当天的天气描述"
        },
      }
    }
  }
]
"""

TOOLS_NAMES = "get_current_time", "get_location_weathe"


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_location_weather(city: str) -> str:
    return f"{city}今日暴雨，气温23℃。"


def query_model(prompt: str) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    body = {
        "model": "qwen-max",
        "input": {"prompt": prompt},
        "stream": True,
        "stop": "\nObservation",  # 停止词
        "extra_body": {"enable_search": False},  # 关闭互联网搜索
    }

    try:
        resp = requests.post(URL, headers=headers, json=body, timeout=60)
        resp_json = resp.json()
        # print(resp_json)
    except requests.exceptions.RequestException as e:
        print(f"LLM返回异常: {e}")
        if "resp" in locals():
            print(f"status_code: {resp.status_code}, text: {resp.text}")
        return ""
    except json.JSONDecodeError as e:
        print(f"JSON解析异常: {e}")
        if "resp" in locals():
            print(f"status_code: {resp.status_code}, text: {resp.text}")
        return ""

    input_token, output_tokens, total_tokens = (
        resp_json["usage"]["input_tokens"],
        resp_json["usage"]["output_tokens"],
        resp_json["usage"]["total_tokens"],
    )
    content = resp_json["output"]["text"]

    print("-----------------LLM返回开始-----------------")
    print(f"Prompt:\n{prompt}\n")
    print(f"LLM response:\n{content}")
    print(
        f"input_token: {input_token}, output_tokens: {output_tokens}, total_tokens: {total_tokens}"
    )
    print("-----------------LLM返回结束-----------------")

    return content


def agent_execute(query: str, historic_messages: List) -> str:
    iter_num: int = 0
    answer: str = ""
    content: str = ""
    react: str = ""
    agent_scratchpad: str = ""
    historic: str = "\n".join(historic_messages)
    action: str = ""
    action_input: str = ""
    observation: str = ""

    print(
        "-------------------------------------ReAct开始-------------------------------------"
    )

    # 执行思考链，直到思考出最终答案或超过最大迭代次数
    while action != "Final Answer" and iter_num < MAX_ITER_NUM:
        # 拼接模板
        prompt = jinja2.Template(TEMPLATE).render(
            query=query,
            instruction=INTRODUCTION,
            tools=TOOLS_DESC,
            tool_names=TOOLS_NAMES,
            agent_scratchpad=agent_scratchpad,
            historic_messages=historic,
        )

        # 调用大模型
        content = query_model(prompt)

        # 解析模型返回结果
        react = content.split("Action:\n```")
        if len(react) != 2:
            print("LLM返回格式异常")
            print(f"content: {content}")
            break

        thought = react[0].strip()
        action_data = react[1].replace("```", "").strip()
        action_dict = json.loads(action_data)
        action = action_dict["action"]
        action_input = action_dict["action_input"]

        # print("**************************************")
        # print(f"react: {react}, thought: {thought}, action_data: {action_data}")
        # print("**************************************")

        # 如果是最终答案，直接返回
        if action == "Final Answer":
            answer = action_input
            break

        # 调用工具记录结果
        if action == "get_current_time":
            observation = get_current_time()
        elif action == "get_location_weather":
            city = action_input["city"]
            observation = get_location_weather(city)
        else:
            print("工具不存在")
            break

        agent_scratchpad = (
            agent_scratchpad
            + thought
            + "\n"
            + action_data
            + "\n"
            + "Observation: "
            + observation
            + "\n"
            + "Thought: "
        )

        iter_num += 1

    print(
        "-------------------------------------ReAct结束-------------------------------------"
    )

    return answer


historic_messages: List = []
while True:
    query = input("Query: ")
    answer = agent_execute(query, historic_messages)
    historic_messages.append(f"Query: {query}\nAnswer: {answer}\n")

    print(answer)
