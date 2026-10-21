from config import model_gpt, model_groq, openai, groq,MAX_TOOL_ROUNDS
from  prompts import system_for
from tools import build_tools
import json

CLIENTS = {"Gpt": (openai, model_gpt), "Groq": (groq, model_groq)}

def handle_tool_calls(message, funcs):
    out = []
    for call in message.tool_calls:
        try:
            result = funcs[call.function.name](**json.loads(call.function.arguments))
        except Exception as e:
            result = f"Tool error: {e}"
        out.append({'role': 'tool', 'tool_call_id': call.id, 'content': str(result)})
    return out

def chat_with_tools(message, history, provider, doc_ids):
    client, model = CLIENTS[provider]
    tools, funcs = build_tools(doc_ids)
    history = [{'role': h['role'], 'content': h['content']}
               for h in history if isinstance(h['content'], str)]
    messages = ([{'role': 'system', 'content': system_for(doc_ids)}]
                + history + [{'role': 'user', 'content': message}])
    for _ in range(MAX_TOOL_ROUNDS):
        msg = client.chat.completions.create(model=model, messages=messages, tools=tools).choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        messages.extend(handle_tool_calls(msg, funcs))
    return client.chat.completions.create(model=model, messages=messages).choices[0].message.content