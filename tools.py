
from config import MCP_SERVERS
from mcp_client import MCPToolbox
from rag.retrieval import search_kb

toolbox = MCPToolbox(MCP_SERVERS)

search_kb_function = {                     # schema for the local, session-scoped tool
    "name": "search_kb",
    "description": (
        "Search the user's uploaded documents. Use for any question about their content. "
        "Returns relevant extracts with file names. If the extracts don't contain the answer, "
        "call again with different search terms."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": ("Self-contained search query. Include the specific names, metrics and "
                                'terms from the conversation; never use pronouns like "it" or "that model".'),
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

def _mcp_func(name):
    return lambda **kw: toolbox.call(name, kw)

def build_tools(doc_ids):
    schemas = list(toolbox.schemas)                   # calculator, weather, anything else on the servers
    funcs = {name: _mcp_func(name) for name in toolbox.sessions}
    if doc_ids:                                       # local, session-scoped tool
        schemas.append({"type": "function", "function": search_kb_function})
        funcs["search_kb"] = lambda query: search_kb(query, doc_ids=doc_ids)
    return schemas, funcs

