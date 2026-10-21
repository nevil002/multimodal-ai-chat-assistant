# mcp_client.py
import asyncio, threading
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPToolbox:
    def __init__(self, servers: dict[str, StdioServerParameters]):
        self.servers = servers
        self.sessions, self.schemas = {}, []          # tool name -> session, OpenAI-format schemas
        self._ready, self._error, self._stop = threading.Event(), None, None
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._main(), self.loop)
        if not self._ready.wait(timeout=30):
            raise RuntimeError("MCP servers did not start within 30 seconds")
        if self._error:
            raise RuntimeError(f"MCP startup failed: {self._error}") from self._error

    async def _main(self):
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:     # all connections live in this one task
                for params in self.servers.values():
                    read, write = await stack.enter_async_context(stdio_client(params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    for t in (await session.list_tools()).tools:
                        self.sessions[t.name] = session
                        self.schemas.append({"type": "function", "function": {
                            "name": t.name, "description": t.description or "",
                            "parameters": t.inputSchema}})
                self._ready.set()
                await self._stop.wait()               # keep connections open until close()
        except Exception as e:
            self._error = e
            self._ready.set()

    def call(self, name, arguments):
        async def _call():
            res = await self.sessions[name].call_tool(name, arguments)
            text = "\n".join(c.text for c in res.content if hasattr(c, "text"))
            return f"Tool error: {text}" if res.isError else text
        return asyncio.run_coroutine_threadsafe(_call(), self.loop).result(timeout=60)

    def close(self):
        self.loop.call_soon_threadsafe(self._stop.set)