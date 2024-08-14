import operator
from typing import Annotated, Sequence, TypedDict

import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import (AIMessage, AnyMessage, HumanMessage,
                                     SystemMessage)
from langchain_core.prompts import (AIMessagePromptTemplate,
                                    ChatPromptTemplate, MessagesPlaceholder,
                                    SystemMessagePromptTemplate)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class ChatState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], operator.add]


async def chat_node(state: ChatState, config: RunnableConfig) -> ChatState:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You're a careful thinker. Answer in spanish."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    llm = ChatOpenAI(streaming=True)
    chain: Runnable = prompt | llm
    response = await chain.ainvoke(state, config=config)
    return {
        "messages": [response]
    }

@cl.on_chat_start
async def on_chat_start():
    # start graph
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_edge("chat", END)
    graph.set_entry_point("chat")

    # initialize state
    state = ChatState(messages=[])

    # save graph and state to the user session
    cl.user_session.set("graph", graph.compile())
    cl.user_session.set("state", state)

@cl.on_message
async def on_message(message: cl.Message):
    runnable: Runnable = cl.user_session.get("graph")
    state = cl.user_session.get("state")
    state["messages"] += [HumanMessage(content=message.content)]

    msg = cl.Message(content="")
    await msg.send()
    async for event in runnable.astream_events(
        state, version="v1",
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content or ""
                await msg.stream_token(token=content)
    await msg.update()