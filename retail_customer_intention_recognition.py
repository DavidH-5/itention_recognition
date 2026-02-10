# %%

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import sqlite3
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.sqlite import SqliteSaver

from intention_recognition_abstract_class import IntentionRecognizer

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# %%

# =========================
# 1. STATE (Multi-turn)
# =========================

class IntentState(MessagesState):
    domain: Optional[str]
    domain_reasoning: Optional[str]
    intents: Optional[List[Dict[str, Any]]]
    # # use confidence score to determine if it can proceed to next step (main graph)
    # confidence_scores: Optional[List[float]]
    # intents_reasoning: Optional[list[str]]


# =========================
# 2. SHARED OUTPUT SCHEMAS
# =========================

class DomainResult(BaseModel):
    domain: str
    reasoning: str


class IntentItem(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class IntentResult(BaseModel):
    intents: List[IntentItem]


# # =========================
# # 3. INTENTION RECOGNIZER CONTRACT
# # =========================

# class IntentionRecognizer(ABC):

#     @abstractmethod
#     def recognize(self, state: IntentState) -> Dict:
#         pass


# =========================
# 4. MODEL
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# =========================
# 5. DOMAIN RECOGNIZER (L1)
# =========================

class DomainIntentionRecognizer(IntentionRecognizer):

    DOMAINS = ["COMMERCE", "POST_PURCHASE", "GENERAL_POLICY"]

    def recognize(self, state: IntentState) -> Dict:
        system = SystemMessage(
            content=(
                "You are a domain classifier for Bunnings.\n"
                f"Valid domains: {self.DOMAINS}\n"
                "Classify the user's intent domain and explain briefly."
                # "The whole message history is provided for context; however focus on the latest user message to detect domain shift."
                "The whole message history is provided for context; however if the latest human message has intent shift, use the latest message only"
            )
        )

        latest_human = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]

        # if state.get("last_processed_message_id") != latest_human.id:
        #     state["domain"] = None  # reset domain
        #     state["intents"] = None  # reset intents
        #     state["last_processed_message_id"] = latest_human.id

        structured_llm = llm.with_structured_output(DomainResult)
        result = structured_llm.invoke([system] + [latest_human])

        print("Domain Node Run")

        return result.model_dump()


# =========================
# 6. L2 INTENT RECOGNIZERS
# =========================

class CommerceIntentionRecognizer(IntentionRecognizer):

    INTENTS = [
        "SEARCH", "DISCOVER", "COMPARE", "RECOMMEND", "HOW_TO",
        "CHECK_COMPATIBILITY", "CHECK_PRICE", "CHECK_AVAILABILITY",
        "FULFILMENT_OPTIONS", "ADD_TO_CART",
    ]

    def recognize(self, state: IntentState) -> Dict:
        system = SystemMessage(
            content=(
                f"Identify all relevant commerce intents.\n"
                f"Valid intents: {self.INTENTS}\n"
                "Rank by confidence and explain."
                # "The whole message history is provided for context, however, focus on the latest user message to detect intent shift."
                # "The whole message history is provided for context; however if the latest human message has intent shift, use the latest message only"
            )
        )

        latest_human = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
        structured_llm = llm.with_structured_output(IntentResult)
        result = structured_llm.invoke([system] + [latest_human])

        intents = [i.model_dump() for i in result.intents]
        filtered_intents = [
            intent for intent in intents
            if intent.get("confidence", 0) >= 0.8
        ]

        return {
            "intents": filtered_intents,
            "clarification_required": True if len(filtered_intents) == 0 else False,
            # "clarification_message": (
            #     f"I understand this is about {', '.join(i['intent'] for i in intents)}. "
            #     "Can you confirm this is what you are looking for?"
            # ),
        }


class PostPurchaseIntentionRecognizer(CommerceIntentionRecognizer):
    INTENTS = [
        "INSTALL_SUPPORT", "TROUBLESHOOT", "RETURNS",
        "WARRANTY", "PARTS_REPLACEMENT", "CARE_MAINTENANCE",
    ]


class GeneralPolicyIntentionRecognizer(CommerceIntentionRecognizer):
    INTENTS = [
        "STORE_INFO", "OPENING_HOURS",
        "POLICIES", "TRADE_ACCOUNT", "CONTACT_SUPPORT",
    ]


# =========================
# 7. GRAPH NODES
# =========================

domain_recognizer = DomainIntentionRecognizer()
commerce_recognizer = CommerceIntentionRecognizer()
post_sale_recognizer = PostPurchaseIntentionRecognizer()
general_recognizer = GeneralPolicyIntentionRecognizer()


def domain_node(state: IntentState) -> IntentState:
    # if state.get("domain"):
    #     return state

    result = domain_recognizer.recognize(state)
    state["domain"] = result["domain"]
    # state["domain_confidence"] = result["confidence"]

    # if result["confidence"] < 0.6:
    #     state["messages"].append(
    #         AIMessage(
    #             content=(
    #                 f"I believe this is related to **{result['domain']}**, "
    #                 "but I may be mistaken. Could you confirm?"
    #             )
    #         )
    #     )

    return state


def commerce_intent_node(state: IntentState) -> IntentState:
    result = commerce_recognizer.recognize(state)
    state["intents"] = result["intents"]

    if result["clarification_required"]:
        state["messages"].append(
            AIMessage(content=result["clarification_message"])
        )

    return state


def post_sale_intent_node(state: IntentState) -> IntentState:
    result = post_sale_recognizer.recognize(state)
    state["intents"] = result["intents"]

    # print(result["clarification_required"])

    if result["clarification_required"]:
        state["messages"].append(
            AIMessage(content=result["clarification_message"])
        )

    return state


def general_intent_node(state: IntentState) -> IntentState:
    result = general_recognizer.recognize(state)
    state["intents"] = result["intents"]

    if result["clarification_required"]:
        state["messages"].append(
            AIMessage(content=result["clarification_message"])
        )

    return state


def domain_router(state: IntentState) -> str:
    domain = state.get("domain")

    if domain == "COMMERCE":
        return "commerce_intent"
    elif domain == "POST_PURCHASE":
        return "post_sale_intent"
    elif domain == "GENERAL_POLICY":
        return "general_intent"
    else:
        # safety fallback
        return END
    

# def intent_node(state: IntentState) -> IntentState:
#     if state["domain"] == "COMMERCE":
#         result = commerce_recognizer.recognize(state)
#     elif state["domain"] == "POST_PURCHASE":
#         result = post_recognizer.recognize(state)
#     else:
#         result = general_recognizer.recognize(state)

#     state["intents"] = result["intents"]
#     state["clarification_required"] = result["clarification_required"]

#     if result["clarification_required"]:
#         state["messages"].append(
#             AIMessage(content=result["clarification_message"])
#         )

#     return state


# =========================
# 8. GRAPH + CHECKPOINT
# =========================

# Ensure the directory exists
os.makedirs("./db", exist_ok=True)

conn = sqlite3.connect("./db/intent_graph.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(IntentState)

graph.add_node("domain", domain_node)
graph.add_node("commerce_intent", commerce_intent_node)
graph.add_node("post_sale_intent", post_sale_intent_node)
graph.add_node("general_intent", general_intent_node)

graph.add_edge(START, "domain")
graph.add_conditional_edges(
    "domain",
    domain_router,
    {
        "commerce_intent": "commerce_intent",
        "post_sale_intent": "post_sale_intent",
        "general_intent": "general_intent",
        END: END,
    },
)

intent_graph = graph.compile(checkpointer=checkpointer)


# %%
# =========================
# 9. RUN EXAMPLE
# =========================

thread_id = "10"

initial_state = {
    "messages": [HumanMessage(content="I bought a drill and now it won’t start")],
    # "domain": None,
    # "intents": None,
    # "clarification_required": None,
}

result = intent_graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": thread_id}},
)

print("DOMAIN:", result["domain"])
print("INTENTS:", result["intents"])


# %%
# Test cases

# Simple test cases:
# I'm looking for a cordless drill that works well for concrete.
# I bought a drill last week and it suddenly stopped working.

# Tricky test cases:
# I want to return a drill, but if that’s hard, maybe I should just fix it.
# I need help with a drill I got from your store.

# thread_id = "5"

initial_state = {
    "messages": [HumanMessage(content="What are your store opening hours on weekends?")],
    # "domain": None,
    # "intents": None,
    # "clarification_required": None,
}

result = intent_graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": thread_id}},
)

print("DOMAIN:", result["domain"])
print("INTENTS:", result["intents"])


# %%