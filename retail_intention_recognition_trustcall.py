# %%

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import sqlite3
from pydantic import BaseModel, Field
from trustcall import create_extractor
import uuid
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
# STATE (Multi-turn)
# =========================

class IntentState(MessagesState):
    domain: Optional[str]
    domain_reasoning: Optional[str]
    intent: Optional[str]
    intent_reasoning: Optional[str]
    intent_confidence: Optional[float]
    # # use confidence score to determine if it can proceed to next step (main graph)
    # confidence_scores: Optional[List[float]]
    # intents_reasoning: Optional[list[str]]


# =========================
# SHARED OUTPUT SCHEMAS
# =========================

class DomainResult(BaseModel):
    domain: str
    reasoning: str


class IntentResult(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# class IntentResult(BaseModel):
#     intents: List[IntentItem]


# %%


# =========================
# MODEL
# =========================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


# =========================
# DOMAIN RECOGNIZER (L1)
# =========================

class DomainIntentionRecognizer(IntentionRecognizer):

    DOMAINS = ["COMMERCE", "POST_PURCHASE", "GENERAL_POLICY"]

    def recognize(self, state: IntentState) -> Dict:
        
        domain_extractor = create_extractor(
            llm = llm,
            tools = [DomainResult],
            tool_choice = 'DomainResult',
            enable_inserts = True
        )

        # Instruction
        instruction = (
            f"""
                You are an domain classifier.

                Rules:
                - Decide the domain ONLY from the LATEST user message.
                - Use previous messages ONLY to resolve references like "it", "that", or "the item".
                - DO NOT let earlier intents influence your decision.
                - If the latest message is ambiguous without context, use history to clarify.
            """
            f"""Extract customer intention domain from the following conversation."""
            f"""The possible domains are: {self.DOMAINS}."""
        )

        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        latest = human_msgs[-1]
        history = human_msgs[:-1] 

        messages = [
            SystemMessage(content=instruction),
        ]

        if history:
            messages.append(
                SystemMessage(content="Conversation context (reference resolution only):")
            )
            messages.extend(history)

        messages.extend([
            SystemMessage(content="Latest user message (intent must be based on this):"),
            latest,
        ])

        extracted_domain = domain_extractor.invoke({
            "messages": messages
        })

        return extracted_domain["responses"][0]


# =========================
# INTENTION RECOGNIZER (L2)
# =========================

class CommerceIntentionRecognizer(IntentionRecognizer):

    INTENTS = [
        "SEARCH", "DISCOVER", "COMPARE", "RECOMMEND", "HOW_TO",
        "CHECK_COMPATIBILITY", "CHECK_PRICE", "CHECK_AVAILABILITY",
        "FULFILMENT_OPTIONS", "ADD_TO_CART",
    ]

    def recognize(self, state: IntentState) -> Dict:
        
        intention_extractor = create_extractor(
            llm = llm,
            tools = [IntentResult],
            tool_choice = 'IntentResult',
            enable_inserts = True
        )

        # Instruction
        instruction = (
            f"""
                You are an domain classifier.

                Rules:
                - Decide the domain ONLY from the LATEST user message.
                - Use previous messages ONLY to resolve references like "it", "that", or "the item".
                - DO NOT let earlier intents influence your decision.
                - If the latest message is ambiguous without context, use history to clarify.
            """
            f"""Extract customer intention domain from the following conversation."""
            f"""The possible domains are: {self.INTENTS}."""
        )

        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        latest = human_msgs[-1]
        history = human_msgs[:-1] 

        messages = [
            SystemMessage(content=instruction),
        ]

        if history:
            messages.append(
                SystemMessage(content="Conversation context (reference resolution only):")
            )
            messages.extend(history)

        messages.extend([
            SystemMessage(content="Latest user message (intent must be based on this):"),
            latest,
        ])

        extracted_intent = intention_extractor.invoke({
            "messages": messages
        })

        return extracted_intent["responses"][0]


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


domain_recognizer = DomainIntentionRecognizer()
commerce_recognizer = CommerceIntentionRecognizer()
post_sale_recognizer = PostPurchaseIntentionRecognizer()
general_recognizer = GeneralPolicyIntentionRecognizer()

def domain_node(State: IntentState) -> IntentState:
    
    result = domain_recognizer.recognize(State)
    State["domain"] = result.domain
    State["domain_reasoning"] = result.reasoning
    
    return State

def commerce_node(State: IntentState) -> IntentState:
    
    result = commerce_recognizer.recognize(State)
    State["intent"] = result.intent
    State["intent_reasoning"] = result.reasoning
    State['intent_confidence'] = result.confidence
    
    return State

def post_sale_node(State: IntentState) -> IntentState:
    
    result = post_sale_recognizer.recognize(State)
    State["intent"] = result.intent
    State["intent_reasoning"] = result.reasoning
    State['intent_confidence'] = result.confidence
    
    return State

def general_node(State: IntentState) -> IntentState:
    
    result = general_recognizer.recognize(State)
    State["intent"] = result.intent
    State["intent_reasoning"] = result.reasoning
    State['intent_confidence'] = result.confidence
    
    return State

def domain_router(state: IntentState) -> str:
    
    domain = state.get("domain")

    if domain == "COMMERCE":
        return "commerce_intent"
    elif domain == "POST_PURCHASE":
        return "post_sale_intent"
    elif domain == "GENERAL_POLICY":
        return "general_intent"
    else:
        return END


# =========================
# GRAPH + CHECKPOINT
# =========================

# Ensure the directory exists
os.makedirs("./db", exist_ok=True)

conn = sqlite3.connect("./db/intent_graph.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(IntentState)

graph.add_node("domain", domain_node)
graph.add_node("commerce_intent", commerce_node)
graph.add_node("post_sale_intent", post_sale_node)
graph.add_node("general_intent", general_node)

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


thread_id = uuid.uuid4()


# %%


initial_state = {
    "messages": [HumanMessage(content="I bought a drill and now it won’t start")]
}

result = intent_graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": thread_id}},
)

print("Domain:", result["domain"])
print("Domain Reason:", result["domain_reasoning"])
print("Intent:", result["intent"])
print("Intent Reasoning:", result["intent_reasoning"])
print("Intent Reasoning:", result["intent_confidence"])



# %%
# Test cases

# Simple test cases:
# I'm looking for a cordless drill that works well for concrete.
# I bought a drill last week and it suddenly stopped working.

# Tricky test cases:
# I want to return a drill, but if that’s hard, maybe I should just fix it.
# I need help with a drill I got from your store.


initial_state = {
    "messages": [HumanMessage(content="What are your store opening hours on weekends?")],
}

# initial_state = {
#     "messages": [HumanMessage(content="I'm looking for a cordless drill that works well for concrete?")],
# }

result = intent_graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": thread_id}},
)

print("Domain:", result["domain"])
print("Domain Reason:", result["domain_reasoning"])
print("Intent:", result["intent"])
print("Intent Reasoning:", result["intent_reasoning"])
print("Intent Reasoning:", result["intent_confidence"])


# %%