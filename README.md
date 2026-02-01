# 🧠 Intention Recognition (2-Level) with LangGraph + TrustCall

This project implements a **multi-turn intention recognition system** using a **2-level classifier pipeline**:

- **Level 1 (L1): Domain Classification**
- **Level 2 (L2): Intent Classification (within the detected domain)**

It is built with:

- **LangGraph** (`StateGraph`, `MessagesState`) for orchestration
- **TrustCall** (`create_extractor`) for structured extraction into Pydantic schemas
- **OpenAI Chat Model** (`ChatOpenAI`) for classification
- **SQLite checkpointing** (`SqliteSaver`) for persistent multi-turn state

---

## ✅ What It Does

Given a conversation (multi-turn messages), the graph will:

1. Look at the **latest user message**
2. Classify it into a **domain**:
   - `COMMERCE`
   - `POST_PURCHASE`
   - `GENERAL_POLICY`
3. Route to the correct intent recognizer based on the domain
4. Predict an **intent** + **confidence score (0–1)** + reasoning

---

## 🏗️ Architecture Overview

### 1) State Definition

The graph uses a shared multi-turn state (`IntentState`) extending `MessagesState`:

- `domain`, `domain_reasoning`
- `intent`, `intent_reasoning`
- `intent_confidence`

This makes it easy to store classification results across turns.

---

### 2) Output Schemas (Structured Extraction)

Two Pydantic schemas define the structured outputs:

- **DomainResult**
  - `domain: str`
  - `reasoning: str`

- **IntentResult**
  - `intent: str`
  - `confidence: float (0–1)`
  - `reasoning: str`

TrustCall extracts model output into these schemas reliably.

---

### 3) Domain Classifier (L1)

`DomainIntentionRecognizer` determines the domain using strict rules:

- Decide based on the **LATEST user message only**
- Use earlier messages only for **reference resolution**
  - e.g. “it”, “that”, “the item”

---

### 4) Intent Classifiers (L2)

Once the domain is selected, the graph routes to one of:

#### `CommerceIntentionRecognizer`
Supported intents:

- `SEARCH`, `DISCOVER`, `COMPARE`, `RECOMMEND`, `HOW_TO`
- `CHECK_COMPATIBILITY`, `CHECK_PRICE`, `CHECK_AVAILABILITY`
- `FULFILMENT_OPTIONS`, `ADD_TO_CART`

#### `PostPurchaseIntentionRecognizer`
Supported intents:

- `INSTALL_SUPPORT`, `TROUBLESHOOT`, `RETURNS`
- `WARRANTY`, `PARTS_REPLACEMENT`, `CARE_MAINTENANCE`

#### `GeneralPolicyIntentionRecognizer`
Supported intents:

- `STORE_INFO`, `OPENING_HOURS`
- `POLICIES`, `TRADE_ACCOUNT`, `CONTACT_SUPPORT`

Each recognizer returns:

- `intent`
- `confidence`
- `reasoning`

---

## 🔁 LangGraph Flow

```text
START
  ↓
domain (L1 classifier)
  ↓
(domain_router)
  ├── COMMERCE       → commerce_intent
  ├── POST_PURCHASE  → post_sale_intent
  └── GENERAL_POLICY → general_intent
  ↓
END
