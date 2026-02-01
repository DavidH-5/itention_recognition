
# %%

from abc import ABC, abstractmethod
from typing import Dict, Any

# %%

class IntentionRecognizer(ABC):
    """
    Contract for intention recognition.
    Concrete implementations can handle:
    - domain detection which is the level 1 intent
    - level 2 intent recognition
    - confidence estimation
    - clarification decision
    """

    @abstractmethod
    def recognize(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns:
        {
            "domain": str,
            "intent": str,
            "confidence": float,
            "clarification_required": bool,
            "clarification_question": str | None
        }
        """
        pass
