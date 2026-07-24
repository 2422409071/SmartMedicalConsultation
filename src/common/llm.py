"""
LLM Client Module
Provides a singleton ChatOpenAI instance configured from project settings.
"""

import sys
from pathlib import Path
from functools import lru_cache

# Ensure project root is in sys.path for direct execution
_project_root = Path(__file__).parent.parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from langchain_openai import ChatOpenAI

from config.settings import settings
from src.common.logger import setup_logger

logger = setup_logger(__name__, "llm.log")


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """
    Get a cached ChatOpenAI instance (singleton pattern).

    Uses lru_cache to ensure only one LLM connection is created,
    avoiding redundant API connections and reducing latency.

    Args:
        temperature: Controls randomness of output (0.0 - 1.0).
                     Lower = more deterministic, Higher = more creative.
                     Default: 0.7 (balanced for medical consultation).

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        ValueError: If MODEL_API_KEY is not configured in .env.

    Usage:
        >>> llm = get_llm()
        >>> response = llm.invoke("What are the symptoms of hypertension?")

        >>> # For deterministic tasks (entity extraction)
        >>> llm_strict = get_llm(temperature=0.1)
    """
    # Validate API key
    if not settings.model_api_key:
        logger.error("MODEL_API_KEY is not configured in .env")
        raise ValueError(
            "MODEL_API_KEY is empty. "
            "Please set it in the .env file. "
            "Get your key from: https://dashscope.console.aliyun.com/apiKey"
        )

    logger.info(
        f"Creating LLM client: model={settings.model_name}, "
        f"base_url={settings.model_base_url}, temperature={temperature}"
    )

    llm = ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.model_api_key,
        openai_api_base=settings.model_base_url,
        temperature=temperature,
        max_retries=2,
        request_timeout=60,
    )

    logger.info("LLM client created successfully")
    return llm


if __name__ == "__main__":
    # Demo: test LLM connection
    print("=" * 60)
    print("LLM Client Test")
    print("=" * 60)

    try:
        llm = get_llm()
        print(f"[PASS] Model: {settings.model_name}")
        print(f"[PASS] Base URL: {settings.model_base_url}")

        # Quick connectivity test
        response = llm.invoke("Say 'hello' in one word")
        print(f"[PASS] Response: {response.content}")

        # Test singleton (should return same instance)
        llm2 = get_llm()
        assert llm is llm2, "[FAIL] Singleton pattern broken"
        print("[PASS] Singleton pattern verified")

        print("\n[SUCCESS] LLM client is working correctly!")

    except ValueError as e:
        print(f"[FAIL] Configuration error: {e}")
    except Exception as e:
        print(f"[FAIL] Connection error: {e}")
        print("  Please check your MODEL_API_KEY and network connection.")
