from enum import auto, Enum
from logging import Logger
from os import environ
from time import sleep
from typing import Final, Optional

from openai import OpenAI, RateLimitError

from src.core.logging_manager import LoggingManager


class DailyRateLimitExceededError(Exception):
    """
    Raised when the daily (RPD) rate limit for the LLM API has been exceeded.

    This is considered an unrecoverable error for the current execution session,
    as waiting a minute will not resolve it.
    """
    pass


class LLMProvider(Enum):
    """
    Enumeration for different LLM providers.

    This provides a type-safe way to identify and manage different LLM services.
    """
    GEMINI = auto()
    GPT = auto()
    GEMMA = auto()

    @classmethod
    def from_string(cls, model_id: str) -> 'LLMProvider':
        """
        Determines the provider from a model identifier string.

        :param model_id: The string identifier for the model (e.g., 'gpt-4o-mini', 'gemma2:latest').
        :returns: The corresponding LLMProvider enum member.
        :raises ValueError: If the model_id does not match any known provider.
        """
        model_id_lower: str = model_id.lower()
        if 'gemini' in model_id_lower:
            return cls.GEMINI
        if 'gpt' in model_id_lower:
            return cls.GPT
        if 'gemma' in model_id_lower:
            return cls.GEMMA
        raise ValueError(f"Unsupported or unrecognized model provider for id: '{model_id}'")


class LLMClient:
    """
    A robust client for interacting with various Large Language Models (LLMs).

    This class provides a unified interface to communicate with different LLM APIs,
    handling client initialization, API key management, prompt formatting, and
    sophisticated error handling including rate-limiting and response validation.

    :ivar _target_model_id: The specific model identifier string (e.g., 'gpt-4o-mini').
    :ivar _provider: The LLMProvider determined from the model ID.
    :ivar _client: The initialized OpenAI-compatible client instance.
    :ivar _logger: The logger instance for this class.
    """
    _target_model_id: Final[str]
    _provider: Final[LLMProvider]
    _client: Final[OpenAI]
    _logger: Final[Logger]

    # Constants for retry logic
    _MAX_RESPONSE_RETRIES: Final[int] = 3
    _RATE_LIMIT_WAIT_SECONDS: Final[int] = 61

    def __init__(self, target_model_id: str, local_llm_url: Optional[str] = None) -> None:
        """
        Initializes the LLMClient.

        :param target_model_id: The specific model identifier (e.g., 'gpt-4o-mini', 'gemma2:latest').
        :param local_llm_url: The base URL for a locally hosted LLM. Required if the provider is local (Gemma).
        :raises ValueError: If configuration is invalid (e.g., missing API key or local URL).
        """
        self._target_model_id = target_model_id
        self._logger = LoggingManager().get_logger('root')
        self._provider = LLMProvider.from_string(model_id=target_model_id)

        api_key: Optional[str] = None
        if self._provider in (LLMProvider.GPT, LLMProvider.GEMINI):
            api_key = self.__get_api_key()
        elif self._provider is LLMProvider.GEMMA and not local_llm_url:
            raise ValueError("A 'local_llm_url' is required when using a local LLM provider (Gemma).")

        self._client = self.__create_client(llm_url=local_llm_url, api_key=api_key)
        self._logger.debug(
            f"LLMClient initialized for model '{self._target_model_id}' (Provider: {self._provider.name})")

    def __create_client(self, llm_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
        """
        Creates and configures the OpenAI client based on the provider.

        :param llm_url: The URL for a local LLM.
        :param api_key: The API key for an online LLM.
        :returns: An initialized OpenAI client instance.
        """
        match self._provider:
            case LLMProvider.GPT:
                return OpenAI(api_key=api_key)
            case LLMProvider.GEMINI:
                base_url: Final[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
                return OpenAI(base_url=base_url, api_key=api_key)
            case LLMProvider.GEMMA:
                return OpenAI(base_url=llm_url, api_key='ignored', timeout=1200.0)

    def __get_api_key(self) -> str:
        """
        Retrieves the API key from environment variables based on the provider.

        :returns: The API key string.
        :raises ValueError: If the required environment variable is not found.
        """
        var_name_map: dict[LLMProvider, str] = {
            LLMProvider.GPT: 'GPT_API_KEY',
            LLMProvider.GEMINI: 'GEMINI_API_KEY',
        }
        var_name: Optional[str] = var_name_map.get(self._provider)
        if not var_name:
            # This should not be reached due to the logic in __init__
            raise ValueError(f"API key configuration not defined for provider: {self._provider.name}")

        env_key: Optional[str] = environ.get(var_name)
        if not env_key:
            raise ValueError(f"Required environment variable '{var_name}' not found.")
        return env_key

    def __generate_prompt_message(
        self, prompt_texts: list[str], rule_message: Optional[str] = None
    ) -> list[dict[str, str]]:
        """
        Constructs the list of messages for the API chat completion request.

        :param prompt_texts: A list of user prompts.
        :param rule_message: An optional system or assistant rule message.
        :returns: A list of message dictionaries.
        """
        messages: list[dict[str, str]] = []
        if rule_message:
            # Local models often prefer 'system' role for instructions.
            role: str = "system" if self._provider is LLMProvider.GEMMA else "assistant"
            messages.append({"role": role, "content": rule_message})

        for single_prompt in prompt_texts:
            messages.append({"role": "user", "content": single_prompt})
        return messages

    def generate_response(self, prompt_texts: list[str] | str, rule_message: Optional[str] = None) -> str:
        """
        Generates a response from the LLM with robust retry and error handling.

        This method attempts to get a valid response. It intelligently handles
        rate limits by parsing the error response:
        - If a 'PerDay' quota is detected, it raises a `DailyRateLimitExceededError`.
        - If a 'PerMinute' quota (or a generic 429) is detected, it waits and retries once.
        - For other recoverable errors, it retries up to a configured maximum.

        :param prompt_texts: A single prompt string or a list of strings for a multi-turn conversation.
        :param rule_message: An optional instruction message for the model.
        :returns: The content of the model's response as a string.
        :raises DailyRateLimitExceededError: If the daily rate limit (RPD) is hit.
        :raises RuntimeError: If a valid response cannot be obtained after all retries.
        """
        if isinstance(prompt_texts, str):
            prompt_contents: list[str] = [prompt_texts]
        elif not isinstance(prompt_texts, list):
            raise TypeError("Argument 'prompt_texts' must be a string or a list of strings.")
        else:
            prompt_contents: list[str] = prompt_texts

        prompt_messages: list[dict[str, str]] = self.__generate_prompt_message(
            prompt_texts=prompt_contents,
            rule_message=rule_message
        )

        last_exception: Optional[Exception] = None
        rate_limit_retry_done: bool = False

        for attempt in range(self._MAX_RESPONSE_RETRIES):
            try:
                self._logger.debug(
                    f"Attempt {attempt + 1}/{self._MAX_RESPONSE_RETRIES} to generate response from '{self._target_model_id}'.")
                response = self._client.chat.completions.create(
                    model=self._target_model_id,
                    messages=prompt_messages
                )
                try:
                    content: Optional[str] = response.choices[0].message.content
                    if not content:
                        raise ValueError("API response content is empty.")

                except (AttributeError, IndexError, TypeError, ValueError) as parse_error:
                    last_exception = parse_error
                    self._logger.warning(
                        f"Failed to parse a valid response content (Attempt {attempt + 1}). "
                        f"Error: {parse_error}. Raw response: {response}"
                    )
                    sleep(2)
                    continue
                self._logger.debug(f"Successfully received response from '{self._target_model_id}'.")
                return content.strip()

            except RateLimitError as e:
                last_exception = e

                is_daily_limit: bool = False
                try:
                    if e.body and isinstance(e.body, list) and e.body:
                        error_details = e.body[0].get('error', {}).get('details', [])
                        if isinstance(error_details, list) and error_details:
                            violations = error_details[0].get('violations', [])
                            if isinstance(violations, list) and violations:
                                quota_id = violations[0].get('quotaId', '')
                                if 'perday' in quota_id.lower():
                                    is_daily_limit = True
                except (AttributeError, IndexError, KeyError, TypeError):
                    # If parsing fails, we log it but proceed with the RPM assumption
                    self._logger.warning("Could not definitively parse RateLimitError body. Assuming RPM.")

                if is_daily_limit:
                    self._logger.critical(f"Daily rate limit (RPD) detected. Aborting all operations. Error: {e}")
                    raise DailyRateLimitExceededError(
                        "The daily API quota has been reached. Further requests will fail.") from e

                # If it's not a daily limit, treat it as a per-minute limit (RPM)
                if rate_limit_retry_done:
                    self._logger.error("Rate limit error occurred again after waiting. Aborting this request.")
                    raise RuntimeError("Failed to resolve rate limit after waiting.") from e

                self._logger.warning(
                    f"Rate limit reached (assumed RPM). Waiting for {self._RATE_LIMIT_WAIT_SECONDS} seconds before one-time retry.")
                sleep(self._RATE_LIMIT_WAIT_SECONDS)
                rate_limit_retry_done = True
                continue  # Continue to the next iteration for the single retry

            except Exception as e:
                last_exception = e
                self._logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                sleep(5)  # Wait a bit longer for unexpected errors

        # If the loop completes without returning, it means all retries failed.
        final_error_message: str = (
            f"Failed to get a valid response from '{self._target_model_id}' after "
            f"{self._MAX_RESPONSE_RETRIES} attempts."
        )
        self._logger.critical(final_error_message)
        raise RuntimeError(final_error_message) from last_exception
