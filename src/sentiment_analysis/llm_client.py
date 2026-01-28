from dataclasses import dataclass
from enum import Enum
from logging import Logger
from os import environ
from time import sleep
from typing import cast, Final, Optional

from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletionUserMessageParam

from src.core.logging_manager import LoggingManager


class DailyRateLimitExceededError(Exception):
    """
    Raised when the daily (RPD) rate limit for the LLM API has been exceeded.

    This is considered an unrecoverable error for the current execution session,
    as waiting a minute will not resolve it.
    """
    pass


@dataclass(frozen=True)
class _ProviderConfig:
    """
    Stores configuration for a specific LLM provider.

    :ivar base_url: The fixed base URL for the provider's API, if any.
    :ivar api_key_env_var: The name of the environment variable for the API key.
    :ivar api_path_suffix: The API path suffix for local backends.
    :ivar keywords: A tuple of keywords to identify this provider from a model ID string.
    :ivar is_local: A boolean indicating if the provider is locally hosted.
    """
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None
    api_path_suffix: Optional[str] = None
    keywords: tuple[str, ...] = ()
    is_local: bool = False


class LLMProvider(Enum):
    """
    Enum for different LLM providers, holding their specific configuration.
    """
    GPT = _ProviderConfig(
        api_key_env_var='GPT_API_KEY',
        keywords=('gpt',)
    )
    GEMINI = _ProviderConfig(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env_var='GEMINI_API_KEY',
        keywords=('gemini',)
    )
    DOCKER_DMR = _ProviderConfig(
        api_path_suffix="engines/v1",
        keywords=('dmr',),
        is_local=True,
    )
    OLLAMA = _ProviderConfig(
        api_path_suffix="v1",
        keywords=('ollama',),
        is_local=True,
    )
    LM_STUDIO = _ProviderConfig(
        api_path_suffix="v1",
        keywords=('lm-studio',),
        is_local=True,
    )

    @classmethod
    def from_string(cls, model_id: str) -> tuple['LLMProvider', str]:
        """
        Determines the provider and actual model name from a model identifier string.

        The format can be '<provider>/<model_name>' (e.g., 'ollama/gemma2') or
        a model name that implies the provider (e.g., 'gpt-4o').

        :param model_id: The user-provided model identifier string.
        :returns: A tuple containing the determined LLMProvider and the actual model name for the API.
        :raises ValueError: If the provider cannot be determined.
        """
        model_id_lower: str = model_id.lower()

        # Strategy 1: Check for explicit provider prefix like 'ollama/gemma2'
        if '/' in model_id:
            provider_str, actual_model_name = model_id.split('/', 1)
            provider_str_lower: str = provider_str.lower()
            for provider in cls:
                if provider_str_lower in provider.value.keywords:
                    return provider, actual_model_name

        # Strategy 2: Fallback for implicit providers like 'gpt-4o'
        for provider in cls:
            if any(model_id_lower.startswith(keyword) for keyword in provider.value.keywords):
                return provider, model_id  # The full string is the model name

        raise ValueError(f"Could not determine a valid provider from model ID: '{model_id}'")

    @classmethod
    def is_local_from_string(cls, model_id: str) -> bool:
        """
        Checks if a model ID string implies a local provider.

        :param model_id: The user-provided model identifier string.
        :returns: True if the determined provider is local, False otherwise.
        """
        try:
            provider, _ = cls.from_string(model_id)
            return provider.value.is_local
        except ValueError:
            return False


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

    def __init__(
        self,
        target_model_id: str,
        local_host: Optional[str] = None,
        local_port: Optional[int] = None
    ) -> None:
        """
        Initializes the LLMClient.

        :param target_model_id: The user-provided model identifier string (e.g., 'gpt-4o-mini', 'ollama/gemma:latest').
        :param local_host: The hostname or IP address for a local provider.
        :param local_port: The port number for a local provider.
        :raises ValueError: If configuration is invalid (e.g., missing API key or local URL).
        """
        self._logger = LoggingManager().get_logger('root')

        # Deconstruct the provider and the actual model name from the target ID
        self._provider, self._target_model_id = LLMProvider.from_string(model_id=target_model_id)

        api_key: Optional[str] = None

        if self.config.is_local:
            if not all([local_host, local_port]):
                raise ValueError(
                    f"Arguments 'local_host' and 'local_port' are required for the local provider '{self._provider.name}'."
                )
            if not self.config.api_path_suffix:
                # This is a safeguard, should not happen with current enums
                raise ValueError(
                    f"Provider '{self._provider.name}' is local but has no api_path_suffix self.configured.")
            # noinspection HttpUrlsUsage
            base_url = f"http://{local_host}:{local_port}/{self.config.api_path_suffix}"
            api_key = 'ignored'
        else:  # Remote provider
            if self.config.api_key_env_var:
                api_key = self.__get_api_key(env_var_name=self.config.api_key_env_var)
            base_url = self.config.base_url  # Can be None, client will use default

        self._client = self.__create_client(base_url=base_url, api_key=api_key)
        self._logger.debug(
            f"LLMClient initialized for model '{self._target_model_id}' (Provider: {self._provider.name})")

    @property
    def config(self) -> _ProviderConfig:
        """
        Returns the configuration object for the current provider.

        :return: The _ProviderConfig dataclass instance associated with the client's provider.
        """
        return self._provider.value

    def __create_client(self, base_url: Optional[str] = None, api_key: Optional[str] = None) -> OpenAI:
        """
        Creates and configures the OpenAI client based on the provider.

        :param base_url: The base URL for the API. If None, the default OpenAI URL is used.
        :param api_key: The API key for an online LLM.
        :returns: An initialized OpenAI client instance.
        """
        # Local models might have long-running inference, so a longer timeout is beneficial.
        timeout: float = 1200.0 if self._provider.value.is_local else 600.0
        return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    @staticmethod
    def __get_api_key(env_var_name: str) -> str:
        """
        Retrieves an API key from an environment variable.

        :param env_var_name: The name of the environment variable.
        :returns: The API key string.
        :raises ValueError: If the required environment variable is not found.
        """
        env_key: Optional[str] = environ.get(env_var_name)
        if not env_key:
            raise ValueError(f"Required environment variable '{env_var_name}' not found.")
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
            role: str = "system" if self._provider.value.is_local else "assistant"
            messages.append({"role": role, "content": rule_message})

        for single_prompt in prompt_texts:
            messages.append({"role": "user", "content": single_prompt})
        return messages

    def generate_response(
        self, prompt_texts: list[str] | str, rule_message: Optional[str] = None, temperature: float = 0.1
    ) -> str:
        """
        Generates a response from the LLM with robust retry and error handling.

        This method attempts to get a valid response. It intelligently handles
        rate limits by parsing the error response:
        - If a 'PerDay' quota is detected, it raises a `DailyRateLimitExceededError`.
        - If a 'PerMinute' quota (or a generic 429) is detected, it waits and retries once.
        - For other recoverable errors, it retries up to a configured maximum.

        :param prompt_texts: A single prompt string or a list of strings for a multi-turn conversation.
        :param rule_message: An optional instruction message for the model.
        :param temperature: Sampling temperature. Higher values (e.g., 0.8) make output more random,
                            lower values (e.g., 0.1) make it more deterministic.
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
                    messages=cast(list[ChatCompletionUserMessageParam], cast(object, prompt_messages)),
                    temperature=temperature
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
                                # noinspection SpellCheckingInspection
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
