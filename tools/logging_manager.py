import sys
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from io import TextIOBase
from dataclasses import dataclass, field
from colorama.ansitowin32 import StreamWrapper
from typing import Optional, Literal, TypeAlias, overload
from logging import Logger, Handler, Formatter, FileHandler, StreamHandler


class LogLevel(Enum):
    """
    An enumeration representing standard logging levels.

    Member Variables:
        NOTSET (int): Corresponds to logging.NOTSET (0).
        DEBUG (int): Corresponds to logging.DEBUG (10).
        INFO (int): Corresponds to logging.INFO (20).
        WARNING (int): Corresponds to logging.WARNING (30).
        ERROR (int): Corresponds to logging.ERROR (40).
        CRITICAL (int): Corresponds to logging.CRITICAL (50).
    """
    NOTSET: int = logging.NOTSET
    DEBUG: int = logging.DEBUG
    INFO: int = logging.INFO
    WARNING: int = logging.WARNING
    ERROR: int = logging.ERROR
    CRITICAL: int = logging.CRITICAL


@dataclass
class LoggerSettings:
    """
    Represents the settings for a single Logger instance.

    Member Variables:
        name (str): The name of the logger.
        level (LogLevel): The logging level for this logger.
        linked_handlers (list[str]): A list of names of handlers to link to this logger.
    """
    name: str
    level: LogLevel
    linked_handlers: list[str] = field(default_factory=list)


@dataclass
class HandlerSettings:
    """
    Represents the settings for a single Handler instance.

    Member Variables:
        name (str): The name of the handler.
        level (LogLevel): The logging level for this handler.
        output (Path | TextIOBase): The output destination for the handler.
                                  Can be a file path (Path) or a text I/O stream (e.g., sys.stdout, sys.stderr).
    """
    name: str
    level: LogLevel
    output: Path | TextIOBase | StreamWrapper


LogComponentSettings: TypeAlias = LoggerSettings | HandlerSettings


class LoggingManager:
    """
    Manages all loggers and handlers in a centralized manner using a singleton pattern.

    Member Variables:
        _instance (Optional[LoggingManager]): The singleton instance of LoggingManager.
        _DEFAULT_INITIAL_COMPONENTS (list[LogComponentSettings]): A class-level list of default
                                                                 logger and handler settings used
                                                                 during manager initialization if
                                                                 no custom configurations are provided.
        _loggers (list[Logger]): A list of all managed Logger objects. The root logger
                                 is guaranteed to be the first element if managed.
        _handlers (dict[str, Handler]): A dictionary of all managed Handler objects,
                                        where keys are handler names and values are Handler instances.
                                        The 'stdout' handler is guaranteed to exist with a StreamHandler.
        _logger_handler_connections (dict[str, list[str]]): A dictionary mapping logger names to
                                                           a list of names of handlers connected to them.
        _default_formatter (Formatter): The default Formatter instance used for all handlers
                                        added via this manager, ensuring a consistent log format.
        _initialized (bool): A flag to prevent re-initialization of the singleton instance.
    """

    _instance: Optional['LoggingManager'] = None
    _DEFAULT_INITIAL_COMPONENTS: list[LogComponentSettings] = [
        HandlerSettings(name='stdout', level=LogLevel.DEBUG, output=sys.stdout),
        LoggerSettings(name='root', level=LogLevel.DEBUG, linked_handlers=['stdout'])
    ]

    def __new__(cls, *args: any, **kwargs: any) -> 'LoggingManager':
        """
        Ensures that only one instance of LoggingManager is created (Singleton pattern).

        Returns:
            LoggingManager: The singleton instance of the LoggingManager.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, initial_configs: Optional[list[LogComponentSettings]] = None) -> None:
        """
        Initializes the LoggingManager instance.
        Allows for optional initial configuration of loggers and handlers.

        Input Variables:
            initial_configs (Optional[list[LogComponentSettings]]): A list of settings to apply during initialization.
                                                                     If None, default settings will be used.
        """
        if not hasattr(self, '_initialized'):
            self._loggers: list[logging.Logger] = []
            self._handlers: dict[str, logging.Handler] = {}
            self._logger_handler_connections: dict[str, list[str]] = {}
            self._default_formatter: Formatter = Formatter(
                fmt="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")

            print("Starting initial configuration...")
            if initial_configs is None:
                print("No initial configurations provided. Using default setup.")

                self._configure_initial_components(self._DEFAULT_INITIAL_COMPONENTS)
            else:
                print("Initial configurations provided. Applying custom setup.")

                self._configure_initial_components(initial_configs)

            self._initialized: bool = True
            print("LoggingManager initialized.")

    @property
    def loggers(self) -> list[Logger]:
        """
        Provides a read-only copy of the managed loggers list.

        Returns:
            list[Logger]: A shallow copy of the internal list of Logger objects.
        """
        return list(self._loggers)

    @property
    def handlers(self) -> dict[str, Handler]:
        """
        Provides a read-only copy of the managed handlers dictionary.

        Returns:
            dict[str, Handler]: A shallow copy of the internal dictionary of Handler objects.
        """
        return self._handlers.copy()

    @property
    def logger_handler_connections(self) -> dict[str, list[str]]:
        """
        Provides a read-only copy of the logger-handler connections mapping.

        Returns:
            dict[str, list[str]]: A shallow copy of the internal dictionary mapping
                                  logger names to lists of connected handler names.
                                  The lists within the dictionary are also shallow copies.
        """
        connections_copy: dict[str, list[str]] = {
            logger_name: list(handler_names)
            for logger_name, handler_names in self._logger_handler_connections.items()
        }
        return connections_copy

    def get_default_formatter(self) -> logging.Formatter:
        """
        Retrieves the default Formatter used by the LoggingManager.

        Returns:
            logging.Formatter: The default Formatter instance.
        """
        return self._default_formatter

    def get_handler(self, name: str) -> Optional[Handler]:
        """
        Retrieves a handler by its name.

        Input Variables:
            name (str): The name of the handler to retrieve.

        Returns:
            Optional[Handler]: The Handler object if found, otherwise None.
        """
        return self._handlers.get(name)

    def add_handler(self, handler_settings: HandlerSettings) -> Handler:
        """
        Adds a new handler to the manager based on provided settings.

        Input Variables:
            handler_settings (HandlerSettings): Settings for the handler to be added.

        Returns:
            Handler: The newly created and managed Handler instance.

        Raises:
            ValueError: If a handler with the same name already exists in the manager,
                        or if a handler with the same output target already exists.
            TypeError: If an unsupported handler output type is specified in settings.
        """
        if self.get_handler(handler_settings.name):
            raise ValueError(f"Handler with name '{handler_settings.name}' already exists in the manager.")

        target_type: type[Handler]
        target_value: str | TextIOBase | StreamWrapper
        new_handler: Handler
        if isinstance(handler_settings.output, Path):
            target_value = str(handler_settings.output.resolve())
            handler_settings.output.parent.mkdir(parents=True, exist_ok=True)
            existing_file_handler: dict[str:FileHandler] = \
                {name: handler for name, handler in self._handlers.items() if isinstance(handler, FileHandler)}
            existing_handler_name: Optional[str] = next(
                (name for name, handler in existing_file_handler.items() if handler.baseFilename == target_value), None)
            if existing_handler_name:
                raise ValueError(
                    f"Conflict detected: A FileHandler named '{existing_handler_name}' "
                    f"already writes to the same file: '{target_value}'. "
                    f"Cannot add new handler '{handler_settings.name}' targeting '{target_value}'."
                )
            new_handler = FileHandler(filename=handler_settings.output, delay=True)
            print(f"Constructing FileHandler with delay=True.")

        elif isinstance(handler_settings.output, (TextIOBase, StreamWrapper)):
            target_value = handler_settings.output
            existing_stream_handler: dict[str:StreamHandler|StreamWrapper] = \
                {name: handler for name, handler in self._handlers.items() if isinstance(handler, StreamHandler)}
            existing_handler_name: Optional[str] = next(
                (name for name, handler in existing_stream_handler.items() if handler.stream == target_value), None)
            if existing_handler_name:
                raise ValueError(
                    f"Conflict detected: A StreamHandler named '{existing_handler_name}' "
                    f"already writes to the same stream: {target_value!r}. "
                    f"Cannot add new handler '{handler_settings.name}' targeting {target_value!r}."
                )
            new_handler = StreamHandler(stream=handler_settings.output)
        else:
            raise TypeError("Unsupported handler output type. Must be Path or TextIOBase.")

        new_handler.setLevel(handler_settings.level.value)
        new_handler.setFormatter(self.get_default_formatter())
        self._handlers[handler_settings.name] = new_handler
        print(f"Handler '{handler_settings.name}' created and added with target {target_value!r}.")
        return new_handler

    def remove_handler(self, name: str) -> None:
        """
        Removes a handler from the manager and disconnects it from all loggers.

        Input Variables:
            name (str): The name of the handler to remove.

        Raises:
            ValueError: If the handler with the specified name is not found.
        """
        handler_to_remove: Optional[Handler] = self._handlers.get(name)

        if not handler_to_remove:
            raise ValueError(f"Handler with name '{name}' not found in the manager.")

        loggers_to_update: list[str] = [
            logger_name
            for logger_name, handler_names in list(self._logger_handler_connections.items())
            if name in handler_names
        ]

        for logger_name in loggers_to_update:
            logger_instance: Logger = logging.getLogger(logger_name)
            self._disconnect_handler_from_logger(logger_instance=logger_instance, handler_instance=handler_to_remove)
            print(f"Handler '{name}' removed from logger '{logger_name}'.")
            self._update_logger_connection_mapping(logger_name=logger_name, handler_name=name, action='remove')

        handler_to_remove.close()
        print(f"Handler '{name}' closed.")

        del self._handlers[name]
        print(f"Handler '{name}' removed from manager.")

    @overload
    def get_logger(self, name: str) -> Optional[Logger]:
        ...

    @overload
    def get_logger(self, settings: LoggerSettings) -> Logger:
        ...

    def get_logger(self, arg: str | LoggerSettings) -> Optional[Logger]:
        """
        Retrieves or creates a logger based on its name or settings.

        If a string name is provided:
        Retrieves an existing logger from the logging module's registry or the manager's internal list.

        If LoggerSettings are provided:
        Retrieves a logger by its name. If a logger with the same name already exists,
        it will be returned, and a warning will be issued via the root logger.
        Otherwise, a new logger will be created and configured based on the provided settings.

        Input Variables:
            arg (str | LoggerSettings): The name of the logger (str) or a LoggerSettings object.

        Returns:
            Optional[Logger]: The Logger object if found or created.
                                      Returns None only if `arg` is a string and the logger is not found.
        Raises:
            ValueError: If a Handler specified in LoggerSettings.linked_handlers is not found.
            TypeError: If the provided argument type is neither str nor LoggerSettings.
        """
        if isinstance(arg, str):
            name: str = arg
            if name == 'root' or name == '':
                return logging.getLogger('')

            logger_instance: Logger = logging.getLogger(name)
            return logger_instance if logger_instance in self._loggers else None

        elif isinstance(arg, LoggerSettings):
            settings: LoggerSettings = arg
            logger_name: str = settings.name

            logger_instance: Logger = logging.getLogger(logger_name)

            if logger_instance in self._loggers:
                root_logger_for_warning: Logger = logging.getLogger('')
                root_logger_for_warning.warning(
                    f"Logger '{logger_name}' already exists and is managed. "
                    "Returning existing instance without applying new settings. "
                    "If you need to update, consider a dedicated update_logger method."
                )
                return logger_instance
            else:
                self._loggers.append(logger_instance)

                logger_instance.setLevel(settings.level.value)

                for handler in list(logger_instance.handlers):
                    logger_instance.removeHandler(handler)

                for handler_name in settings.linked_handlers:
                    handler: Optional[Handler] = self.get_handler(handler_name)
                    if handler:
                        logger_instance.addHandler(handler)

                        self._update_logger_connection_mapping(logger_name=logger_name, handler_name=handler_name,
                                                               action='add')
                    else:
                        if logger_name in self._loggers:
                            self._loggers.remove(logger_instance)
                        if logger_name in self._logger_handler_connections:
                            del self._logger_handler_connections[logger_name]
                        raise ValueError(f"Handler '{handler_name}' not found for logger '{logger_name}'. "
                                         "Ensure all linked handlers are added first.")
            return logger_instance
        else:
            raise TypeError("Argument must be a string (logger name) or a LoggerSettings object.")

    def remove_logger(self, name: str) -> None:
        """
        Removes a logger from the manager and disconnects all its handlers.

        Input Variables:
            name (str): The name of the logger to remove.

        Raises:
            ValueError: If the logger with the specified name is not found or cannot be removed (e.g., root logger).
        """
        if name == 'root' or name == '':
            raise ValueError("Root logger cannot be explicitly removed by this method.")

        logger_instance: Logger = logging.getLogger(name)

        if logger_instance not in self._loggers:
            raise ValueError(f"Logger '{name}' not found or not managed by this manager.")

        for handler in list(logger_instance.handlers):
            logger_instance.removeHandler(handler)
            print(
                f"Handler '{handler.name if hasattr(handler, 'name') else handler.__class__.__name__}' removed from logger '{name}'.")

        if name in self._logger_handler_connections:
            self._logger_handler_connections.pop(name)
            print(f"Handler connections for logger '{name}' removed from manager.")

        logger_instance.setLevel(logging.NOTSET)
        print(f"Logger '{name}' level set to NOTSET.")

        self._loggers.remove(logger_instance)
        print(f"Logger '{name}' removed from manager's managed loggers list.")

    def link_handler_to_logger(self, logger_name: str, handler_name: str) -> None:
        """
        Links an existing handler to an existing logger.

        Input Variables:
            logger_name (str): The name of the logger to link the handler to.
            handler_name (str): The name of the handler to link.

        Raises:
            ValueError: If the specified logger or handler is not found,
                        or if the handler is already linked to the logger.
        """

        logger_instance: Logger = logging.getLogger(logger_name)

        if logger_instance not in self._loggers and logger_name != 'root' and logger_name != '':
            self._loggers.append(logger_instance)
            print(f"Logger '{logger_name}' is now managed by LoggingManager for handler linking.")

        handler_instance: Optional[Handler] = self.get_handler(handler_name)
        if not handler_instance:
            raise ValueError(f"Handler with name '{handler_name}' not found in the manager.")

        if handler_instance in logger_instance.handlers:
            raise ValueError(f"Handler '{handler_name}' is already linked to logger '{logger_name}'.")

        logger_instance.addHandler(handler_instance)
        print(f"Handler '{handler_name}' linked to logger '{logger_name}'.")

        self._update_logger_connection_mapping(logger_name=logger_name, handler_name=handler_name, action='add')
        print(f"Connection for logger '{logger_name}' to handler '{handler_name}' recorded.")

    def unlink_handler_to_logger(self, logger_name: str, handler_name: str) -> None:
        """
        Unlinks an existing handler from an existing logger.

        Input Variables:
            logger_name (str): The name of the logger to unlink the handler from.
            handler_name (str): The name of the handler to unlink.

        Raises:
            ValueError: If the specified logger or handler is not found,
                        or if the handler is not currently linked to the logger.
        """
        logger_instance: Logger = logging.getLogger(logger_name)

        if logger_instance not in self._loggers and logger_name != 'root' and logger_name != '':
            raise ValueError(f"Logger '{logger_name}' not found or not managed by this manager. Cannot unlink handler.")

        handler_instance: Optional[Handler] = self.get_handler(handler_name)
        if not handler_instance:
            raise ValueError(f"Handler with name '{handler_name}' not found in the manager.")

        if handler_instance not in logger_instance.handlers:
            raise ValueError(f"Handler '{handler_name}' is not linked to logger '{logger_name}'. Cannot unlink.")

        self._disconnect_handler_from_logger(logger_instance=logger_instance, handler_instance=handler_instance)
        print(f"Handler '{handler_name}' unlinked from logger '{logger_name}'.")

        self._update_logger_connection_mapping(logger_name=logger_name, handler_name=handler_name, action='remove')

    def add_components(self, settings_list: list[LogComponentSettings]) -> None:
        """
        Adds multiple loggers and/or handlers to the manager based on a list of settings.
        Handlers are processed before loggers to ensure proper linking.

        Input Variables:
            settings_list (list[LogComponentSettings]): A list of LoggerSettings and/or HandlerSettings
                                                      objects to be added.
        Raises:
            ValueError: If a handler specified in LoggerSettings is not found, or if
                        any component name conflicts with an existing one.
            TypeError: If an unsupported component type is found in the list.
        """

        handlers_to_add: list[HandlerSettings]
        loggers_to_add: list[LoggerSettings]
        handlers_to_add, loggers_to_add = self._classify_log_components(settings_list=settings_list)

        print("Adding handlers from settings list...")
        for handler_setting in handlers_to_add:
            try:
                self.add_handler(handler_settings=handler_setting)
                print(f"Successfully added handler '{handler_setting.name}'.")
            except (ValueError, TypeError) as e:
                print(f"Error adding handler '{handler_setting.name}': {e}")
                raise

        print("Adding loggers from settings list...")
        for logger_setting in loggers_to_add:
            try:

                self.get_logger(logger_setting)
                print(f"Successfully added/configured logger '{logger_setting.name}'.")
            except ValueError as e:
                print(f"Error adding logger '{logger_setting.name}': {e}")
                raise

        print("Finished processing components from settings list.")

    def remove_components(self, settings_list: list[LogComponentSettings]) -> None:
        """
        Removes multiple loggers and/or handlers from the manager based on a list of settings.
        Loggers are processed before handlers to ensure handlers are disconnected first.

        Input Variables:
            settings_list (list[LogComponentSettings]): A list of LoggerSettings and/or HandlerSettings
                                                      objects to be removed.
        Raises:
            ValueError: If a logger or handler name to be removed is not found.
            TypeError: If an unsupported component type is found in the list.
        """

        handlers_to_remove: list[HandlerSettings]
        loggers_to_remove: list[LoggerSettings]
        handlers_to_remove, loggers_to_remove = self._classify_log_components(settings_list=settings_list)

        print("Removing loggers from settings list...")
        for logger_setting in loggers_to_remove:
            try:
                self.remove_logger(name=logger_setting.name)
                print(f"Successfully removed logger '{logger_setting.name}'.")
            except ValueError as e:
                print(f"Error removing logger '{logger_setting.name}': {e}")

        print("Removing handlers from settings list...")
        for handler_setting in handlers_to_remove:
            try:
                self.remove_handler(name=handler_setting.name)
                print(f"Successfully removed handler '{handler_setting.name}'.")
            except ValueError as e:
                print(f"Error removing handler '{handler_setting.name}': {e}")

        print("Finished processing component removals from settings list.")

    def remove_unused_handler(self, handler_name: str) -> None:
        """
        Removes a handler from the manager if it is no longer linked to any managed logger.
        This function will call remove_handler if the handler is determined to be unused.

        Input Variables:
            handler_name (str): The name of the handler to check and potentially remove.

        Raises:
            ValueError: If the handler with the specified name is not found in the manager.
        """
        handler_instance: Optional[Handler] = self.get_handler(handler_name)
        if not handler_instance:
            raise ValueError(f"Handler with name '{handler_name}' not found in the manager.")

        if not self._is_handler_in_use(handler_name=handler_name, handler_instance=handler_instance):
            print(f"Handler '{handler_name}' is not attached to any active logger. Proceeding to remove.")
            self.remove_handler(handler_name)
        else:
            print(
                f"Handler '{handler_name}' is still attached to some logger(s) in the global logging registry. Not removing.")

    def _configure_initial_components(self, components: list[LogComponentSettings]) -> None:
        """
        Configures initial loggers and handlers based on a provided list of settings.
        This method is called during __init__ to set up the logging environment.
        It handles merging with default settings and ensures root logger and stdout handler are
        correctly configured and positioned.

        Input Variables:
            components (list[LogComponentSettings]): A list of settings to be processed.
                                                    This list may be default or user-provided.
        """

        user_handlers, user_loggers = self._classify_log_components(settings_list=components)

        stdout_default_setting: HandlerSettings = next(
            s for s in self._DEFAULT_INITIAL_COMPONENTS if isinstance(s, HandlerSettings) and s.name == 'stdout')
        user_stdout_setting: Optional[HandlerSettings] = next((s for s in user_handlers if s.name == 'stdout'), None)

        final_stdout_handler_setting: HandlerSettings
        if user_stdout_setting:
            if user_stdout_setting.output is not sys.stdout:
                raise ValueError(
                    f"Stdout handler '{user_stdout_setting.name}' specified an output '{user_stdout_setting.output!r}' "
                    "that is not sys.stdout. The 'stdout' handler must exclusively target sys.stdout."
                )

            final_stdout_handler_setting = HandlerSettings(
                name='stdout',
                level=user_stdout_setting.level,
                output=sys.stdout
            )

            user_handlers.remove(user_stdout_setting)
        else:
            final_stdout_handler_setting = stdout_default_setting

        root_default_setting: LoggerSettings = next(
            s for s in self._DEFAULT_INITIAL_COMPONENTS if isinstance(s, LoggerSettings) and s.name == 'root')
        user_root_setting: Optional[LoggerSettings] = next((s for s in user_loggers if s.name == 'root'), None)

        final_root_logger_setting: LoggerSettings
        if user_root_setting:
            linked_handlers_for_root: list[str] = list(user_root_setting.linked_handlers)
            if 'stdout' not in linked_handlers_for_root:
                linked_handlers_for_root.append('stdout')

            final_root_logger_setting = LoggerSettings(
                name='root',
                level=user_root_setting.level,
                linked_handlers=linked_handlers_for_root
            )

            user_loggers.remove(user_root_setting)
        else:
            final_root_logger_setting = root_default_setting

        final_ordered_components: list[LogComponentSettings] = []
        final_ordered_components.append(final_stdout_handler_setting)
        final_ordered_components.extend(user_handlers)
        final_ordered_components.append(final_root_logger_setting)
        final_ordered_components.extend(user_loggers)

        print("Configuring initial components via internal processing...")
        self.add_components(final_ordered_components)
        print("Initial components configured.")

    @staticmethod
    def _classify_log_components(settings_list: list[LogComponentSettings]) -> tuple[
        list[HandlerSettings], list[LoggerSettings]]:
        """
        Internal helper to classify LogComponentSettings into handlers and loggers,
        and warn about unknown types.

        Input Variables:
            settings_list (list[LogComponentSettings]): A list of settings to classify.

        Returns:
            tuple[list[HandlerSettings], list[LoggerSettings]]: A tuple containing
            two lists: the first for HandlerSettings and the second for LoggerSettings.
        """
        handlers: list[HandlerSettings] = [
            setting for setting in settings_list if isinstance(setting, HandlerSettings)
        ]
        loggers: list[LoggerSettings] = [
            setting for setting in settings_list if isinstance(setting, LoggerSettings)
        ]

        for setting in settings_list:
            if not isinstance(setting, (HandlerSettings, LoggerSettings)):
                print(f"Warning: Unknown LogComponentSettings type encountered: {type(setting)}. Skipping.")

        return handlers, loggers

    def _update_logger_connection_mapping(self, logger_name: str, handler_name: str,
                                          action: Literal['add', 'remove']) -> None:
        """
        Internal helper to manage the logger-handler connection mapping.

        Input Variables:
            logger_name (str): The name of the logger.
            handler_name (str): The name of the handler.
            action (Literal['add', 'remove']): The action to perform ('add' to link, 'remove' to unlink).

        Raises:
            ValueError: If an invalid action is specified.
        """
        if action == 'add':
            if logger_name not in self._logger_handler_connections:
                self._logger_handler_connections[logger_name]: list[str] = []
            if handler_name not in self._logger_handler_connections[logger_name]:
                self._logger_handler_connections[logger_name].append(handler_name)
                print(f"Connection mapping: Handler '{handler_name}' added for logger '{logger_name}'.")
            else:
                print(f"Connection mapping: Handler '{handler_name}' already mapped to logger '{logger_name}'.")
        elif action == 'remove':
            if logger_name in self._logger_handler_connections:
                if handler_name in self._logger_handler_connections[logger_name]:
                    self._logger_handler_connections[logger_name].remove(handler_name)
                    print(f"Connection mapping: Handler '{handler_name}' removed for logger '{logger_name}'.")

                    if not self._logger_handler_connections[logger_name]:
                        del self._logger_handler_connections[logger_name]
                        print(f"Connection mapping: Logger '{logger_name}' has no more handlers, entry removed.")
                else:
                    print(f"Connection mapping: Handler '{handler_name}' not found for logger '{logger_name}'.")
            else:
                print(f"Connection mapping: Logger '{logger_name}' has no recorded connections.")
        else:
            raise ValueError(f"Invalid action '{action}'. Must be 'add' or 'remove'.")

    @staticmethod
    def _disconnect_handler_from_logger(logger_instance: Logger, handler_instance: Handler) -> None:
        """
        Internal helper to safely remove a specific handler from a logger instance.

        Input Variables:
            logger_instance (Logger): The logger object from which to remove the handler.
            handler_instance (Handler): The handler object to remove.
        """
        if handler_instance in logger_instance.handlers:
            logger_instance.removeHandler(handler_instance)


        else:
            print(
                f"Warning: Handler '{handler_instance.name if hasattr(handler_instance, 'name') else handler_instance.__class__.__name__}' was not attached to logger '{logger_instance.name}'.")

    def _is_handler_in_use(self, handler_name: str, handler_instance: Handler) -> bool:
        """
        Internal helper to check if a handler is currently in use by any managed logger
        or any logger in the global logging registry.

        Input Variables:
            handler_name (str): The name of the handler.
            handler_instance (Handler): The handler object to check.

        Returns:
            bool: True if the handler is in use, False otherwise.
        """

        is_linked_by_manager: bool = any(
            handler_name in connections for connections in self._logger_handler_connections.values()
        )

        if is_linked_by_manager:
            return True

        is_attached_to_managed_logger: bool = any(
            handler_instance in logger_obj.handlers for logger_obj in self._loggers
        )
        if is_attached_to_managed_logger:
            return True

        if handler_instance in logging.getLogger().handlers:
            return True

        return False

    @classmethod
    def create_predefined_manager(cls) -> 'LoggingManager':
        """
        Creates and returns a LoggingManager instance configured with a specific set
        of loggers and handlers as defined by the user.

        The configuration includes:
        - root_logger: level=LogLevel.INFO, linked_handlers=['stdout', 'main_log']
        - stdout_handler: level=LogLevel.INFO
        - main_log_handler: name='main_log', level=LogLevel.INFO,
                            output based on current time and root/main_log levels
        - machine_learning_logger: name='machine_learning', level=LogLevel.INFO, linked_handlers=[]

        Returns:
            LoggingManager: An initialized LoggingManager instance with the predefined components.
        """

        common_log_level: LogLevel = LogLevel.INFO

        log_directory: Path = Path(__file__).resolve(strict=True).parent.parent / "log"
        log_directory.mkdir(parents=True, exist_ok=True)

        level_name_for_filename: str = logging.getLevelName(max(common_log_level.value, common_log_level.value))

        current_time_str: str = datetime.now().strftime('%Y-%m-%dT%H：%M：%S%Z')
        main_log_file_name: str = f"{current_time_str}_MAIN_{level_name_for_filename}.log"

        main_log_handler_settings: HandlerSettings = HandlerSettings(
            name='main_log',
            level=common_log_level,
            output=log_directory / main_log_file_name
        )

        stdout_handler_settings: HandlerSettings = HandlerSettings(
            name='stdout',
            level=common_log_level,
            output=sys.stdout
        )

        root_logger_settings: LoggerSettings = LoggerSettings(
            name='root',
            level=common_log_level,
            linked_handlers=['stdout', 'main_log']
        )

        machine_learning_logger_settings: LoggerSettings = LoggerSettings(
            name='machine_learning',
            level=LogLevel.INFO,
            linked_handlers=[]
        )

        initial_components_list: list[LogComponentSettings] = [
            stdout_handler_settings,
            main_log_handler_settings,
            root_logger_settings,
            machine_learning_logger_settings
        ]

        print("Creating LoggingManager with predefined components...")
        return cls(initial_configs=initial_components_list)


if __name__ == '__main__':
    manager: LoggingManager = LoggingManager.create_predefined_manager()
    main_logger = manager.get_logger('root')
    ml_logger = manager.get_logger('machine_learning')
    manager.add_handler(
        handler_settings=HandlerSettings(name='machine_learning', level=LogLevel.INFO, output=Path('ml.log')))
    manager.link_handler_to_logger(logger_name='machine_learning', handler_name='machine_learning')
    ml_logger.info("aaaa")
    main_logger.info("bbbb")
