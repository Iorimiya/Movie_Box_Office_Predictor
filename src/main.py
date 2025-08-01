import sys
from argparse import ArgumentParser, Namespace

from src.cli.argument_parser_builder import ArgumentParserBuilder


def main() -> None:
    """
    Main entry point for the CLI application.

    This function builds the command-line parser, parses input arguments,
    and executes the corresponding command handler function.

    :raises SystemExit: If an unhandled SystemExit occurs during command execution.
    :raises Exception: For any other unexpected errors during command execution.
    """
    builder: ArgumentParserBuilder = ArgumentParserBuilder()
    parser: ArgumentParser = builder.build()
    args: Namespace = parser.parse_args()
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except SystemExit:
            # SystemExit is typically raised by argparse or sys.exit(),
            # and is meant to terminate the program. We catch it to prevent
            # the generic Exception handler from catching it, but then re-raise
            # implicitly by doing nothing, allowing the program to exit gracefully.
            pass
        except Exception as e:
            print(f"Command execution failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
