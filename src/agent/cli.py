"""
CLI entry point for the GUI automation agent.
"""

import argparse
import os
from subprocess import DEVNULL, run

from .agent import GUIAutomationAgent


def clear_terminal() -> None:
    run(
        ["cls"] if os.name == "nt" else ["clear"],
        check=False,
        stdout=DEVNULL,
        stderr=DEVNULL,
        shell=os.name == "nt",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gui-agent",
        description="Vision-based GUI automation agent.",
    )

    parser.add_argument(
        "task",
        nargs="*",
        help="Task to execute.",
    )

    return parser


def interactive(agent: GUIAutomationAgent) -> None:
    print("GUI Automation Agent")
    print("\nCommands\n")
    print("  exit        Quit")
    print("  quit        Quit")
    print("  help        Show help")
    print("  clear       Clear the terminal")
    print("\nEnter a GUI task.\n")
    print("\n>\n")

    command: str | None = None

    while True:
        try:
            command = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            continue

        command = command.lower()

        if command == "help":
            print(
                """
    Available commands

      help      Show this help message.
      clear     Clear the terminal.
      status    Show the current agent state.
      exit      Exit the program.
      quit      Exit the program.

    Anything else is treated as a GUI automation task.
    """
            )
            continue

        elif command == "clear":
            clear_terminal()
            continue

        elif command == "status":
            try:
                print(agent.context)
            except RuntimeError:
                print("No task is currently executing.")

            continue

        elif command in {"exit", "quit"}:
            break

        try:
            result = agent.run(command)

            if result:
                print(result)

        except Exception as exc:
            print(f"Error: {exc}")

        print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    with GUIAutomationAgent() as agent:
        if args.task:
            result = agent.run(" ".join(args.task))

            if result:
                print(result)

        else:
            interactive(agent)


if __name__ == "__main__":
    main()
