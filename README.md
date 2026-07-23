# Auto GUI

A vision-based GUI automation framework with pluggable grounding providers.

Auto GUI combines a Vision Language Model (VLM), GUI grounding engines, and desktop automation into a modular framework for building intelligent desktop agents.

The project is designed around interchangeable components:

* **Vision models** decide what to do next.
* **Grounding providers** locate GUI elements on screen.
* **Action backends** perform mouse and keyboard interactions.
* **Screenshot backends** capture the desktop.
* **LangChain** is currently used as the orchestration layer, but the core libraries are intentionally kept framework-independent.

---

## Features

* Vision-driven GUI automation
* Multiple grounding providers

  * VLM Run
  * OmniParser
  * GroundingDINO
  * Easily extensible
* Pluggable screenshot backends
* Pluggable action backends
* Typed Pydantic models throughout
* LangChain agent integration
* Runtime execution context
* Sync and async APIs
* Fully type checked
* Poetry-based dependency management

---

## Project Structure

```text
src/
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── cli.py
│   ├── context.py
│   ├── prompts.py
│   ├── vlm.py
│   ├── config.py
│   ├── prompts/
│   │   ├── system.md
│   │   └── task.md
│   └── tools/
│       ├── action/
│       │   ├── __init__.py
│       │   ├── action_models.py
│       │   ├── actions.py
│       │   └── actions_async.py
│       ├── screenshot/
│       │   ├── __init__.py
│       │   ├── backend.py
│       │   ├── pyautogui_backend.py
│       │   └── screenshot.py
│       ├── __init__.py
│       ├── grounding.py
│       ├── tool_models.py
│       ├── windows.py
│       └── agent_tools.py
│
└── grounding/
    ├── prompts/
    │   └── orion_grounding_system_prompt.md
    ├── providers/
    │   ├── __init__.py
    │   ├── base.py
    │   └── vlmrun.py
    ├── schemas/
    │   └── orion_grounding.schema.json
    ├── utils/
    │   ├── __init__.py
    │   └── images.py
    ├── __init__.py
    ├── client.py
    ├── config.py
    ├── exceptions.py
    ├── registry.py
    ├── models.py
    └── interfaces.py
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/uchokoro/auto-gui.git

cd auto-gui
```

Install dependencies.

```bash
poetry install
```

Activate the virtual environment.

```bash
poetry shell
```

---

## Configuration

Create a `.env` file.

Example:

```text
AGENT_API_KEY=...
AGENT_BASE_URL=...

VLMRUN_API_KEY=...
VLMRUN_BASE_URL=...

GROUNDING_PROVIDER=vlmrun

SCREENSHOT_DIRECTORY=./screenshots

KEEP_SCREENSHOTS=true
```

Additional configuration options are available in:

```text
src/agent/config.py
and
src/grounding/config.py
```

---

## Running

Start the interactive CLI.

```bash
poetry run gui-agent
```

or

```bash
python -m agent.cli
```

Example session

```text
GUI Agent
==========

gui> Open Calculator

✓ Calculator opened.

gui> Type "123+456"

✓ Done.

gui> exit
```

---

## Architecture

```text
                +---------------------+
                | Vision Language     |
                | Model               |
                +----------+----------+
                           |
                           v
                  LangChain Agent
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
 Screenshot         Grounding          Action Manager
  Manager            Manager
        |                  |
        +--------+---------+
                 |
          Grounding Provider
```

---

## Execution Flow

1. Capture the screen.
2. Locate relevant GUI elements.
3. Choose the appropriate action.
4. Execute the action.
5. Observe the updated screen.
6. Repeat until the task is complete.

---

## Grounding Providers

Grounding providers convert natural-language queries into GUI element detections.

Example:

```python
response = locate_image(
    image="desktop.png",
    query="Search box",
    provider=vlmrun_provider
)
```

The currently supported providers is VLM Run, however, it is intended to support the following additional providers:

* OmniParser
* GroundingDINO

Additional providers can be registered through the grounding registry.

---

## Development

Install development dependencies.

```bash
poetry install --with dev
```

Run Ruff.

```bash
ruff check .
```

Format code.

```bash
ruff format .
```

Run MyPy.

```bash
mypy src
```

Run tests.

```bash
pytest
```

---

## Pre-commit Hooks

Install the hooks.

```bash
pre-commit install
```

Run them manually.

```bash
pre-commit run --all-files
```

---

## Design Goals

* Modular
* Strong typing
* Backend agnostic
* Easily extensible
* Framework independent where possible
* Testable
* Production ready

---

## Roadmap

* [ ] Additional grounding providers
* [ ] Additional screenshot backends
* [ ] Additional action backends
* [ ] Browser automation backend
* [ ] Agent memory
* [ ] Planning module
* [ ] Replay and session recording
* [ ] Visual debugging tools
* [ ] Benchmark suite
* [ ] Plugin system

---

## License

MIT License.
