# Coding Agent

A lightweight, coding-focused autonomous agent inspired by OpenClaw. Processes tasks while you sleep and creates isolated git branches for review.

## Inspired by OpenClaw

This agent draws inspiration from [OpenClaw](https://github.com/OpenClaw) (187K+ GitHub stars), a popular 2026 AI agent framework. While OpenClaw is designed for general automation with messaging platform integrations, this is a **lightweight, focused version for coding only**.

**Key concepts from OpenClaw adopted:**
- **Skills System**: Structured guides for different coding tasks
- **Execution Loop**: Plan → Execute → Review → Iterate
- **Tool System**: File operations, bash commands, git operations
- **Context Persistence**: Memory across sessions

**Simplified/removed:**
- No messaging platform integrations (WhatsApp, Telegram, etc.)
- No web dashboard or mobile apps
- No complex gateway architecture
- No Docker sandboxing

## Features

- **Skill-Based**: Auto-detects task type (refactor, feature, bugfix, docs, test)
- **Codebase Memory**: Indexes your codebase to avoid re-reading files (saves tokens)
- **Planning Phase**: Creates execution plans before coding
- **Review Phase**: Self-reviews work and iterates if needed
- **Git Integration**: Creates `agent/task-name-MMDD-HHMM` branches
- **Local Only**: Never pushes branches - review locally in the morning
- **Tool System**: File read/write, bash execution, git operations, web search
- **Incremental Memory**: Updates memory after each task with modified files
- **Multi-Repo Support**: Works across multiple repositories (see [MULTI_REPO.md](./MULTI_REPO.md))

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Vector Memory Database

For codebase indexing with semantic search, start the database:

```bash
docker-compose up -d
```

This is optional - the agent works without it.

### 2. Setup LLM

Before running the agent, you need a local LLM running. See [LLM_SETUP.md](./LLM_SETUP.md) for details.

### 3. Setup Your Repository

```bash
cd your-project
git status  # Ensure you're in a git repo

# Create directories
mkdir -p tasks skills .coding-agent
```

### 4. Create a Task

Add tasks as text files in the `tasks/` directory:

```bash
echo "Refactor the auth module to use dependency injection" > tasks/refactor-auth.txt
echo "Add error handling to the API endpoints" > tasks/error-handling.txt
echo "Write unit tests for the UserService class" > tasks/test-userservice.txt
```

### 5. Run the Agent

```bash
# With default settings
python coding_agent.py

# With specific model
python coding_agent.py --model codellama

# Custom repository
python coding_agent.py --repo /path/to/project
```

### 6. Review in the Morning

```bash
# See branches created by the agent
git branch | grep agent/

# Review a branch
git checkout agent/refactor-auth-0217-0830
git diff main

# If it looks good
git checkout main
git merge agent/refactor-auth-0217-0830

# Or discard
 git branch -D agent/refactor-auth-0217-0830
```

## Architecture

### Execution Flow

```
Task Discovery → Skill Detection → Planning → Execution → Review → Commit
```

1. **Task Discovery**: Reads `.txt` files from `tasks/` directory
2. **Skill Detection**: Auto-detects task type (refactor/feature/bugfix/docs)
3. **Planning**: Creates step-by-step execution plan using LLM
4. **Execution**: Runs plan using available tools
5. **Review**: Self-reviews changes, suggests improvements
6. **Iteration**: Cycles back if review doesn't PASS
7. **Commit**: Creates branch and commits changes locally

### Skills System

Built-in skills for different coding tasks:

| Skill | Description | Auto-Detection Keywords |
|-------|-------------|------------------------|
| `refactor` | Restructure code for better quality | "refactor", "clean", "restructure", "improve" |
| `feature` | Implement new functionality | Default skill |
| `bugfix` | Fix bugs and issues | "bug", "fix", "error", "crash", "broken" |
| `docs` | Add documentation | "doc", "comment", "readme", "guide" |
| `test` | Create unit tests | "test", "tests", "testing", "unittest", "pytest", "spec" |

Each skill defines:
- System prompt (persona and guidelines)
- Planning prompt (how to create execution plans)
- Review prompt (how to evaluate changes)
- Preferred tools

### Tool System

Available tools the agent can use:

- `file_read(path)` - Read file contents
- `file_write(path, content)` - Write to a file
- `bash(command)` - Execute bash commands
- `list_files(path, pattern)` - List directory contents
- `git_status()` - Check git status

### Directory Structure

```
your-project/
  tasks/                    # Drop task files here
    refactor-auth.txt
    add-tests.txt
  skills/                   # Custom skills (optional)
    my-skill.json
  soul.md                   # Custom context for the agent (optional)
  .coding-agent/            # Agent workspace
    archive/               # Completed tasks moved here
    agent.log              # Detailed logs
    reports/               # Abort reports when tasks are blocked
  coding_agent.py          # The agent
```

## Codebase Memory System

The agent maintains a **persistent memory** of your codebase to avoid re-reading files for every task. This saves tokens and improves performance.

### How It Works

1. **First Run**: The agent indexes your entire codebase
   - Scans all source files (Python, JavaScript, TypeScript, Java, Go, Rust, etc.)
   - Extracts function/class names
   - Identifies imports/dependencies
   - Generates summaries using LLM (if available)
   - Saves to `.coding-agent/codebase_memory.json`

2. **Subsequent Runs**: Loads existing memory
   - Fast startup - no re-indexing needed
   - Memory is used to provide context to the LLM

3. **Automatic Updates**: After each task
   - Identifies modified files
   - Re-indexes only those files
   - Updates memory with new summaries
   - Memory is refreshed if files changed outside the agent

4. **Stale Detection**: Memory auto-refreshes if
   - Older than 24 hours
   - Tracked files have been modified

### Memory Contents

```json
{
  "indexed_at": "2026-02-17T23:00:00",
  "files": {
    "src/auth.py": {
      "file_type": "source",
      "summary": "Authentication module with JWT handling",
      "key_functions": ["login", "logout", "verify_token"],
      "dependencies": ["jwt", "flask", "bcrypt"]
    }
  },
  "test_files": ["tests/test_auth.py"],
  "entry_points": ["src/main.py"],
  "dependencies": {
    "python": ["flask", "pytest", "requests"]
  }
}
```

### Benefits

- **Saves Tokens**: No need to re-ingest files for every task
- **Better Context**: Agent knows the codebase structure upfront
- **Faster Planning**: Can reference existing patterns and conventions
- **Incremental Updates**: Only modified files are re-indexed

## Configuration

### Command Line Options

```
--repo PATH               Git repository path (default: current directory)
--tasks-dir PATH          Tasks directory (default: ./tasks)
--skills-dir PATH         Skills directory (default: ./skills)
--base-branch NAME        Base git branch (default: main)
--branch-prefix PREFIX   Branch prefix (default: agent/)

--repos PATH             Path to scan for repos (use with --auto-discover)
--repo-list PATH          File with list of repos (one per line)
--auto-discover           Auto-find git repos in --repos path
--ignore PATTERN          Ignore patterns (can be repeated)

--llm-url URL            LLM API server URL
--model NAME             LLM model (default: codellama)
--temperature FLOAT      LLM temperature (default: 0.2)

--max-iterations N       Max iterations per task (default: 5)
--no-commit              Don't auto-commit changes

--list-skills            List available skills
-v, --verbose            Enable verbose logging
```

For multi-repo configuration, see [MULTI_REPO.md](./MULTI_REPO.md).

### Creating Custom Skills

Add custom skills by creating JSON files in `skills/`:

```json
{
  "name": "performance",
  "description": "Optimize code for better performance",
  "system_prompt": "You are a performance optimization specialist...",
  "planning_prompt": "Analyze performance bottlenecks and create optimization plan...",
  "review_prompt": "Review the performance improvements...",
  "preferred_tools": ["file_read", "file_write", "bash"]
}
```

## Task Examples

### Refactoring Task
```
tasks/refactor-user-service.txt:
Refactor the UserService class to reduce complexity:
- Extract validation logic into a separate Validator class
- Use dependency injection for the database connection
- Add proper error handling
- Keep all existing tests passing
```

### Feature Task
```
tasks/add-pagination.txt:
Add pagination support to the /api/users endpoint:
- Accept page and limit query parameters
- Return total count in response headers
- Default to page 1, limit 20
- Validate parameters
```

### Bug Fix Task
```
tasks/fix-null-pointer.txt:
Fix the null pointer exception in OrderProcessor:
- The error occurs when processing orders without a customer
- Add null check before accessing customer fields
- Add a test case for this scenario
```

### Documentation Task
```
tasks/document-api.txt:
Document the REST API endpoints:
- Add docstrings to all public methods in api/routes.py
- Create API.md with endpoint descriptions
- Include request/response examples
```

### Unit Test Task
```
tasks/test-user-service.txt:
Write comprehensive unit tests for the UserService class:
- Test user registration with valid/invalid data
- Test login with correct and incorrect credentials
- Test edge cases (empty strings, null values, special characters)
- Mock the database layer
- Follow the existing test patterns in tests/test_*.py
- Aim for at least 90% code coverage
```

## Automation / Cron

Run nightly with cron:

```bash
# Edit crontab
crontab -e

# Run at 2 AM every day
0 2 * * * cd /path/to/project && python /path/to/coding_agent.py >> /path/to/coding-agent-cron.log 2>&1
```

Or use a systemd timer for better logging and control.

## Comparison with OpenClaw

| Feature | OpenClaw | Coding Agent |
|---------|----------|--------------|
| **Focus** | General automation | Coding only |
| **Architecture** | Gateway + Agent Runtime | Single script |
| **Integrations** | WhatsApp, Telegram, Discord, Slack, etc. | None |
| **Skills System** | ✓ | ✓ (simplified) |
| **Memory** | SQLite + vector embeddings | File-based |
| **Sandboxing** | Docker | None |
| **Channels** | Multiple | Git only |
| **Complexity** | High | Low |
| **Git Workflow** | Various | Local branches only |

## Requirements

- Python 3.8+
- GitPython
- LangChain + LangChain-Community
- Ollama (for local LLMs)

## License

MIT License
