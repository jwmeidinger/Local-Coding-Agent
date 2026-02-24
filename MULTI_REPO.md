# Multi-Repo Configuration

If you have a workspace folder (e.g., `~/Code`) with multiple repositories, you can configure which repos the agent should work with.

## Quick Examples

### Auto-discover all git repos in a folder:
```bash
# Discover all git repos in ~/Code
python coding_agent.py --repos ~/Code --auto-discover

# Ignore certain repos
python coding_agent.py --repos ~/Code --auto-discover --ignore "archive/*" "old-*" "temp*"
```

### Specify repos explicitly:
```bash
# Command line
python coding_agent.py \
  --repo ~/Code/project1 \
  --repo ~/Code/project2 \
  --repo ~/Code/my-api

# Or use a file
python coding_agent.py --repo-list repos.txt
```

### repos.txt format:
```
# Active projects:
~/Code/project1
~/Code/project2
~/Code/api-service

# Commented out (ignored):
# ~/Code/archived-project
```

## Ignore Patterns

Use `--ignore` with glob patterns to skip repos:

```bash
# Ignore repos starting with "temp-"
python coding_agent.py --repos ~/Code --auto-discover --ignore "temp-*"

# Ignore repos in archive folder
python coding_agent.py --repos ~/Code --auto-discover --ignore "archive/*"

# Multiple patterns
python coding_agent.py --repos ~/Code --auto-discover --ignore "*old*" "*backup*" "archive/*"
```

## Specifying Which Repo for a Task

When you create a task, you can explicitly specify which repo it belongs to:

```
tasks/add-feature.txt:
REPO: project1

Add user authentication to the login endpoint
```

Or the agent will try to auto-detect based on the task content using vector search.

## How It Works

1. **Task Discovery**: Looks in the central `tasks/` directory or each repo's `tasks/` subdirectory
2. **Repo Routing**: 
   - Checks for explicit `REPO: reponame` in task
   - Uses vector search to find which repo has relevant code
   - Defaults to first repo if can't determine
3. **Execution**: Each repo has its own ExecutionEngine with isolated git operations
4. **Indexing**: Each repo is indexed separately in the vector database
5. **Search**: Searches across all indexed repos

## Recommended Setup

For a `~/Code` workspace with many repos:

```bash
# 1. Create repos.txt with active projects
cat > ~/Code/repos.txt << 'EOF'
~/Code/my-main-project
~/Code/shared-library
~/Code/api-service
EOF

# 2. Start Postgres
docker-compose up -d

# 3. Index all repos
python coding_agent.py --repo-list ~/Code/repos.txt --index

# 4. Run agent
python coding_agent.py --repo-list ~/Code/repos.txt
```

## Central vs Per-Repo Tasks

You have two options for task organization:

### Option 1: Central Tasks Directory
```
~/Code/
  repos.txt
  tasks/           # Central task queue
    task1.txt
    task2.txt
  project1/        # Individual repos
  project2/
```

### Option 2: Per-Repo Tasks
```
~/Code/
  project1/
    tasks/         # Tasks specific to project1
      task1.txt
  project2/
    tasks/         # Tasks specific to project2
      task2.txt
```

The agent checks both locations and routes tasks to the appropriate repo.
