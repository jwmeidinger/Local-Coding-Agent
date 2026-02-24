# LLM Setup

The agent connects to a local LLM API to reason and generate code. You need to have a model running before starting the agent.

## Options

### Ollama

[Ollama](https://ollama.ai) is the default option.

1. Install from ollama.ai
2. Start the server: `ollama serve`
3. Pull a model: `ollama pull <model-name>`

### LM Studio

[LM Studio](https://lmstudio.ai) is another popular option.

1. Install the app
2. Download a model through the UI
3. Start a local server (usually port 1234)

## Choosing a Model

You need a good reasoning and coding model that fits on your system's RAM.

Check your system's capabilities and choose accordingly.

## Configuration

Pass the URL and model name when running the agent:

```bash
# Ollama (default)
python coding_agent.py --model codellama

# LM Studio (example port)
python coding_agent.py --llm-url http://localhost:1234 --model llama3

# With specific source IP to bypass VPN
python coding_agent.py --llm-url http://192.168.1.100:1234 --source-ip 192.168.1.50 --model llama3
```

The agent will fail gracefully if it can't connect to the LLM.

## Optional Features

### Web Search

For web search capability, install exa-py:

```bash
pip install exa-py
```

This enables the agent to search the web for general information (code searches are blocked for safety).

### Vector Memory (Codebase Indexing)

For persistent codebase memory with semantic search, you'll need PostgreSQL with pgvector. Use docker-compose:

```bash
docker-compose up -d
```

This starts a database container. The agent will automatically use it for indexing your codebase and semantic search. Without this, the agent still works but will re-read files on each run.

### Bypassing VPNs

If a VPN blocks connections to your local LLM provider, you can bind to a specific network interface:

```bash
# Find your local IP (e.g., Wi-Fi interface)
ip addr show | grep "inet "

# Use that IP as source-ip to bind outgoing connections
python coding_agent.py --llm-url http://10.152.50.147:1234 --source-ip 10.152.50.103 --model llama3
```

This binds the agent's outgoing connections to your Wi-Fi IP, bypassing the VPN tunnel.
