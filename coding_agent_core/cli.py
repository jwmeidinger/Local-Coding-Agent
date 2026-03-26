from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AgentConfig
from .deps import VECTOR_MEMORY_AVAILABLE, VectorMemoryManager
from .skills import SkillRegistry


def parse_args() -> AgentConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Coding Agent - Autonomous coding assistant inspired by OpenClaw",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (current directory)
  python coding_agent.py
  
  # Multiple repos with auto-discovery
  python coding_agent.py --repos ~/Code --auto-discover --ignore "archive/*" "old-*"
  
  # Explicitly specify repos
  python coding_agent.py --repo ~/Code/project1 --repo ~/Code/project2
  
  # Specify repos via file
  python coding_agent.py --repo-list repos.txt
  
  # Index all repos
  python coding_agent.py --repos ~/Code --auto-discover --index
  
  # List available skills
  python coding_agent.py --list-skills
        """
    )
    
    # Multi-repo options
    parser.add_argument("--repo", type=Path, action="append", dest="repo_paths",
                        help="Path to git repository (can specify multiple)")
    parser.add_argument("--repos", type=Path,
                        help="Parent directory containing multiple repos")
    parser.add_argument("--auto-discover", action="store_true",
                        help="Auto-discover git repos in --repos directory")
    parser.add_argument("--ignore", type=str, action="append", dest="ignore_patterns",
                        help="Patterns to ignore (e.g., 'archive/*', 'temp-*')")
    parser.add_argument("--repo-list", type=Path,
                        help="File containing list of repo paths (one per line)")
    
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"),
                        help="Directory containing task files")
    parser.add_argument("--skills-dir", type=Path, default=Path("skills"),
                        help="Directory containing custom skills")
    parser.add_argument("--base-branch", default="main",
                        help="Base git branch")
    parser.add_argument("--branch-prefix", default="agent/",
                        help="Prefix for agent branches")
    
    parser.add_argument("--llm-url", default="http://localhost:11434",
                        help="LLM API server URL")
    parser.add_argument("--model", default="codellama",
                        help="LLM model to use")
    parser.add_argument("--source-ip", default="",
                        help="Bind to specific local IP (e.g., 10.152.50.103) to bypass VPNs")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM temperature")
    parser.add_argument("--embed-model", default="",
                        help="Embedding model loaded in LM Studio for semantic search "
                             "(e.g. nomic-embed-text-v1.5). Uses same LLM_URL server.")
    parser.add_argument("--embed-dims", type=int, default=768,
                        help="Output dimensions of the embedding model "
                             "(nomic-embed-text-v1.5=768, mxbai-embed-large-v1=1024). "
                             "Must match the model exactly. (default: 768)")
    
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max execution iterations per task")
    parser.add_argument("--max-prompt-chars", type=int, default=80000,
                        help="Max characters sent to LLM per request (~20k tokens). "
                             "Lower this if your model has a small context window. (default: 80000)")
    parser.add_argument("--max-tool-result-chars", type=int, default=3000,
                        help="Max characters kept per tool result in older history entries (default: 3000)")
    parser.add_argument("--max-consecutive-errors", type=int, default=2,
                        help="Abort after N consecutive LLM failures/timeouts (default: 2)")
    parser.add_argument("--no-commit", action="store_true",
                        help="Don't auto-commit changes")
    parser.add_argument("--build-command", type=str, default="",
                        help="Build/check command to run after every file edit (e.g. 'npm run build', 'tsc --noEmit', 'python -m py_compile')")
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable auto-verification after file writes")
    
    parser.add_argument("--list-skills", action="store_true",
                        help="List available skills and exit")
    parser.add_argument("--index", action="store_true",
                        help="Index the codebase and exit (builds vector memory)")
    parser.add_argument("--search", type=str, metavar="QUERY",
                        help="Search codebase and exit (for testing vector memory)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Collect repo paths
    repo_paths = []
    
    # From explicit --repo args
    if args.repo_paths:
        repo_paths.extend(args.repo_paths)
    
    # From repo list file
    if args.repo_list and args.repo_list.exists():
        with open(args.repo_list) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    repo_paths.append(Path(line))
    
    # From --repos directory with auto-discovery
    if args.repos:
        if args.auto_discover:
            # Find all git repos in the directory
            for item in args.repos.iterdir():
                if item.is_dir() and (item / '.git').exists():
                    repo_paths.append(item)
        else:
            # Just use the directory itself as a single repo
            repo_paths.append(args.repos)
    
    # Default to current directory if nothing specified
    if not repo_paths:
        repo_paths = [Path(".")]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_repo_paths = []
    for path in repo_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_repo_paths.append(path)
    repo_paths = unique_repo_paths
    
    if args.list_skills:
        skills = SkillRegistry(args.skills_dir)
        print("Available Skills:")
        print(skills.list_skills())
        sys.exit(0)
    
    if args.index:
        if not VECTOR_MEMORY_AVAILABLE:
            print("Error: Vector memory not available. Install psycopg2-binary", file=sys.stderr)
            sys.exit(1)

        embed_label = args.embed_model or "fallback (non-semantic)"
        print(f"Embedding model: {embed_label}  dims: {args.embed_dims}")

        for repo_path in repo_paths:
            print(f"\nIndexing codebase: {repo_path}")
            try:
                memory_manager = VectorMemoryManager(
                    repo_path,
                    embed_url=args.llm_url,
                    embed_model=args.embed_model,
                    embed_dims=args.embed_dims,
                )
                stats = memory_manager.index_codebase()
                print(f"  Files indexed: {stats['indexed']}")
                print(f"  Files updated: {stats['updated']}")
                print(f"  Files skipped: {stats['skipped']}")
                print(f"  Errors: {stats['errors']}")
                
                summary = memory_manager.get_codebase_summary()
                print(f"  {summary}")
                memory_manager.close()
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
        
        print("\nIndexing complete!")
        sys.exit(0)
    
    if args.search:
        if not VECTOR_MEMORY_AVAILABLE:
            print("Error: Vector memory not available. Install psycopg2-binary", file=sys.stderr)
            sys.exit(1)
        
        print(f"Searching for: {args.search}")
        print(f"In repos: {[str(p) for p in repo_paths]}")
        
        for repo_path in repo_paths:
            print(f"\n--- Results from {repo_path.name} ---")
            try:
                memory_manager = VectorMemoryManager(
                    repo_path,
                    embed_url=args.llm_url,
                    embed_model=args.embed_model,
                    embed_dims=args.embed_dims,
                )
                results = memory_manager.search_codebase(args.search, limit=10)
                if results:
                    for i, r in enumerate(results, 1):
                        print(f"{i}. {r['file_path']} (score: {r['combined_score']:.3f})")
                        print(f"   Summary: {r.get('summary') or ''}")
                        kf = r.get('key_functions') or []
                        if kf:
                            print(f"   Functions: {', '.join(kf[:5])}")
                else:
                    print("  No results")
                memory_manager.close()
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
        
        sys.exit(0)
    
    return AgentConfig(
        repo_paths=repo_paths,
        ignore_patterns=args.ignore_patterns or [],
        tasks_dir=args.tasks_dir,
        skills_dir=args.skills_dir,
        base_branch=args.base_branch,
        branch_prefix=args.branch_prefix,
        llm_url=args.llm_url,
        model=args.model,
        source_ip=args.source_ip,
        temperature=args.temperature,
        max_prompt_chars=args.max_prompt_chars,
        max_tool_result_chars=args.max_tool_result_chars,
        max_iterations=args.max_iterations,
        max_consecutive_errors=args.max_consecutive_errors,
        auto_commit=not args.no_commit,
        verbose=args.verbose,
        build_command=args.build_command,
        verify_after_write=not args.no_verify,
        embed_model=args.embed_model,
        embed_dims=args.embed_dims,
    )