from __future__ import annotations

from typing import Iterable


TOOL_DESCRIPTIONS = [
    {
        "name": "todo_read",
        "arguments": {},
        "description": "Read the in-session todo list for complex multi-step work.",
    },
    {
        "name": "todo_write",
        "arguments": {"items": "list of {content,status,id?}; status pending|in_progress|completed"},
        "description": "Replace the in-session todo list. Keep at most one item in_progress.",
    },
    {
        "name": "list_files",
        "arguments": {"path": "relative path, default .", "max_depth": "int, default 4", "limit": "int, default 200"},
        "description": "List files and directories under the workspace.",
    },
    {
        "name": "read_file",
        "arguments": {"path": "relative path", "start": "1-based start line, default 1", "end": "inclusive end line, default 200"},
        "description": "Read a file with line numbers.",
    },
    {
        "name": "search",
        "arguments": {"query": "regex or plain text", "path": "relative path, default .", "limit": "int, default 100"},
        "description": "Search text in the workspace.",
    },
    {
        "name": "file_search",
        "arguments": {"query": "filename/path terms", "path": "relative path, default .", "limit": "int, default 100"},
        "description": "Search cached workspace file paths quickly without reading file contents.",
    },
    {
        "name": "fd_search",
        "arguments": {"query": "fd filename pattern", "path": "relative path, default .", "limit": "int, default 100", "kind": "any|file|dir, default any"},
        "description": "Use fd/fdfind for fast repo-local file discovery when installed.",
    },
    {
        "name": "file_index_refresh",
        "arguments": {"path": "relative path, default .", "limit": "int, default 50000"},
        "description": "Refresh the persistent workspace file path index for fast future file searches.",
    },
    {
        "name": "everything_search",
        "arguments": {"query": "Everything search query", "path": "relative path, default .", "limit": "int, default 100"},
        "description": "Use the native Everything CLI es.exe when installed, constrained to the workspace path.",
    },
    {
        "name": "search_symbols",
        "arguments": {"query": "symbol name regex/text", "path": "relative path, default .", "limit": "int, default 50"},
        "description": "Search code symbols by name without reading full files.",
    },
    {
        "name": "code_outline",
        "arguments": {"path": "file or directory, default .", "max_symbols": "int, default 120"},
        "description": "Show compact code symbols and line ranges without function bodies.",
    },
    {
        "name": "read_symbol",
        "arguments": {"path": "code file", "symbol": "name or qualified name", "include_context": "lines around symbol, default 2"},
        "description": "Read one code symbol body by AST/definition range.",
    },
    {
        "name": "inspect_library_source",
        "arguments": {"target": "Python import path like json.loads or json:loads", "context": "lines around source, default 3", "max_lines": "int, default 160", "include_disassembly": "bool, default false"},
        "description": "Inspect installed Python library source/signature/doc; falls back to bytecode or builtin diagnostics when source is unavailable.",
    },
    {
        "name": "python_sdk_search",
        "arguments": {"query": "natural language or API terms", "limit": "int, default 8", "refresh": "bool, default false", "use_embeddings": "bool, default false", "embedding_model": "optional Ollama embedding model"},
        "description": "Search the installed Python stdlib/API index for current signatures, docstrings, and source locations; optionally rerank with local Ollama embeddings.",
    },
    {
        "name": "python_sdk_refresh",
        "arguments": {"limit": "int, default 5000", "embedding_model": "optional Ollama embedding model", "embedding_host": "optional Ollama host", "embedding_timeout": "seconds, default 120"},
        "description": "Refresh the Python SDK API index from the currently installed Python stdlib and optionally cache local embedding vectors.",
    },
    {
        "name": "repo_index_search",
        "arguments": {"query": "natural query or symbol", "path": "relative path, default .", "limit": "int, default 10"},
        "description": "Search code with compact ranked snippets instead of whole files.",
    },
    {
        "name": "fts_search",
        "arguments": {"query": "terms or phrase", "path": "relative path, default .", "limit": "int, default 20", "refresh": "bool, default false"},
        "description": "Search the local SQLite FTS5 cache across paths, symbols, headings, and snippets.",
    },
    {
        "name": "fts_refresh",
        "arguments": {"path": "relative path, default .", "limit": "int, default 2000"},
        "description": "Refresh the local SQLite FTS5 cache for fast lexical repository queries.",
    },
    {
        "name": "indexed_search",
        "arguments": {"query": "plain terms or regex-like text", "path": "relative path, default .", "limit": "int, default 100"},
        "description": "Search cached line snippets from the persistent repo index without scanning every file.",
    },
    {
        "name": "repo_index_refresh",
        "arguments": {"path": "relative path, default .", "limit": "int, default 1000"},
        "description": "Refresh the persistent repo index for faster future symbol and text searches.",
    },
    {
        "name": "semgrep_scan",
        "arguments": {"pattern": "Semgrep pattern", "path": "relative path, default .", "lang": "optional language", "limit": "int, default 50"},
        "description": "Run Semgrep structurally when semgrep is installed; useful for syntax-aware code search.",
    },
    {
        "name": "ast_search",
        "arguments": {"pattern": "ast-grep pattern", "path": "relative path, default .", "lang": "optional language", "limit": "int, default 50"},
        "description": "Run ast-grep structural search when ast-grep/sg is installed.",
    },
    {
        "name": "lsp_diagnostics",
        "arguments": {"path": "file or directory, default .", "lang": "optional language", "timeout": "seconds, default 60"},
        "description": "Run available language-server-style diagnostics without installing tools.",
    },
    {
        "name": "lsp_definition",
        "arguments": {"path": "file", "line": "1-based line", "column": "1-based column", "limit": "int, default 20"},
        "description": "Find likely definitions for the symbol at a location using available local indexes.",
    },
    {
        "name": "lsp_references",
        "arguments": {"path": "file", "line": "1-based line", "column": "1-based column", "limit": "int, default 40"},
        "description": "Find likely references for the symbol at a location using local search.",
    },
    {
        "name": "context_pack",
        "arguments": {"request": "user request or task text", "path": "relative path, default .", "limit": "int, default 8"},
        "description": "Build compact ranked evidence from repo index, tests, imports, symbols, and git state.",
    },
    {
        "name": "systems_lens",
        "arguments": {"request": "user request or task text", "path": "relative path, default .", "evidence": "optional current evidence", "limit": "int, default 8"},
        "description": "Frame complex work with boundary, state, scale, viewpoints, coupling, and validation checks.",
    },
    {
        "name": "find_implementation_target",
        "arguments": {"test_path": "optional test file", "path": "optional test/source path alias", "query": "optional symbol/query", "output": "optional failing test output/traceback", "limit": "int, default 12"},
        "description": "Map tests or tracebacks to likely implementation files and symbols.",
    },
    {
        "name": "diagnose_test_failure",
        "arguments": {"output": "run_test output", "path": "relative path, default .", "limit": "int, default 12"},
        "description": "Group failing tests by likely root cause with expected/actual and likely target files.",
    },
    {
        "name": "test_spec_extract",
        "arguments": {"test_path": "unittest file", "source_path": "optional source file", "limit": "int, default 20"},
        "description": "Extract compact unittest examples from assertions and assertRaises for repair guidance.",
    },
    {
        "name": "implementation_spec",
        "arguments": {"source_path": "Python source file", "test_path": "optional unittest file", "limit": "int, default 40"},
        "description": "Build a compact implementation spec from source signatures, stubs, static risks, and test examples.",
    },
    {
        "name": "run_function_probe",
        "arguments": {"module": "python module", "expressions": "list of Python expressions", "function": "optional function name", "timeout": "seconds, default 30"},
        "description": "Run small Python expressions against a target module/function and return actual values/errors.",
    },
    {
        "name": "call_graph",
        "arguments": {"path": "file or directory", "symbol": "optional symbol", "limit": "int, default 40"},
        "description": "Show static Python callers, callees, imports, and affected tests.",
    },
    {
        "name": "contract_graph",
        "arguments": {"path": "file or directory, default .", "symbol": "optional symbol", "limit": "int, default 40"},
        "description": "Show compact function contracts, callers/callees, return-shape, and purity hints.",
    },
    {
        "name": "verified_function_index",
        "arguments": {"path": "relative path, default .", "limit": "int, default 500"},
        "description": "Build repo-local Python verified-function cards in the SQLite card index.",
    },
    {
        "name": "verified_function_search",
        "arguments": {"query": "behavior/function query", "signature": "optional signature terms", "examples": "optional examples text/list", "path": "relative path, default .", "limit": "int, default 10"},
        "description": "Search repo-local verified/probable/unverified function cards; retrieval is not proof.",
    },
    {
        "name": "verified_function_show",
        "arguments": {"id": "card id or prefix"},
        "description": "Show a verified-function card, evidence, source excerpt, and stale/fresh hash status.",
    },
    {
        "name": "verify_function_contract",
        "arguments": {"path": "Python source file", "symbol": "function/method symbol"},
        "description": "Recheck one function card with purity, contract checks, doc probes, and focused tests.",
    },
    {
        "name": "compose_verified_functions",
        "arguments": {"goal": "desired behavior", "candidates": "list of card ids or search result ids"},
        "description": "Plan glue-code composition from verified function cards without editing files.",
    },
    {
        "name": "promote_verified_function",
        "arguments": {"path": "Python source file", "symbol": "function/method symbol"},
        "description": "Promote one Python function to verified only after static checks plus probes/tests pass.",
    },
    {
        "name": "lint_typecheck",
        "arguments": {"paths": "path or list of paths, default .", "command": "optional lint/type command", "timeout": "seconds, default 120"},
        "description": "Run deterministic syntax/lint/type checks with exact file lines.",
    },
    {
        "name": "discover_validators",
        "arguments": {"path": "relative path, default .", "limit": "int, default 12"},
        "description": "Detect focused test/lint/typecheck commands for Python, JS/TS, Go, Rust, Java, and C/C++ projects.",
    },
    {
        "name": "diagnose_dependency_error",
        "arguments": {"output": "error output", "path": "relative path, default ."},
        "description": "Classify missing dependency/import/command/path failures and suggest safe next steps.",
    },
    {
        "name": "contract_check",
        "arguments": {"changed_files": "list of changed source files", "changed_symbols": "optional list of symbols", "limit": "int, default 80"},
        "description": "Check changed Python functions for arity, return-shape, async/yield, and annotation contract issues.",
    },
    {
        "name": "select_tests",
        "arguments": {"changed_files": "list of changed source files", "changed_symbols": "optional list of symbols", "limit": "int, default 8"},
        "description": "Select focused test commands for changed files using imports, names, and symbols.",
    },
    {
        "name": "replace_symbol",
        "arguments": {"path": "code file", "symbol": "name or qualified name", "content": "replacement symbol source"},
        "description": "Replace one function/class/method by symbol range. Python replacements must keep the full file syntactically valid.",
    },
    {
        "name": "replace_symbols",
        "arguments": {"path": "code file", "replacements": "list of {symbol, content} objects"},
        "description": "Replace multiple functions/classes/methods in one syntactically checked edit.",
    },
    {
        "name": "write_file",
        "arguments": {"path": "relative path", "content": "full new file content"},
        "description": "Create or fully replace a file.",
    },
    {
        "name": "replace_in_file",
        "arguments": {
            "path": "relative path",
            "old": "text to replace",
            "new": "replacement text",
            "replace_all": "bool, default false",
            "match_whole_word": "bool, default false",
        },
        "description": "Replace text inside an existing file. Prefer unique snippets; use match_whole_word for standalone words.",
    },
    {
        "name": "apply_structured_edit",
        "arguments": {"operation": "JSON operation: rename_symbol, replace_symbol, add_import, move_symbol, delete_symbol"},
        "description": "Apply a mechanical code edit from an intent JSON, then syntax-check and return diff.",
    },
    {
        "name": "edit_intent",
        "arguments": {"path": "relative path", "intent": "rename|replace_text|replace_symbol|replace_body|change_signature|add_import", "target": "old text/symbol", "replacement": "new text/source", "scope": "file|project, default file", "apply": "bool, default true"},
        "description": "Route an edit intent to the safest low-level edit tool and avoid common symbol/text mixups.",
    },
    {
        "name": "generate_tests_from_spec",
        "arguments": {"target_symbol": "symbol under test", "behavior": "expected behavior", "test_path": "optional output test path", "apply": "bool, default false"},
        "description": "Generate a compact pytest test patch from a spec; preview by default, apply only when requested.",
    },
    {
        "name": "browser_smoke",
        "arguments": {"url": "URL to open", "actions": "optional action list", "wait_for": "optional selector/text", "viewport": "desktop|mobile, default desktop", "screenshot": "bool, default false"},
        "description": "Run a Playwright smoke check for browser/UI tasks when Playwright is installed.",
    },
    {
        "name": "security_scan",
        "arguments": {"path": "relative path, default .", "scanners": "auto or scanner list", "limit": "int, default 80", "timeout": "seconds, default 120"},
        "description": "Run available local security/dependency scanners only when explicitly requested.",
    },
    {
        "name": "mcp_list_tools",
        "arguments": {"server": "optional configured MCP server name", "timeout": "seconds, default 15"},
        "description": "List tools exposed by configured MCP servers.",
    },
    {
        "name": "mcp_call",
        "arguments": {"server": "configured MCP server name", "tool": "tool name", "arguments": "tool arguments object", "timeout": "seconds, default 30"},
        "description": "Call a configured MCP server tool through a guarded stdio JSON-RPC request.",
    },
    {
        "name": "run_shell",
        "arguments": {"command": "shell command", "cwd": "relative path, default .", "timeout": "seconds, default 30"},
        "description": "Run a shell command inside the workspace.",
    },
    {
        "name": "run_test",
        "arguments": {
            "command": "optional test command override; defaults to the configured test command",
            "cwd": "relative path, default .",
            "timeout": "seconds, default 1200",
        },
        "description": "Run the project's test command inside the workspace.",
    },
    {
        "name": "git_status",
        "arguments": {"path": "optional relative path filter"},
        "description": "Show git status for the workspace or a specific path.",
    },
    {
        "name": "git_diff",
        "arguments": {"path": "optional relative path filter", "cached": "bool, default false", "context": "int, default 3"},
        "description": "Show a git diff for working tree or staged changes. Leave cached unset for working-tree changes; use cached=true only when the user explicitly asks for staged diff.",
    },
    {
        "name": "git_branch",
        "arguments": {"path": "optional relative path inside the repo", "all_branches": "bool, default false"},
        "description": "List git branches for the active repo, optionally including remotes.",
    },
    {
        "name": "git_log",
        "arguments": {"path": "optional relative path inside the repo", "max_count": "int, default 10", "oneline": "bool, default true"},
        "description": "Show recent git commits for the active repo or path.",
    },
    {
        "name": "git_commit",
        "arguments": {"message": "commit message", "add_all": "bool, default true"},
        "description": "Create a git commit for the current workspace changes.",
    },
    {
        "name": "run_agent",
        "arguments": {
            "prompt": "subtask prompt for the child agent",
            "model": "optional model override",
            "approval_mode": "optional ask|auto|read-only",
            "max_tool_rounds": "optional int, default inherited",
        },
        "description": "Start a nested agent for a scoped subtask and return its final answer.",
    },
]


def format_tool_help(tool_names: Iterable[str] | None = None) -> str:
    lines: list[str] = []
    allowed = set(tool_names) if tool_names is not None else None
    for tool in TOOL_DESCRIPTIONS:
        if allowed is not None and tool["name"] not in allowed:
            continue
        arg_text = ", ".join(f"{key}: {value}" for key, value in tool["arguments"].items())
        lines.append(f"- {tool['name']}({arg_text}) -> {tool['description']}")
    return "\n".join(lines)


def format_compact_tool_help(tool_names: Iterable[str] | None = None) -> str:
    signatures = {
        "todo_read": "todo_read()",
        "todo_write": 'todo_write(items=[{"content":"inspect","status":"pending"}])',
        "list_files": "list_files(path='.',depth=4,limit=200)",
        "read_file": "read_file(path,start=1,end=200)",
        "search": "search(query,path='.',limit=100)",
        "file_search": "file_search(query,path='.',limit=100)",
        "fd_search": "fd_search(query,path='.',limit=100,kind='any')",
        "file_index_refresh": "file_index_refresh(path='.',limit=50000)",
        "everything_search": "everything_search(query,path='.',limit=100)",
        "search_symbols": "search_symbols(query,path='.',limit=50)",
        "code_outline": "code_outline(path,max_symbols=120)",
        "read_symbol": "read_symbol(path,symbol,context=2)",
        "inspect_library_source": "inspect_library_source(target,context=3,max_lines=160,include_disassembly=false)",
        "python_sdk_search": "python_sdk_search(query,limit=8,refresh=false,use_embeddings=false,embedding_model?)",
        "python_sdk_refresh": "python_sdk_refresh(limit=5000,embedding_model?,embedding_host?,embedding_timeout=120)",
        "repo_index_search": "repo_index_search(query,path='.',limit=10)",
        "fts_search": "fts_search(query,path='.',limit=20,refresh=false)",
        "fts_refresh": "fts_refresh(path='.',limit=2000)",
        "indexed_search": "indexed_search(query,path='.',limit=100)",
        "repo_index_refresh": "repo_index_refresh(path='.',limit=1000)",
        "semgrep_scan": "semgrep_scan(pattern,path='.',lang?,limit=50)",
        "ast_search": "ast_search(pattern,path='.',lang?,limit=50)",
        "lsp_diagnostics": "lsp_diagnostics(path='.',lang?,timeout=60)",
        "lsp_definition": "lsp_definition(path,line,column,limit=20)",
        "lsp_references": "lsp_references(path,line,column,limit=40)",
        "context_pack": "context_pack(request,path='.',limit=8)",
        "systems_lens": "systems_lens(request,path='.',evidence?,limit=8)",
        "find_implementation_target": "find_implementation_target(test_path?/path?,query?,output?,limit=12)",
        "diagnose_test_failure": "diagnose_test_failure(output,path='.',limit=12)",
        "test_spec_extract": "test_spec_extract(test_path,source_path?,limit=20)",
        "implementation_spec": "implementation_spec(source_path,test_path?,limit=40)",
        "run_function_probe": "run_function_probe(module,expressions,function?,timeout=30)",
        "call_graph": "call_graph(path,symbol?,limit=40)",
        "contract_graph": "contract_graph(path='.',symbol?,limit=40)",
        "verified_function_index": "verified_function_index(path='.',limit=500)",
        "verified_function_search": "verified_function_search(query,signature?,examples?,path='.',limit=10)",
        "verified_function_show": "verified_function_show(id)",
        "verify_function_contract": "verify_function_contract(path,symbol)",
        "compose_verified_functions": "compose_verified_functions(goal,candidates)",
        "promote_verified_function": "promote_verified_function(path,symbol)",
        "lint_typecheck": "lint_typecheck(paths='.',command?,timeout=120)",
        "discover_validators": "discover_validators(path='.',limit=12)",
        "diagnose_dependency_error": "diagnose_dependency_error(output,path='.')",
        "contract_check": "contract_check(changed_files,changed_symbols?,limit=80)",
        "select_tests": "select_tests(changed_files,changed_symbols?,limit=8)",
        "replace_symbol": "replace_symbol(path,symbol,content)",
        "replace_symbols": 'replace_symbols(path,replacements=[{"symbol":"f","content":"def f():\\n    return ..."}])',
        "write_file": "write_file(path,content)",
        "replace_in_file": "replace_in_file(path,old,new,all=false,whole_word=false)",
        "apply_structured_edit": 'apply_structured_edit(operation={"op":"rename_symbol|replace_function_body|change_signature|...","path":"a.py"})',
        "edit_intent": "edit_intent(path,intent=rename|replace_text|replace_symbol|replace_body|change_signature|add_import,target?,replacement?,scope='file',apply=true)",
        "generate_tests_from_spec": "generate_tests_from_spec(target_symbol,behavior,test_path?,apply=false)",
        "browser_smoke": "browser_smoke(url,actions=[],wait_for?,viewport='desktop',screenshot=false)",
        "security_scan": "security_scan(path='.',scanners='auto',limit=80)",
        "mcp_list_tools": "mcp_list_tools(server?,timeout=15)",
        "mcp_call": "mcp_call(server,tool,arguments={},timeout=30)",
        "run_shell": "run_shell(command,cwd='.',timeout=30)",
        "run_test": "run_test(command?,cwd='.',timeout=1200)",
        "git_status": "git_status(path?)",
        "git_diff": "git_diff(path?,cached=false,context=3)",
        "git_branch": "git_branch(path?,all_branches=false)",
        "git_log": "git_log(path?,max_count=10,oneline=true)",
        "git_commit": "git_commit(message,add_all=true)",
        "run_agent": "run_agent(prompt,model?,approval?,rounds?)",
    }
    allowed = set(tool_names) if tool_names is not None else None
    return "\n".join(signatures[tool["name"]] for tool in TOOL_DESCRIPTIONS if allowed is None or tool["name"] in allowed)
