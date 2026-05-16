from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ollama_code.tools import TOOL_DESCRIPTIONS


@dataclass
class AgentResult:
    message: str
    rounds: int
    completed: bool = True


@dataclass(frozen=True)
class ExactFileWriteSpec:
    path: str
    line: str


@dataclass(frozen=True)
class FinalRewriteOutcome:
    accepted_message: str | None = None
    retry_decision: dict[str, Any] | None = None
    rejected_message: str | None = None


@dataclass(frozen=True)
class TargetLineReadSpec:
    path: str
    start: int
    end: int
    line: int


@dataclass(frozen=True)
class SymbolReadSpec:
    path: str
    symbol: str


@dataclass(frozen=True)
class MechanicalToolSpec:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class MechanicalToolOccurrence:
    start: int
    end: int
    spec: MechanicalToolSpec
    fragment: str


KNOWN_TOOL_NAMES = {tool["name"] for tool in TOOL_DESCRIPTIONS}
APPROVAL_RANK = {"read-only": 0, "ask": 1, "auto": 2}
VERIFICATION_HISTORY_LIMIT = 1
VERIFICATION_CONTENT_LIMIT = 2200
PRIMARY_CONTEXT_RECENT_MESSAGE_LIMIT = 12
PRIMARY_CONTEXT_CONTENT_LIMIT = 900
PRIMARY_CURRENT_REQUEST_LIMIT = 3600
MAX_VERIFICATION_RETRIES = 2
MAX_VERIFICATION_REWRITE_ATTEMPTS = 1
MAX_ASSUMPTION_AUDIT_RETRIES = 2
MAX_RECONCILIATION_RETRIES = 2
SPEC_GUIDED_REPAIR_CANDIDATE_TIMEOUT = 150
SPEC_GUIDED_REPAIR_MAX_ATTEMPTS = 3
PREEMPTIVE_SPEC_GUIDED_SYNTHESIS_TOOL_NAMES = (
    "synthesize_simple_expression_candidate",
    "synthesize_string_normalizer_candidate",
    "synthesize_sequence_utilities_candidate",
)
SPEC_GUIDED_SYNTHESIS_TOOL_NAMES = (
    "synthesize_bowling_game_candidate",
    "synthesize_discounted_set_pricing_candidate",
    "synthesize_countdown_song_candidate",
    "synthesize_affine_substitution_candidate",
    "synthesize_noarg_literal_candidate",
    "synthesize_proverb_chain_candidate",
    "synthesize_typed_graph_dsl_candidate",
    "synthesize_parent_record_tree_candidate",
    "synthesize_domino_chain_candidate",
    "synthesize_food_chain_song_candidate",
    "synthesize_grep_filter_candidate",
    "synthesize_bucket_measure_candidate",
    "synthesize_reactive_cells_candidate",
    "synthesize_hangman_state_candidate",
    "synthesize_rest_api_debt_candidate",
    "synthesize_forth_interpreter_candidate",
    "synthesize_sgf_tree_parser_candidate",
    "synthesize_poker_ranking_candidate",
    "synthesize_metered_io_candidate",
    "synthesize_tree_pov_candidate",
    "synthesize_binary_zipper_candidate",
    "synthesize_go_territory_candidate",
    "synthesize_hex_connect_candidate",
    "synthesize_word_arithmetic_candidate",
    "synthesize_prefix_rotation_candidate",
    "synthesize_text_matrix_transpose_candidate",
    "synthesize_string_normalizer_class_candidate",
    "synthesize_grouped_roster_candidate",
    "synthesize_vlq_candidate",
    "synthesize_cyclic_interval_scale_candidate",
    "synthesize_unique_regex_identifier_candidate",
    "synthesize_node_collection_candidate",
    "synthesize_relative_import_candidate",
)
AUDIT_LIST_ITEM_LIMIT = 3
AUDIT_TEXT_ITEM_LIMIT = 140
CANDIDATE_CLAIM_LIMIT = 5
CANDIDATE_CLAIM_TEXT_LIMIT = 180
VERIFICATION_EVIDENCE_LIMIT = 3
VERIFICATION_EVIDENCE_TEXT_LIMIT = 150
QUESTION_PLANNER_EVIDENCE_LIMIT = 5
QUESTION_PLANNER_MAX_QUESTIONS = 3
TODO_TOOL_NAMES = {"todo_read", "todo_write"}
MUTATING_TOOL_NAMES = {"write_file", "replace_symbol", "replace_symbols", "replace_in_file", "apply_structured_edit", "edit_intent", "git_commit"}
RETRY_PRONE_MUTATING_TOOL_NAMES = {"edit_intent", "replace_symbol", "replace_symbols", "replace_in_file", "apply_structured_edit"}
VERIFIED_FUNCTION_TOOL_NAMES = {"verified_function_index", "verified_function_search", "verified_function_show", "verify_function_contract", "compose_verified_functions", "promote_verified_function"}
PYTHON_SDK_TOOL_NAMES = {"python_sdk_search", "python_sdk_refresh"}
READ_ONLY_CACHEABLE_TOOL_NAMES = {
    "list_files",
    "read_file",
    "search",
    "file_search",
    "fd_search",
    "file_index_refresh",
    "everything_search",
    "search_symbols",
    "code_outline",
    "read_symbol",
    "inspect_library_source",
    "python_sdk_search",
    "python_sdk_refresh",
    "repo_index_search",
    "fts_search",
    "fts_refresh",
    "indexed_search",
    "repo_index_refresh",
    "semgrep_scan",
    "ast_search",
    "lsp_diagnostics",
    "lsp_definition",
    "lsp_references",
    "context_pack",
    "systems_lens",
    "find_implementation_target",
    "diagnose_test_failure",
    "implementation_spec",
    "diagnose_dependency_error",
    "call_graph",
    "contract_graph",
    "verified_function_search",
    "verified_function_show",
    "compose_verified_functions",
    "lint_typecheck",
    "contract_check",
    "select_tests",
    "git_status",
    "git_diff",
}
READ_ONLY_WORKSPACE_TOOL_NAMES = {"list_files", "read_file", "search", "file_search", "fd_search", "file_index_refresh", "everything_search", "search_symbols", "code_outline", "read_symbol", "inspect_library_source", "python_sdk_search", "python_sdk_refresh", "repo_index_search", "fts_search", "fts_refresh", "indexed_search", "repo_index_refresh", "semgrep_scan", "ast_search", "lsp_diagnostics", "lsp_definition", "lsp_references", "context_pack", "systems_lens", "find_implementation_target", "implementation_spec", "diagnose_dependency_error", "call_graph", "contract_graph", "verified_function_search", "verified_function_show", "compose_verified_functions", "discover_validators", "mcp_list_tools"}
CORE_READ_ONLY_WORKSPACE_TOOL_NAMES = {"list_files", "read_file", "search", "file_search", "fd_search", "search_symbols", "code_outline", "read_symbol", "inspect_library_source", "python_sdk_search", "repo_index_search", "fts_search", "indexed_search", "find_implementation_target", "diagnose_dependency_error"}
INDEX_REFRESH_TOOL_NAMES = {"file_index_refresh", "fts_refresh", "repo_index_refresh", "verified_function_index", "python_sdk_refresh"}
STRUCTURAL_SEARCH_TOOL_NAMES = {"semgrep_scan", "ast_search"}
LSP_TOOL_NAMES = {"lsp_diagnostics", "lsp_definition", "lsp_references"}
GRAPH_TOOL_NAMES = {"call_graph", "contract_graph"}
EDIT_TOOL_NAMES = {"edit_intent", "write_file", "replace_symbol", "replace_symbols", "replace_in_file", "apply_structured_edit"}
LOW_LEVEL_EDIT_TOOL_NAMES = {"write_file", "replace_symbol", "replace_symbols", "replace_in_file", "apply_structured_edit"}
TEST_TOOL_NAMES = {"run_test", "diagnose_test_failure", "test_spec_extract", "implementation_spec", "diagnose_dependency_error", "find_implementation_target", "run_function_probe", "lint_typecheck", "contract_check", "select_tests", "discover_validators", "generate_tests_from_spec"}
SHELL_TOOL_NAMES = {"run_shell", "run_function_probe"}
GIT_TOOL_NAMES = {"git_status", "git_diff", "git_branch", "git_log", "git_commit"}
AGENT_TOOL_NAMES = {"run_agent"}
CONTEXT_GATHERING_TOOL_NAMES = {"list_files", "read_file", "search", "file_search", "fd_search", "file_index_refresh", "everything_search", "search_symbols", "code_outline", "read_symbol", "inspect_library_source", "python_sdk_search", "python_sdk_refresh", "repo_index_search", "fts_search", "fts_refresh", "indexed_search", "repo_index_refresh", "semgrep_scan", "ast_search", "lsp_diagnostics", "lsp_definition", "lsp_references", "context_pack", "systems_lens", "contract_graph", "verified_function_index", "verified_function_search", "verified_function_show", "compose_verified_functions", "discover_validators", "test_spec_extract", "implementation_spec", "mcp_list_tools"}
BROAD_CONTEXT_GATHERING_TOOL_NAMES = {
    "list_files",
    "read_file",
    "search",
    "file_search",
    "fd_search",
    "file_index_refresh",
    "everything_search",
    "repo_index_search",
    "fts_search",
    "fts_refresh",
    "indexed_search",
    "repo_index_refresh",
    "semgrep_scan",
    "ast_search",
    "discover_validators",
}
GROUNDING_EVIDENCE_TOOL_NAMES = {"read_file", "file_search", "fd_search", "everything_search", "read_symbol", "inspect_library_source", "python_sdk_search", "context_pack", "repo_index_search", "fts_search", "indexed_search", "semgrep_scan", "ast_search", "lsp_diagnostics", "lsp_definition", "lsp_references", "find_implementation_target", "diagnose_test_failure", "implementation_spec", "diagnose_dependency_error", "contract_graph", "verified_function_search", "verified_function_show", "compose_verified_functions"}
VALIDATION_TOOL_NAMES = {"run_test", "run_function_probe", "lint_typecheck", "contract_check", "verify_function_contract", "select_tests", "discover_validators", "diagnose_dependency_error", "lsp_diagnostics"}
RISKY_VERIFICATION_TOOL_NAMES = {"search", "git_status", "git_diff", "run_shell", "run_test", "run_agent"}
CODE_EDIT_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".kts"}
MODEL_TOOL_RESULT_LIMITS = {
    "list_files": 500,
    "read_file": 1000,
    "search": 700,
    "file_search": 700,
    "fd_search": 700,
    "file_index_refresh": 400,
    "everything_search": 700,
    "search_symbols": 700,
    "code_outline": 900,
    "read_symbol": 1100,
    "inspect_library_source": 1400,
    "python_sdk_search": 1400,
    "python_sdk_refresh": 500,
    "repo_index_search": 900,
    "fts_search": 900,
    "fts_refresh": 500,
    "indexed_search": 900,
    "repo_index_refresh": 500,
    "semgrep_scan": 900,
    "ast_search": 900,
    "lsp_diagnostics": 900,
    "lsp_definition": 700,
    "lsp_references": 900,
    "context_pack": 900,
    "systems_lens": 900,
    "find_implementation_target": 800,
    "diagnose_test_failure": 900,
    "implementation_spec": 1300,
    "run_function_probe": 700,
    "call_graph": 900,
    "contract_graph": 1000,
    "verified_function_index": 500,
    "verified_function_search": 1200,
    "verified_function_show": 1400,
    "verify_function_contract": 1400,
    "compose_verified_functions": 1200,
    "promote_verified_function": 1200,
    "lint_typecheck": 800,
    "discover_validators": 800,
    "diagnose_dependency_error": 800,
    "contract_check": 900,
    "select_tests": 800,
    "edit_intent": 1000,
    "replace_symbol": 900,
    "replace_symbols": 1000,
    "apply_structured_edit": 1000,
    "generate_tests_from_spec": 1000,
    "browser_smoke": 900,
    "security_scan": 900,
    "mcp_list_tools": 900,
    "mcp_call": 900,
    "git_status": 700,
    "git_diff": 900,
    "run_shell": 700,
    "run_test": 700,
    "run_agent": 900,
    "todo_read": 500,
    "todo_write": 700,
}
MODEL_TOOL_DIFF_LIMIT = 700
VERIFICATION_TOOL_RESULT_LIMIT = 450
VERIFICATION_TOOL_DIFF_LIMIT = 450
