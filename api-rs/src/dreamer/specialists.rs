//! Specialist prompt builders + the agentic `run()` orchestration — port of
//! `BaseSpecialist` / `DeductionSpecialist` / `InductionSpecialist`
//! (`src/dreamer/specialists.py`). The prompt builders are pure deterministic
//! strings; [`run_specialist`] wires the preflight, prompts, tool executor, and
//! the (already-ported) `execute_tool_loop` into one specialist run.

use std::time::Instant;

use serde_json::{Value, json};
use sqlx::PgPool;

use super::executor::{DreamerToolExecutor, DreamerToolMetrics};
use super::tools;
use crate::db;
use crate::dialectic::{Embedder, ToolContext};
use crate::llm::executor::HonchoCaller;
use crate::llm::http::LlmHttp;
use crate::llm::tool_loop::execute_tool_loop;
use crate::llm::{ModelConfig, credentials::TransportApiKeys};
use crate::telemetry::Emitter;
use crate::telemetry::events::DreamSpecialistEvent;
use chrono::Utc;

/// Deduction's `peer_card_update_instruction` (specialists.py:442).
const DEDUCTION_PEER_CARD_INSTRUCTION: &str = "Update this with `update_peer_card` only for stable identity markers. See the PEER CARD section in the system prompt for the allowed entry kinds and rules.";

/// Port of `BaseSpecialist._build_target_observee_context` (upstream #806): the
/// `Target observee:` block prepended to every specialist user prompt so the
/// system prompts stay peer-agnostic (byte-stable prefix for prompt caching).
fn build_target_observee_context(observed: &str) -> String {
    format!(
        "Target observee:\n{observed}\n\nThe target observee is the peer identified above. When created observations need to name this subject, use the exact observee id above, not the phrase \"the target observee\".\n\n"
    )
}

/// Port of `BaseSpecialist._build_peer_card_context` (specialists.py:120). Empty
/// string when the peer card is absent/empty; otherwise a `## CURRENT PEER CARD`
/// section listing the facts, followed by the specialist's update instruction.
fn build_peer_card_context(peer_card: Option<&[String]>, instruction: &str) -> String {
    let facts = match peer_card {
        Some(card) if !card.is_empty() => card,
        _ => return String::new(),
    };
    let facts_str = facts
        .iter()
        .map(|fact| format!("- {fact}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "\n## CURRENT PEER CARD\n\n{facts_str}\n\n{instruction}\nIf you update it, send the full deduplicated list and remove stale entries.\n\n"
    )
}

/// Port of `DeductionSpecialist.build_system_prompt` (specialists.py:478). When
/// `peer_card_enabled` is true the large `## PEER CARD (REQUIRED)` section is
/// spliced in at the `{peer_card_section}` marker; otherwise that marker is empty.
///
/// Post-#806 the prompt never interpolates the peer id — it refers to "the
/// target observee" throughout, and the id arrives via the user prompt's
/// `Target observee:` block (see [`build_target_observee_context`]) so the
/// system prompt is byte-stable across peers for prompt caching. `observed` is
/// kept in the signature to mirror Python (`_ = observed`).
pub fn deduction_system_prompt(observed: &str, peer_card_enabled: bool) -> String {
    let _ = observed;
    let peer_card_section = if peer_card_enabled {
        "\n\n## PEER CARD (REQUIRED)\n\nThe peer card is the target observee's identity store: stable identity markers that distinguish this entity from others and persist across interactions. Behavior, tendencies, transient state, and episodic facts belong in observations, not on the peer card.\n\nA peer can be anything with identity that changes over time — a human, an agent, a codebase, a team, an organization. Do not assume the target observee is human. Do not require any field; empty is the correct output when evidence is absent.\n\n### Allowed entry kinds\n\nEach entry must start with one of these four prefixes (exact case, followed by a space):\n\n- `IDENTITY: ...` — canonical name, kind, aliases, IDs\n  - `IDENTITY: Name: Alice`\n  - `IDENTITY: Kind: Python monorepo`\n  - `IDENTITY: Version: 4.2`\n  - `IDENTITY: Aliases: alice@example.com`\n- `ATTRIBUTE: ...` — stable durable property of the entity (including explicitly stated standing preferences)\n  - `ATTRIBUTE: Location: NYC`\n  - `ATTRIBUTE: Language: Python`\n  - `ATTRIBUTE: Prefers tea`\n  - `ATTRIBUTE: Charter: ship Honcho infrastructure`\n- `RELATIONSHIP: ...` — durable link to another entity\n  - `RELATIONSHIP: Spouse: Bob`\n  - `RELATIONSHIP: Maintainer: vineeth`\n  - `RELATIONSHIP: Members: vineeth, rajat`\n- `INSTRUCTION: ...` — standing rule of engagement that the target observee has explicitly stated (do/don't for the observer). Only when explicit; never inferred from behavior.\n  - `INSTRUCTION: Call me Vee`\n  - `INSTRUCTION: Never push to main without review`\n\n### Rules\n\n1. **Stable.** If the value plausibly changes within six months absent a deliberate announcement, it does not belong on the card. Prefer leaving the card empty over filling it with volatile content.\n2. **Subject is the target observee.** Every entry must be a fact about the target observee, not about another participant in the session. Never write facts about co-occurring peers into the card, no matter how frequently they appear in the messages.\n3. **Evidence-grounded.** Only write what the target observee has explicitly stated, or what another participant has explicitly stated about the target observee with the target observee's assent. No \"general knowledge\" inferences (`\"co-founder\"` does not imply an age; mentioning a colleague does not imply a family relationship).\n4. **Type-agnostic.** The target observee may not be human. Do not require name/age/location/family/occupation fields.\n5. **No behavioral content.** TRAITs, behavioral tendencies, patterns, and inferred preferences belong in observations, not on the peer card. Do not write `TRAIT:` entries or behavioral `PREFERENCE:` entries — they will be rejected.\n6. **No evidence bundles.** Each entry is one concise fact. No `e.g.` clauses, no parenthetical example lists, no semicolon-separated value dumps.\n\n### Migrating an existing peer card\n\nThe CURRENT PEER CARD shown in the user message may contain entries from an older format that do not start with an allowed prefix (e.g. `Name: Alice`, `Lives in NYC`, `TRAIT: Analytical`, `PREFERENCE: Detailed explanations`). When you call `update_peer_card`, you are responsible for re-emitting the entries you want to keep — entries you omit are dropped, and entries without an allowed prefix are silently rejected.\n\nFor each legacy entry:\n\n- If it is still a valid identity marker, re-emit it under the correct prefix and keep the original content where reasonable. Examples:\n  - `Name: Alice` → `IDENTITY: Name: Alice`\n  - `Lives in NYC` → `ATTRIBUTE: Location: NYC`\n  - `Works at Google` → `ATTRIBUTE: Employer: Google`\n  - `INSTRUCTION: Call me Vee` → keep as is (already correctly prefixed)\n- Drop entries that violate the rules above: behavioral `TRAIT:` lines, inferred behavioral `PREFERENCE:` lines, one-off events, transient state, evidence bundles. Do not re-prefix them — they are not identity markers.\n\nWhen in doubt about a specific legacy entry, prefer migrating it (so valid info isn't lost) over dropping it. Splitting one dense legacy entry into multiple correctly-prefixed entries is fine and encouraged (e.g. a semicolon-separated `Tech Stack:` dump can become several `ATTRIBUTE:` lines, one per durable tool/platform).\n\nCall `update_peer_card` with the complete deduplicated list when there is a durable identity update to record, or when the existing card needs migration. Entries that do not start with one of the four allowed prefixes will be rejected. Keep concise (max 40 entries)."
    } else {
        ""
    };

    format!(
        "You are a deductive reasoning agent analyzing observations about the target observee.\n\n## YOUR JOB\n\nCreate deductive observations by finding logical implications in what's already known. Think like a detective connecting evidence.\n\n## PHASE 1: DISCOVERY\n\nExplore what's actually in memory. Use these tools freely:\n- `get_recent_observations` - See what's been learned recently\n- `search_memory` - Search for specific topics\n- `search_messages` - See actual conversation content\n\nSpend a few tool calls understanding the landscape before creating anything.\n\n## PHASE 2: ACTION\n\nOnce you understand what's there, create observations and clean up:\n\n### Knowledge Updates (HIGH PRIORITY)\nWhen the same fact has different values at different times:\n- \"meeting Tuesday\" [old] → \"meeting moved to Thursday\" [new]\n- Create a deductive update observation\n- DELETE the outdated observation immediately\n\n### Logical Implications\nExtract implicit information:\n- \"works as SWE at Google\" → \"has software engineering skills\", \"employed in tech\"\n- \"has kids ages 5 and 8\" → \"is a parent\", \"has school-age children\"\n\n### Contradictions\nWhen statements can't both be true (not just updates), flag them:\n- \"I love coffee\" vs \"I hate coffee\" → contradiction observation\n{peer_card_section}\n\n## CREATING OBSERVATIONS\n\nUse `create_observations_deductive`.\n\n```json\n{{\n  \"observations\": [{{\n    \"content\": \"The logical conclusion\",\n    \"source_ids\": [\"id1\", \"id2\"],\n    \"premises\": [\"premise 1 text\", \"premise 2 text\"]\n  }}]\n}}\n```\n\n## RULES\n\n1. Don't explain your reasoning - just call tools\n2. Create observations based on what you ACTUALLY FIND, not what you expect\n3. Always include source_ids linking to the observations you're synthesizing\n4. Empty or missing source_ids will be rejected\n5. Delete outdated observations - don't leave duplicates\n6. Quality over quantity - fewer good deductions beat many weak ones"
    )
}

/// Port of `DeductionSpecialist.build_user_prompt` (specialists.py:598). Prepends
/// the `Target observee:` block (#806), then the peer-card context; only the
/// first 5 hints are used.
pub fn deduction_user_prompt(
    observed: &str,
    hints: Option<&[String]>,
    peer_card: Option<&[String]>,
) -> String {
    let target_observee_context = build_target_observee_context(observed);
    let peer_card_context = build_peer_card_context(peer_card, DEDUCTION_PEER_CARD_INSTRUCTION);

    match hints {
        Some(hints) if !hints.is_empty() => {
            let hints_str = hints
                .iter()
                .take(5)
                .map(|q| format!("- {q}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "{target_observee_context}{peer_card_context}Start by exploring recent observations and messages. These topics may be worth investigating:\n\n{hints_str}\n\nBut follow the evidence - if you find something more interesting, pursue that instead.\n\nBegin with `get_recent_observations` to see what's there."
            )
        }
        _ => format!(
            "{target_observee_context}{peer_card_context}Explore the observation space and create deductive observations.\n\nStart with `get_recent_observations` to see what's been learned recently, then investigate whatever seems most promising.\n\nLook for:\n1. Knowledge updates (same fact, different values over time)\n2. Logical implications that haven't been made explicit\n3. Contradictions that need flagging\n\nGo."
        ),
    }
}

/// Port of `InductionSpecialist.build_system_prompt` (specialists.py:663).
/// `observed` is ignored post-#806 (the id arrives via the user prompt's
/// `Target observee:` block; the system prompt is byte-stable for caching).
///
/// NOTE: upstream #806 dropped the `f` prefix from this Python string but kept
/// the escaped `{{`/`}}` braces, so the JSON example now renders with literal
/// double braces. Reproduced faithfully (plain literal, no `format!`).
pub fn induction_system_prompt(observed: &str) -> String {
    let _ = observed;
    "You are an inductive reasoning agent identifying patterns about the target observee.\n\n## YOUR JOB\n\nCreate inductive observations by finding patterns across multiple observations. Think like a psychologist identifying behavioral tendencies.\n\n## PHASE 1: DISCOVERY\n\nExplore broadly to find patterns. Use these tools:\n- `get_recent_observations` - Recent learnings\n- `search_memory` - Topic-specific search\n- `search_messages` - Actual conversation content\n\nLook at BOTH explicit observations AND deductive ones. Patterns often emerge from synthesizing across both levels.\n\n## PHASE 2: ACTION\n\nCreate inductive observations when you see patterns:\n\n### Behavioral Patterns\n- \"Tends to reschedule meetings when stressed\"\n- \"Makes decisions after consulting with partner\"\n- \"Projects follow: enthusiasm → doubt → completion\"\n\n### Preferences\n- \"Prefers morning meetings\"\n- \"Likes detailed technical explanations\"\n\n### Personality Traits\n- \"Generally optimistic about outcomes\"\n- \"Detail-oriented in planning\"\n\n### Temporal Patterns\n- \"Career goals have remained consistent\"\n- \"Living situation changes frequently\"\n\n## CREATING OBSERVATIONS\n\nUse `create_observations_inductive`.\n\n```json\n{{\n  \"observations\": [{{\n    \"content\": \"The pattern or generalization\",\n    \"source_ids\": [\"id1\", \"id2\", \"id3\"],\n    \"sources\": [\"evidence 1\", \"evidence 2\"],\n    \"pattern_type\": \"tendency\", // preference|behavior|personality|tendency|correlation\n    \"confidence\": \"medium\" // low (2 sources), medium (3-4), high (5+)\n  }}]\n}}\n```\n\n## RULES\n\n1. Minimum 2 source observations required - patterns need evidence\n2. Don't just restate a single fact as a pattern\n3. Confidence based on evidence count: 2=low, 3-4=medium, 5+=high\n4. Look for HOW things change over time, not just static facts\n5. Include source_ids - always link back to evidence\n6. Empty or missing source_ids will be rejected"
        .to_string()
}

/// Port of `InductionSpecialist.build_user_prompt` (specialists.py:729). Prepends
/// the `Target observee:` block (#806); the peer card is never consumed by
/// induction; only the first 5 hints are used.
pub fn induction_user_prompt(observed: &str, hints: Option<&[String]>) -> String {
    let target_observee_context = build_target_observee_context(observed);
    match hints {
        Some(hints) if !hints.is_empty() => {
            let hints_str = hints
                .iter()
                .take(5)
                .map(|q| format!("- {q}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "{target_observee_context}Explore and find patterns. These areas may be worth investigating:\n\n{hints_str}\n\nBut follow the evidence - if you find patterns elsewhere, pursue those.\n\nStart with `get_recent_observations`."
            )
        }
        _ => format!(
            "{target_observee_context}Explore the observation space and identify patterns.\n\nRemember: patterns need 2+ sources. Look for tendencies, preferences, and behavioral regularities.\n\nGo."
        ),
    }
}

/// The two dream specialists (port of `DeductionSpecialist` / `InductionSpecialist`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialistKind {
    Deduction,
    Induction,
}

impl SpecialistKind {
    /// `name` (specialist_type slug).
    pub fn name(self) -> &'static str {
        match self {
            SpecialistKind::Deduction => "deduction",
            SpecialistKind::Induction => "induction",
        }
    }

    /// `can_update_peer_card` — induction never writes the peer card.
    pub fn can_update_peer_card(self) -> bool {
        matches!(self, SpecialistKind::Deduction)
    }

    /// `get_max_tokens` (deduction 8192, induction 8192).
    pub fn max_tokens(self) -> i64 {
        8192
    }

    /// `get_max_iterations` (deduction 12, induction 10).
    pub fn max_iterations(self) -> usize {
        match self {
            SpecialistKind::Deduction => 12,
            SpecialistKind::Induction => 10,
        }
    }

    fn build_system_prompt(self, observed: &str, peer_card_enabled: bool) -> String {
        match self {
            SpecialistKind::Deduction => deduction_system_prompt(observed, peer_card_enabled),
            SpecialistKind::Induction => induction_system_prompt(observed),
        }
    }

    fn build_user_prompt(
        self,
        observed: &str,
        hints: Option<&[String]>,
        peer_card: Option<&[String]>,
    ) -> String {
        match self {
            SpecialistKind::Deduction => deduction_user_prompt(observed, hints, peer_card),
            SpecialistKind::Induction => induction_user_prompt(observed, hints),
        }
    }

    /// `get_tools(peer_card_enabled)` — deduction drops `update_peer_card` when
    /// the peer card is disabled (`PEER_CARD_TOOL_NAMES` filter); induction
    /// ignores the flag.
    fn get_tools(self, peer_card_enabled: bool) -> Vec<Value> {
        match self {
            SpecialistKind::Deduction => {
                let tools = tools::deduction_specialist_tools();
                if peer_card_enabled {
                    tools
                } else {
                    tools
                        .into_iter()
                        .filter(|t| t["name"] != "update_peer_card")
                        .collect()
                }
            }
            SpecialistKind::Induction => tools::induction_specialist_tools(),
        }
    }
}

/// Result of a specialist run (port of `SpecialistResult`), plus the tool-call
/// rollups the orchestrator aggregates into the `DreamRunEvent`.
#[derive(Debug, Clone)]
pub struct SpecialistResult {
    pub run_id: String,
    pub specialist_type: String,
    pub iterations: i64,
    pub tool_calls_count: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub duration_ms: f64,
    pub success: bool,
    pub content: String,
    pub metrics: DreamerToolMetrics,
}

/// Extract the peer-card fact list from `db::get_peer_card`'s `{"peer_card": [...]}`
/// JSON, or `None` when absent/empty.
fn peer_card_facts(card: Option<Value>) -> Option<Vec<String>> {
    let facts: Vec<String> = card
        .as_ref()
        .and_then(|v| v.get("peer_card"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if facts.is_empty() { None } else { Some(facts) }
}

/// Port of `BaseSpecialist.run`: preflight (get-or-create peers + peer-card
/// fetch) → build prompts → run the tool loop with a [`DreamerToolExecutor`] →
/// roll up the metrics → emit a [`DreamSpecialistEvent`] (always, success or
/// failure) and return a [`SpecialistResult`].
///
/// Deviation: on the failure path the rollups are reported as zero (matching
/// Python, which computes them only after a successful loop) even though the
/// executor may have written some observations before the loop errored. The
/// emitted event uses `iteration = 0` like the agent-tool events.
#[allow(clippy::too_many_arguments)]
pub async fn run_specialist<H, E>(
    kind: SpecialistKind,
    pool: &PgPool,
    http: &H,
    keys: TransportApiKeys,
    embedder: &E,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    session_name: Option<&str>,
    hints: Option<&[String]>,
    peer_card_create: bool,
    model_config: ModelConfig,
    parent_run_id: &str,
    emitter: &dyn Emitter,
    honcho_version: Option<String>,
    deduplicate: bool,
) -> Result<SpecialistResult, sqlx::Error>
where
    H: LlmHttp + Sync,
    E: Embedder + Sync,
{
    let run_id = parent_run_id.to_string();
    let start = Instant::now();

    // Preflight: get-or-create observer (+ observed when distinct).
    db::get_or_create_peer(pool, workspace_name, observer, None, None).await?;
    if observer != observed {
        db::get_or_create_peer(pool, workspace_name, observed, None, None).await?;
    }

    let peer_card_enabled = kind.can_update_peer_card() && peer_card_create;
    let current_peer_card = if peer_card_enabled {
        peer_card_facts(db::get_peer_card(pool, workspace_name, observer, observed).await?)
    } else {
        None
    };

    let system_prompt = kind.build_system_prompt(observed, peer_card_enabled);
    let user_prompt = kind.build_user_prompt(observed, hints, current_peer_card.as_deref());
    let messages = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": user_prompt}),
    ];
    let tool_schemas = kind.get_tools(peer_card_enabled);

    let executor = DreamerToolExecutor::new(
        pool,
        ToolContext {
            workspace_name: workspace_name.to_string(),
            observer: observer.to_string(),
            observed: observed.to_string(),
            session_name: session_name.map(str::to_string),
        },
        embedder,
        true, // include_observation_ids (dreamer)
        peer_card_create,
        run_id.clone(),
        kind.name().to_string(),
        "dream".to_string(),
        emitter,
        honcho_version.clone(),
        deduplicate,
    );

    // max_output_tokens override on the ModelConfig wins when positive.
    let effective_max_tokens = model_config
        .max_output_tokens
        .filter(|&max| max > 0)
        .unwrap_or_else(|| kind.max_tokens());
    let caller = HonchoCaller::new(http, keys, model_config, effective_max_tokens);

    let loop_result = execute_tool_loop(
        &caller,
        &executor,
        "",
        Some(&messages),
        &tool_schemas,
        None,
        kind.max_iterations(),
        None,
    )
    .await;

    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    let (result, event) = match loop_result {
        Ok(response) => {
            let metrics = executor.metrics_snapshot();
            let content = response
                .content
                .as_str()
                .map(str::to_string)
                .unwrap_or_default();
            let result = SpecialistResult {
                run_id: run_id.clone(),
                specialist_type: kind.name().to_string(),
                iterations: response.iterations as i64,
                tool_calls_count: response.tool_calls_made.len() as i64,
                input_tokens: response.input_tokens,
                output_tokens: response.output_tokens,
                duration_ms,
                success: true,
                content,
                metrics: metrics.clone(),
            };
            let event = DreamSpecialistEvent {
                timestamp: Utc::now(),
                run_id: run_id.clone(),
                specialist_type: kind.name().to_string(),
                workspace_name: workspace_name.to_string(),
                observer: observer.to_string(),
                observed: observed.to_string(),
                iterations: result.iterations,
                tool_calls_count: result.tool_calls_count,
                input_tokens: result.input_tokens,
                output_tokens: result.output_tokens,
                duration_ms,
                success: true,
                created_observation_count: metrics.created_observation_count,
                deleted_observation_count: metrics.deleted_observation_count,
                created_counts_by_level: metrics.created_counts_by_level,
                deleted_counts_by_level: metrics.deleted_counts_by_level,
                peer_card_updated: metrics.peer_card_updated,
                search_tool_calls_count: metrics.search_tool_calls_count,
                error_class: None,
            };
            (result, event)
        }
        Err(error) => {
            // Failure path: zero rollups (Python computes them post-loop only).
            let result = SpecialistResult {
                run_id: run_id.clone(),
                specialist_type: kind.name().to_string(),
                iterations: 0,
                tool_calls_count: 0,
                input_tokens: 0,
                output_tokens: 0,
                duration_ms,
                success: false,
                content: String::new(),
                metrics: DreamerToolMetrics::default(),
            };
            let event = DreamSpecialistEvent {
                timestamp: Utc::now(),
                run_id: run_id.clone(),
                specialist_type: kind.name().to_string(),
                workspace_name: workspace_name.to_string(),
                observer: observer.to_string(),
                observed: observed.to_string(),
                iterations: 0,
                tool_calls_count: 0,
                input_tokens: 0,
                output_tokens: 0,
                duration_ms,
                success: false,
                created_observation_count: 0,
                deleted_observation_count: 0,
                created_counts_by_level: Default::default(),
                deleted_counts_by_level: Default::default(),
                peer_card_updated: false,
                search_tool_calls_count: 0,
                error_class: Some(error.to_string()),
            };
            (result, event)
        }
    };

    emitter.emit(&event);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hints() -> Vec<String> {
        vec![
            "topic one".into(),
            "topic two".into(),
            "topic three".into(),
            "topic four".into(),
            "topic five".into(),
            "topic six".into(),
        ]
    }

    fn card() -> Vec<String> {
        vec![
            "IDENTITY: Name: Alice".into(),
            "ATTRIBUTE: Location: NYC".into(),
        ]
    }

    #[test]
    fn deduction_system_prompt_with_peer_card() {
        assert_eq!(
            deduction_system_prompt("bob", true),
            include_str!("fixtures/ded_sys_pc.txt")
        );
    }

    #[test]
    fn deduction_system_prompt_without_peer_card() {
        assert_eq!(
            deduction_system_prompt("bob", false),
            include_str!("fixtures/ded_sys_nopc.txt")
        );
    }

    #[test]
    fn deduction_user_prompt_no_hints_no_card() {
        assert_eq!(
            deduction_user_prompt("bob", None, None),
            include_str!("fixtures/ded_user_nohints_nocard.txt")
        );
    }

    #[test]
    fn deduction_user_prompt_with_hints_and_card() {
        assert_eq!(
            deduction_user_prompt("bob", Some(&hints()), Some(&card())),
            include_str!("fixtures/ded_user_hints_card.txt")
        );
    }

    #[test]
    fn induction_system_prompt_golden() {
        assert_eq!(
            induction_system_prompt("bob"),
            include_str!("fixtures/ind_sys.txt")
        );
    }

    #[test]
    fn induction_user_prompt_no_hints_golden() {
        assert_eq!(
            induction_user_prompt("bob", None),
            include_str!("fixtures/ind_user_nohints.txt")
        );
    }

    #[test]
    fn induction_user_prompt_with_hints_golden() {
        assert_eq!(
            induction_user_prompt("bob", Some(&hints())),
            include_str!("fixtures/ind_user_hints.txt")
        );
    }

    #[test]
    fn system_prompts_are_static_across_peers() {
        // The #806 cache-prefix property: system prompts never embed the peer id.
        assert_eq!(
            deduction_system_prompt("alice", true),
            deduction_system_prompt("bob", true)
        );
        assert_eq!(induction_system_prompt("alice"), induction_system_prompt("bob"));
        assert!(
            deduction_user_prompt("bob", None, None).starts_with("Target observee:\nbob\n\n")
        );
        assert!(induction_user_prompt("bob", None).starts_with("Target observee:\nbob\n\n"));
    }
}
