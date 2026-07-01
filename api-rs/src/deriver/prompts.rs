//! Minimal deriver prompt, ported from `src/deriver/prompts.py`.
//!
//! Produces byte-identical output to the Python `minimal_deriver_prompt` and
//! the token-estimate helpers. The Python prompts wrap their bodies in
//! `inspect.cleandoc` (imported as `c`); since the main prompt's body sits at
//! column 0 with the custom-instructions placeholder also at column 0, its
//! `cleandoc` only strips the leading/trailing blank line — so the final body
//! is written here already-stripped, and only the indented
//! custom-instructions section needs an actual [`cleandoc`].
//!
//! The prompt body is static up to the trailing `Target peer:` block (upstream
//! #806 "optimize deriver prompt cache prefixes"): the peer id is NOT
//! interpolated into the rules/examples (which use a literal `alice`), keeping
//! the long prefix byte-stable across peers for provider prompt caching.

use crate::text::cleandoc;
use crate::tokens::estimate_tokens;

/// Return stripped custom instructions, or `None` when absent/blank
/// (`_normalized_custom_instructions`).
fn normalized_custom_instructions(custom_instructions: Option<&str>) -> Option<String> {
    let normalized = custom_instructions?.trim();
    if normalized.is_empty() {
        None
    } else {
        Some(normalized.to_string())
    }
}

/// Render the optional custom-instructions section (`_custom_instructions_section`).
/// Empty string when no instructions; otherwise the cleandoc'd block.
fn custom_instructions_section(custom_instructions: Option<&str>) -> String {
    match normalized_custom_instructions(custom_instructions) {
        None => String::new(),
        Some(normalized) => cleandoc(&format!(
            "\n        CUSTOM INSTRUCTIONS:\n        These instructions apply to the target peer identified below.\n        {normalized}\n        "
        )),
    }
}

/// Generate the minimal prompt for fast observation extraction
/// (`minimal_deriver_prompt`).
pub fn minimal_deriver_prompt(
    peer_id: &str,
    messages: &str,
    custom_instructions: Option<&str>,
) -> String {
    let section = custom_instructions_section(custom_instructions);
    format!(
        r#"Analyze messages to extract **explicit atomic facts** about the target peer.

[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- The target peer is the peer identified below under `Target peer:`.
- A peer can be a human user, AI agent, bot, service, or other actor.
- Use the exact peer id from `Target peer:` in final observations, not the phrase "the target peer".
- Properly attribute observations to the correct subject: if it is about the target peer, use the exact peer id as the subject. If the target peer is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand the target peer.
- Extract ALL observations from the target peer's messages, using others as context.
- Contextualize each observation sufficiently (e.g. "Ann is nervous about the job interview at the pharmacy" not just "Ann is nervous")

EXAMPLES (using `alice` as the target peer id):
- EXPLICIT: "I just had my 25th birthday last Saturday" → "alice is 25 years old", "alice's birthday is June 21st"
- EXPLICIT: "I took my dog for a walk in NYC" → "alice has a dog", "alice lives in NYC"
- EXPLICIT: "alice attended college" + general knowledge → "alice completed high school or equivalent"

{section}

Target peer:
{peer_id}

Messages to analyze:
<messages>
{messages}
</messages>"#
    )
}

/// Estimate the static minimal prompt (no custom instructions). Python memoizes
/// this with `@cache`; the count is cheap enough here to recompute.
pub fn estimate_minimal_deriver_prompt_tokens() -> usize {
    estimate_tokens(&minimal_deriver_prompt("", "", None))
}

/// Estimate prompt tokens including custom instructions when present
/// (`estimate_deriver_prompt_tokens`).
pub fn estimate_deriver_prompt_tokens(custom_instructions: Option<&str>) -> usize {
    match normalized_custom_instructions(custom_instructions) {
        None => estimate_minimal_deriver_prompt_tokens(),
        Some(normalized) => estimate_tokens(&minimal_deriver_prompt("", "", Some(&normalized))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden strings captured from Python `minimal_deriver_prompt(...)` at
    // upstream v3.0.11 (post-#806 form: static prefix + `Target peer:` block).
    const GOLDEN_NO_CI: &str = "Analyze messages to extract **explicit atomic facts** about the target peer.\n\n[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.\n   - Transform statements into one or multiple conclusions\n   - Each conclusion must be self-contained with enough context\n   - Use absolute dates/times when possible (e.g. \"June 26, 2025\" not \"yesterday\")\n\nRULES:\n- The target peer is the peer identified below under `Target peer:`.\n- A peer can be a human user, AI agent, bot, service, or other actor.\n- Use the exact peer id from `Target peer:` in final observations, not the phrase \"the target peer\".\n- Properly attribute observations to the correct subject: if it is about the target peer, use the exact peer id as the subject. If the target peer is referencing someone or something else, make that clear.\n- Observations should make sense on their own. Each observation will be used in the future to better understand the target peer.\n- Extract ALL observations from the target peer's messages, using others as context.\n- Contextualize each observation sufficiently (e.g. \"Ann is nervous about the job interview at the pharmacy\" not just \"Ann is nervous\")\n\nEXAMPLES (using `alice` as the target peer id):\n- EXPLICIT: \"I just had my 25th birthday last Saturday\" → \"alice is 25 years old\", \"alice's birthday is June 21st\"\n- EXPLICIT: \"I took my dog for a walk in NYC\" → \"alice has a dog\", \"alice lives in NYC\"\n- EXPLICIT: \"alice attended college\" + general knowledge → \"alice completed high school or equivalent\"\n\n\n\nTarget peer:\nalice\n\nMessages to analyze:\n<messages>\nhello world\n</messages>";

    const GOLDEN_CI: &str = "Analyze messages to extract **explicit atomic facts** about the target peer.\n\n[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.\n   - Transform statements into one or multiple conclusions\n   - Each conclusion must be self-contained with enough context\n   - Use absolute dates/times when possible (e.g. \"June 26, 2025\" not \"yesterday\")\n\nRULES:\n- The target peer is the peer identified below under `Target peer:`.\n- A peer can be a human user, AI agent, bot, service, or other actor.\n- Use the exact peer id from `Target peer:` in final observations, not the phrase \"the target peer\".\n- Properly attribute observations to the correct subject: if it is about the target peer, use the exact peer id as the subject. If the target peer is referencing someone or something else, make that clear.\n- Observations should make sense on their own. Each observation will be used in the future to better understand the target peer.\n- Extract ALL observations from the target peer's messages, using others as context.\n- Contextualize each observation sufficiently (e.g. \"Ann is nervous about the job interview at the pharmacy\" not just \"Ann is nervous\")\n\nEXAMPLES (using `alice` as the target peer id):\n- EXPLICIT: \"I just had my 25th birthday last Saturday\" → \"alice is 25 years old\", \"alice's birthday is June 21st\"\n- EXPLICIT: \"I took my dog for a walk in NYC\" → \"alice has a dog\", \"alice lives in NYC\"\n- EXPLICIT: \"alice attended college\" + general knowledge → \"alice completed high school or equivalent\"\n\nCUSTOM INSTRUCTIONS:\nThese instructions apply to the target peer identified below.\nbe terse\n\nTarget peer:\nalice\n\nMessages to analyze:\n<messages>\nhello world\n</messages>";

    const GOLDEN_CI_ML: &str = "Analyze messages to extract **explicit atomic facts** about the target peer.\n\n[EXPLICIT] DEFINITION: Facts about the target peer that can be derived directly from their messages.\n   - Transform statements into one or multiple conclusions\n   - Each conclusion must be self-contained with enough context\n   - Use absolute dates/times when possible (e.g. \"June 26, 2025\" not \"yesterday\")\n\nRULES:\n- The target peer is the peer identified below under `Target peer:`.\n- A peer can be a human user, AI agent, bot, service, or other actor.\n- Use the exact peer id from `Target peer:` in final observations, not the phrase \"the target peer\".\n- Properly attribute observations to the correct subject: if it is about the target peer, use the exact peer id as the subject. If the target peer is referencing someone or something else, make that clear.\n- Observations should make sense on their own. Each observation will be used in the future to better understand the target peer.\n- Extract ALL observations from the target peer's messages, using others as context.\n- Contextualize each observation sufficiently (e.g. \"Ann is nervous about the job interview at the pharmacy\" not just \"Ann is nervous\")\n\nEXAMPLES (using `alice` as the target peer id):\n- EXPLICIT: \"I just had my 25th birthday last Saturday\" → \"alice is 25 years old\", \"alice's birthday is June 21st\"\n- EXPLICIT: \"I took my dog for a walk in NYC\" → \"alice has a dog\", \"alice lives in NYC\"\n- EXPLICIT: \"alice attended college\" + general knowledge → \"alice completed high school or equivalent\"\n\n        CUSTOM INSTRUCTIONS:\n        These instructions apply to the target peer identified below.\n        line1\nline2\n        \n\nTarget peer:\nalice\n\nMessages to analyze:\n<messages>\nmsgs\n</messages>";

    #[test]
    fn prompt_without_custom_instructions_matches_golden() {
        assert_eq!(
            minimal_deriver_prompt("alice", "hello world", None),
            GOLDEN_NO_CI
        );
    }

    #[test]
    fn prompt_with_custom_instructions_matches_golden() {
        // Surrounding whitespace is stripped by _normalized_custom_instructions.
        assert_eq!(
            minimal_deriver_prompt("alice", "hello world", Some("  be terse  ")),
            GOLDEN_CI
        );
    }

    #[test]
    fn prompt_with_multiline_custom_instructions_matches_golden() {
        assert_eq!(
            minimal_deriver_prompt("alice", "msgs", Some("line1\nline2")),
            GOLDEN_CI_ML
        );
    }

    #[test]
    fn blank_custom_instructions_render_as_no_section() {
        assert_eq!(
            minimal_deriver_prompt("alice", "hello world", Some("   ")),
            GOLDEN_NO_CI
        );
    }

    #[test]
    fn prefix_is_static_across_peers() {
        // The #806 cache-prefix property: everything before the custom
        // instructions / `Target peer:` block is byte-identical for any peer id
        // (the examples keep the literal `alice`).
        let alice = minimal_deriver_prompt("alice", "hi", None);
        let bob = minimal_deriver_prompt("bob", "hi", None);
        let split = "\n\n\n\nTarget peer:\n";
        let alice_prefix = alice.split(split).next().unwrap();
        let bob_prefix = bob.split(split).next().unwrap();
        assert_eq!(alice_prefix, bob_prefix);
        assert!(bob.ends_with("Target peer:\nbob\n\nMessages to analyze:\n<messages>\nhi\n</messages>"));
        assert!(bob_prefix.contains("EXAMPLES (using `alice` as the target peer id):"));
    }

    #[test]
    fn estimate_with_blank_equals_minimal() {
        assert_eq!(
            estimate_deriver_prompt_tokens(Some("   ")),
            estimate_minimal_deriver_prompt_tokens()
        );
        assert_eq!(
            estimate_deriver_prompt_tokens(None),
            estimate_minimal_deriver_prompt_tokens()
        );
    }

    #[test]
    fn estimate_with_instructions_is_larger() {
        assert!(
            estimate_deriver_prompt_tokens(Some("be very terse and specific"))
                > estimate_minimal_deriver_prompt_tokens()
        );
    }
}
