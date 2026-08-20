use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::config::PromptGuard;

const CODEX_PROMPT_PREFIX: &str = "You are Codex,";

pub struct PromptTemplate {
    expected: String,
    template: String,
}

impl PromptTemplate {
    pub async fn load_and_validate(body: &Value, guard: &PromptGuard) -> Result<Option<Self>> {
        if !guard.enabled {
            return Ok(None);
        }

        let expected = read_prompt_file(&guard.expected_path)
            .await
            .with_context(|| format!("load expected Codex prompt from {}", guard.expected_path))?;
        let template = read_prompt_file(&guard.template_path)
            .await
            .with_context(|| {
                format!("load replacement Codex prompt from {}", guard.template_path)
            })?;

        let candidates = prompt_texts(body);
        let exact_matches = candidates
            .iter()
            .filter(|text| text.as_str() == expected)
            .count();
        if exact_matches != 1 {
            let observed = candidates
                .iter()
                .find(|text| text.starts_with(CODEX_PROMPT_PREFIX));
            let detail = observed.map_or_else(
                || {
                    "no developer/system message beginning with 'You are Codex,' was present"
                        .to_string()
                },
                |text| {
                    format!(
                        "expected fingerprint {}, observed fingerprint {}",
                        fingerprint(&expected),
                        fingerprint(text)
                    )
                },
            );
            bail!(
                "CODEX PROMPT GUARD FAILED: expected exactly one byte-for-byte prompt match, found {exact_matches}; {detail}. Refusing to forward the request. Review the new Codex prompt and update {} deliberately",
                guard.expected_path
            );
        }

        Ok(Some(Self { expected, template }))
    }

    pub fn apply(
        &self,
        body: &mut Value,
        requested_model: &str,
        provider: &str,
        upstream_model: &str,
    ) -> Result<()> {
        let rendered = self
            .template
            .replace("{{requested_model}}", requested_model)
            .replace("{{provider}}", provider)
            .replace("{{upstream_model}}", upstream_model);
        let mut replacements = 0;
        visit_prompt_texts_mut(body, &mut |text| {
            if text == &self.expected {
                *text = rendered.clone();
                replacements += 1;
            }
        });
        if replacements != 1 {
            bail!("CODEX PROMPT GUARD FAILED during replacement: replaced {replacements} prompts");
        }
        Ok(())
    }
}

async fn read_prompt_file(path: &str) -> Result<String> {
    let mut text = tokio::fs::read_to_string(path).await?;
    // Text files conventionally end in LF; Codex's JSON string generally does not.
    if text.ends_with('\n') {
        text.pop();
        if text.ends_with('\r') {
            text.pop();
        }
    }
    Ok(text)
}

fn prompt_texts(body: &Value) -> Vec<String> {
    let mut result = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        result.push(instructions.to_owned());
    }
    let Some(items) = body.get("input").and_then(Value::as_array) else {
        return result;
    };
    for item in items {
        if !matches!(
            item.get("role").and_then(Value::as_str),
            Some("developer" | "system")
        ) {
            continue;
        }
        match item.get("content") {
            Some(Value::String(text)) => result.push(text.clone()),
            Some(Value::Array(parts)) => {
                result.extend(parts.iter().filter_map(|part| {
                    part.get("text").and_then(Value::as_str).map(str::to_owned)
                }));
            }
            _ => {}
        }
    }
    result
}

fn visit_prompt_texts_mut(body: &mut Value, visitor: &mut impl FnMut(&mut String)) {
    if let Some(Value::String(instructions)) = body.get_mut("instructions") {
        visitor(instructions);
    }
    let Some(items) = body.get_mut("input").and_then(Value::as_array_mut) else {
        return;
    };
    for item in items {
        if !matches!(
            item.get("role").and_then(Value::as_str),
            Some("developer" | "system")
        ) {
            continue;
        }
        match item.get_mut("content") {
            Some(Value::String(text)) => visitor(text),
            Some(Value::Array(parts)) => {
                for part in parts {
                    if let Some(slot) = part.get_mut("text") {
                        if let Some(text) = slot.as_str() {
                            let mut changed = text.to_owned();
                            visitor(&mut changed);
                            *slot = Value::String(changed);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn fingerprint(text: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.as_bytes() {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn replaces_exact_prompt_and_renders_route_variables() {
        let template = PromptTemplate {
            expected: "You are Codex, old".into(),
            template: "You are Codex, running {{upstream_model}} through {{provider}} for {{requested_model}}".into(),
        };
        let mut body = json!({"input":[{"role":"developer","content":[{"type":"input_text","text":"You are Codex, old"}]}]});
        template
            .apply(&mut body, "alias", "anthropic", "claude-opus-5")
            .unwrap();
        assert_eq!(
            body.pointer("/input/0/content/0/text")
                .and_then(Value::as_str),
            Some("You are Codex, running claude-opus-5 through anthropic for alias")
        );
    }

    #[test]
    fn replaces_top_level_responses_instructions() {
        let template = PromptTemplate {
            expected: "You are Codex, old".into(),
            template: "You are Codex, running {{upstream_model}}".into(),
        };
        let mut body = json!({
            "instructions": "You are Codex, old",
            "input": [{"role":"user","content":"hello"}]
        });
        template
            .apply(&mut body, "alias", "zai", "glm-5.3")
            .unwrap();
        assert_eq!(
            body.get("instructions").and_then(Value::as_str),
            Some("You are Codex, running glm-5.3")
        );
    }
}
