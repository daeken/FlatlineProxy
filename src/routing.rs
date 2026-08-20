use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tokio::sync::RwLock;

use crate::config::{Config, Provider, Route};

#[derive(Default)]
pub struct CacheTracker {
    entries: RwLock<HashMap<(String, String), Instant>>,
}

impl CacheTracker {
    pub async fn mark(&self, provider: &str, fingerprint: &str) {
        self.entries
            .write()
            .await
            .insert((provider.into(), fingerprint.into()), Instant::now());
    }

    pub async fn resident(&self, provider: &str, fingerprint: &str, ttl: Duration) -> bool {
        self.entries
            .read()
            .await
            .get(&(provider.into(), fingerprint.into()))
            .is_some_and(|seen| seen.elapsed() <= ttl)
    }
}

#[derive(Clone)]
pub struct Selection {
    pub provider: Provider,
    pub route: Route,
    pub cache_resident: bool,
}

pub async fn candidates(
    config: &Config,
    model: &str,
    body: &Value,
    cache: &CacheTracker,
) -> Result<Vec<Selection>> {
    let policy = config
        .models
        .get(model)
        .with_context(|| format!("no routing policy for model {model}"))?;
    let fingerprint = prompt_fingerprint(body);
    let ttl = Duration::from_secs(policy.cache_residency_seconds);
    let mut choices = vec![];

    for route in &policy.routes {
        let Some(provider) = config
            .providers
            .iter()
            .find(|p| p.id == route.provider && p.enabled)
        else {
            continue;
        };
        if provider.auth != crate::config::Auth::IncomingOpenAi
            && !provider
                .api_key_env
                .as_deref()
                .is_some_and(|name| std::env::var(name).is_ok())
            && provider.keychain_service.is_none()
        {
            continue;
        }
        let resident = cache.resident(&provider.id, &fingerprint, ttl).await;
        let input_rate = if resident {
            route
                .cached_input_cost_per_million
                .unwrap_or(route.input_cost_per_million)
        } else {
            route.input_cost_per_million
        };
        let approximate_input_tokens = serde_json::to_vec(body)
            .map(|x| x.len() as f64 / 4.0)
            .unwrap_or(0.0);
        let expected_output_tokens = body
            .get("max_output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(4096) as f64;
        let estimated_cost = (approximate_input_tokens * input_rate
            + expected_output_tokens * route.output_cost_per_million)
            / 1_000_000.0;
        choices.push((
            estimated_cost,
            -route.priority,
            provider.clone(),
            route.clone(),
            resident,
        ));
    }
    choices.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    if choices.is_empty() {
        bail!("no enabled route has an available API key")
    }
    Ok(choices
        .into_iter()
        .map(|(_, _, provider, route, cache_resident)| Selection {
            provider,
            route,
            cache_resident,
        })
        .collect())
}

pub fn prompt_fingerprint(body: &Value) -> String {
    // A provider cache normally matches stable prompt prefixes. Track the model,
    // instructions, tools, and first input item instead of the growing full turn.
    let normalized = json_object_prefix(body);
    let bytes = serde_json::to_vec(&normalized).unwrap_or_default();
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn json_object_prefix(body: &Value) -> Value {
    let first_input = match body.get("input") {
        Some(Value::Array(items)) => items.first().cloned(),
        Some(Value::String(text)) => Some(Value::String(text.clone())),
        _ => None,
    };
    serde_json::json!({
        "model": body.get("model"),
        "instructions": body.get("instructions"),
        "tools": body.get("tools"),
        "first_input": first_input,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fingerprint_ignores_transport_and_observability_fields() {
        let first = json!({"model":"x", "input":"hello", "stream":true, "metadata":{"id":1}});
        let second = json!({"model":"x", "input":"hello", "stream":false, "metadata":{"id":2}});
        assert_eq!(prompt_fingerprint(&first), prompt_fingerprint(&second));
    }

    #[test]
    fn fingerprint_stays_stable_as_conversation_grows() {
        let first =
            json!({"model":"x", "instructions":"i", "input":[{"role":"user","content":"one"}]});
        let second = json!({"model":"x", "instructions":"i", "input":[{"role":"user","content":"one"},{"role":"assistant","content":"two"}]});
        assert_eq!(prompt_fingerprint(&first), prompt_fingerprint(&second));
    }

    #[test]
    fn fingerprint_changes_with_initial_prompt() {
        assert_ne!(
            prompt_fingerprint(&json!({"model":"x", "input":"one"})),
            prompt_fingerprint(&json!({"model":"x", "input":"two"}))
        );
    }
}
