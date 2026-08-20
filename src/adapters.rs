use anyhow::{bail, Context, Result};
use async_stream::try_stream;
use axum::body::Body;
use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use tracing::{info, warn};
use uuid::Uuid;

use crate::config::Protocol;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolKind {
    Function,
    Custom,
}

#[derive(Clone, Debug)]
pub struct ToolSpec {
    pub original_name: String,
    pub namespace: Option<String>,
    pub kind: ToolKind,
}

pub type ToolMap = HashMap<String, ToolSpec>;

pub fn redacted_json(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| {
                    let normalized = key.to_ascii_lowercase().replace('-', "_");
                    let sensitive = matches!(
                        normalized.as_str(),
                        "authorization"
                            | "proxy_authorization"
                            | "api_key"
                            | "x_api_key"
                            | "access_token"
                            | "refresh_token"
                            | "id_token"
                            | "password"
                            | "secret"
                            | "client_secret"
                            | "cookie"
                            | "set_cookie"
                            | "credential"
                    );
                    (
                        key.clone(),
                        if sensitive {
                            Value::String("[REDACTED]".to_owned())
                        } else {
                            redacted_json(value)
                        },
                    )
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(redacted_json).collect()),
        Value::String(value) => Value::String(redacted_string(value)),
        value => value.clone(),
    }
}

pub fn redacted_string(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    if [
        "api key",
        "apikey",
        "authorization:",
        "bearer ",
        "password:",
        "secret key",
    ]
    .iter()
    .any(|marker| lower.contains(marker))
    {
        return "[REDACTED SENSITIVE TEXT]".to_owned();
    }

    value
        .split_inclusive(char::is_whitespace)
        .map(|part| {
            let token = part.trim_matches(|character: char| {
                character.is_whitespace() || matches!(character, '"' | '\'' | ',' | ';' | '(' | ')')
            });
            let credential_like = token.starts_with("sk-")
                || token.starts_with("sk_")
                || (token.len() >= 40
                    && !token.contains('/')
                    && token.chars().all(|character| {
                        character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
                    })
                    && token
                        .chars()
                        .any(|character| character.is_ascii_alphabetic())
                    && token.chars().any(|character| character.is_ascii_digit()));
            if credential_like {
                if part.ends_with(char::is_whitespace) {
                    "[REDACTED] "
                } else {
                    "[REDACTED]"
                }
            } else {
                part
            }
        })
        .collect()
}

pub fn request(protocol: Protocol, body: &Value, upstream_model: &str) -> Result<Value> {
    match protocol {
        Protocol::Responses => {
            let mut output = body.clone();
            output["model"] = upstream_model.into();
            if upstream_model == "glm-5.3" {
                let effort = body
                    .pointer("/reasoning/effort")
                    .and_then(Value::as_str)
                    .map(glm_53_reasoning_effort)
                    .unwrap_or("low");
                if !output.get("reasoning").is_some_and(Value::is_object) {
                    output["reasoning"] = json!({});
                }
                output["reasoning"]["effort"] = effort.into();
            }
            Ok(output)
        }
        Protocol::ChatCompletions => responses_to_chat(body, upstream_model),
        Protocol::AnthropicMessages => responses_to_anthropic(body, upstream_model),
    }
}

pub fn response_body(
    protocol: Protocol,
    upstream: reqwest::Response,
    requested_model: String,
    trace_id: String,
    tool_map: ToolMap,
) -> Body {
    match protocol {
        Protocol::Responses => Body::from_stream(upstream.bytes_stream().map(move |chunk| {
            match chunk {
                Ok(bytes) => {
                    let payload = redacted_string(&String::from_utf8_lossy(&bytes));
                    info!(%trace_id, bytes = bytes.len(), payload = %payload, "raw upstream Responses chunk");
                    Ok(bytes)
                }
                Err(error) => {
                    warn!(%trace_id, %error, "upstream Responses transport error");
                    Err(std::io::Error::other(error))
                }
            }
        })),
        Protocol::ChatCompletions => chat_stream(upstream, requested_model, trace_id, tool_map),
        Protocol::AnthropicMessages => {
            anthropic_stream(upstream, requested_model, trace_id, tool_map)
        }
    }
}

fn responses_to_chat(body: &Value, model: &str) -> Result<Value> {
    let (tools, tool_map) = chat_tools(body);
    let mut messages = Vec::new();
    let mut pending_reasoning = String::new();
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        messages.push(json!({"role":"system", "content":instructions}));
    }
    for item in input_items(body)? {
        let kind = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        match kind {
            "reasoning" => {
                pending_reasoning = reasoning_item_text(&item);
                // Responses streams may place a reasoning item immediately after
                // the assistant's visible message and before its tool calls. It is
                // still part of that same Chat Completions assistant message.
                if !pending_reasoning.is_empty() {
                    if let Some(message) = messages.last_mut().filter(|message| {
                        message.get("role").and_then(Value::as_str) == Some("assistant")
                            && message.get("reasoning_content").is_none()
                    }) {
                        message["reasoning_content"] = pending_reasoning.clone().into();
                    }
                }
            }
            "message" => {
                if item.get("content").is_none() && item.get("tools").is_some() {
                    continue;
                }
                let role = match item.get("role").and_then(Value::as_str).unwrap_or("user") {
                    "developer" | "system" => "system",
                    role => role,
                };
                let mut message = json!({
                "role": role,
                "content": content_text(item.get("content").unwrap_or(&Value::Null))
                });
                if role == "assistant" && !pending_reasoning.is_empty() {
                    message["reasoning_content"] = pending_reasoning.clone().into();
                }
                messages.push(message)
            }
            "function_call" | "custom_tool_call" => {
                let leaf_name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                let original_name = item
                    .get("namespace")
                    .and_then(Value::as_str)
                    .map(|namespace| format!("{namespace}.{leaf_name}"))
                    .unwrap_or_else(|| leaf_name.to_owned());
                let upstream_name = tool_map
                    .iter()
                    .find_map(|(upstream, spec)| {
                        (spec.original_name == original_name).then_some(upstream.as_str())
                    })
                    .unwrap_or(original_name.as_str());
                let arguments = if kind == "custom_tool_call" {
                    json!({"input":item.get("input").cloned().unwrap_or_default()}).to_string()
                } else {
                    item.get("arguments")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_else(|| {
                            item.get("arguments")
                                .cloned()
                                .unwrap_or_default()
                                .to_string()
                        })
                };
                let tool_call = json!({
                "id": item.get("call_id").or_else(|| item.get("id")).and_then(Value::as_str).unwrap_or("call_unknown"),
                "type":"function", "function": {"name":upstream_name, "arguments":arguments}
                });
                if let Some(message) = messages.last_mut().filter(|message| {
                    message.get("role").and_then(Value::as_str) == Some("assistant")
                        && (message.get("tool_calls").is_some()
                            || message.get("reasoning_content").is_some())
                }) {
                    if message.get("tool_calls").is_none() {
                        message["tool_calls"] = json!([]);
                    }
                    message["tool_calls"]
                        .as_array_mut()
                        .unwrap()
                        .push(tool_call);
                } else {
                    let mut message =
                        json!({"role":"assistant", "content":null, "tool_calls":[tool_call]});
                    if !pending_reasoning.is_empty() {
                        message["reasoning_content"] = pending_reasoning.clone().into();
                    }
                    messages.push(message);
                }
            }
            "function_call_output" | "custom_tool_call_output" => {
                messages.push(json!({
                    "role":"tool", "tool_call_id":item["call_id"], "content":content_text(&item["output"])
                }));
                pending_reasoning.clear();
            }
            _ => {}
        }
    }
    let mut output = json!({"model":model, "messages":messages, "stream":true, "stream_options":{"include_usage":true}});
    copy_field(body, &mut output, "temperature", "temperature");
    copy_field(body, &mut output, "top_p", "top_p");
    copy_field(body, &mut output, "max_output_tokens", "max_tokens");
    if !tools.is_empty() {
        output["tools"] = tools.into();
    }
    if let Some(choice) = body.get("tool_choice") {
        output["tool_choice"] = chat_tool_choice(choice);
    }
    if model == "glm-5.3" {
        output["thinking"] = json!({"type":"enabled"});
        let effort = body
            .pointer("/reasoning/effort")
            .and_then(Value::as_str)
            .map(glm_53_reasoning_effort)
            .unwrap_or("low");
        output["reasoning_effort"] = effort.into();
    } else if matches!(model, "deepseek-v4-pro" | "deepseek-v4-flash") {
        output["thinking"] = json!({"type":"enabled"});
        let effort = body
            .pointer("/reasoning/effort")
            .and_then(Value::as_str)
            .map(deepseek_v4_reasoning_effort)
            .unwrap_or("high");
        output["reasoning_effort"] = effort.into();
    } else if let Some(effort) = body.pointer("/reasoning/effort").and_then(Value::as_str) {
        output["thinking"] = json!({"type":"enabled"});
        output["reasoning_effort"] = effort.into();
    }
    Ok(output)
}

fn reasoning_item_text(item: &Value) -> String {
    if let Some(encoded) = item
        .get("encrypted_content")
        .and_then(Value::as_str)
        .and_then(|value| value.strip_prefix("flatline:v1:"))
    {
        use base64::Engine as _;
        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(encoded) {
            if let Ok(reasoning) = String::from_utf8(bytes) {
                return reasoning;
            }
        }
    }
    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .chain(
            item.get("summary")
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        )
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

fn chat_tools(body: &Value) -> (Vec<Value>, ToolMap) {
    let mut output = Vec::new();
    let mut map = ToolMap::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        collect_chat_tools(tools, "", &mut output, &mut map);
    }
    if let Some(items) = body.get("input").and_then(Value::as_array) {
        for item in items {
            if let Some(tools) = item.get("tools").and_then(Value::as_array) {
                collect_chat_tools(tools, "", &mut output, &mut map);
            }
        }
    }
    (output, map)
}

fn collect_chat_tools(tools: &[Value], prefix: &str, output: &mut Vec<Value>, map: &mut ToolMap) {
    for tool in tools {
        let name = tool.get("name").and_then(Value::as_str).unwrap_or_default();
        let original_name = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}.{name}")
        };
        if tool.get("type").and_then(Value::as_str) == Some("namespace") {
            if let Some(children) = tool.get("tools").and_then(Value::as_array) {
                collect_chat_tools(children, &original_name, output, map);
            }
            continue;
        }
        let kind = if tool.get("type").and_then(Value::as_str) == Some("custom") {
            ToolKind::Custom
        } else {
            ToolKind::Function
        };
        let upstream_name = original_name.replace('.', "__");
        let parameters = if kind == ToolKind::Custom {
            json!({
                "type":"object",
                "properties":{"input":{"type":"string","description":"Raw custom-tool input"}},
                "required":["input"],
                "additionalProperties":false
            })
        } else {
            tool.get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type":"object"}))
        };
        output.push(json!({"type":"function", "function":{
            "name":upstream_name,
            "description":tool.get("description").cloned().unwrap_or_default(),
            "parameters":parameters
        }}));
        map.insert(
            upstream_name,
            ToolSpec {
                original_name,
                namespace: prefix.rsplit_once('.').map_or_else(
                    || (!prefix.is_empty()).then(|| prefix.to_owned()),
                    |_| Some(prefix.to_owned()),
                ),
                kind,
            },
        );
    }
}

pub fn tool_map(body: &Value) -> ToolMap {
    chat_tools(body).1
}

fn glm_53_reasoning_effort(effort: &str) -> &'static str {
    match effort {
        "none" | "minimal" | "low" => "low",
        "max" | "ultra" => "max",
        "medium" | "high" | "xhigh" => "high",
        _ => "high",
    }
}

fn deepseek_v4_reasoning_effort(effort: &str) -> &'static str {
    match effort {
        "xhigh" | "max" | "ultra" => "max",
        _ => "high",
    }
}

fn responses_to_anthropic(body: &Value, model: &str) -> Result<Value> {
    let mut messages = Vec::new();
    let mut system = body
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    for item in input_items(body)? {
        match item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message")
        {
            "message" => {
                let source_role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                if matches!(source_role, "developer" | "system") {
                    if !system.is_empty() {
                        system.push_str("\n\n");
                    }
                    system.push_str(&content_text(&item["content"]));
                    continue;
                }
                let role = if source_role == "assistant" {
                    "assistant"
                } else {
                    "user"
                };
                push_anthropic_message(
                    &mut messages,
                    role,
                    json!({"type":"text", "text":content_text(&item["content"])}),
                );
            }
            "function_call" => push_anthropic_message(
                &mut messages,
                "assistant",
                json!({
                    "type":"tool_use", "id":item.get("call_id").or_else(|| item.get("id")).cloned().unwrap_or_default(),
                    "name":item["name"], "input":parse_arguments(&item["arguments"])
                }),
            ),
            "function_call_output" => push_anthropic_message(
                &mut messages,
                "user",
                json!({
                    "type":"tool_result", "tool_use_id":item["call_id"], "content":content_text(&item["output"])
                }),
            ),
            _ => {}
        }
    }
    let tools = body.get("tools").and_then(Value::as_array).map(|tools| tools.iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
        .map(|tool| json!({
            "name":tool["name"], "description":tool.get("description").cloned().unwrap_or_default(),
            "input_schema":tool.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object"}))
        }))
        .collect::<Vec<_>>())
        .unwrap_or_default();
    let mut output = json!({"model":model, "messages":messages, "stream":true,
        "max_tokens":body.get("max_output_tokens").and_then(Value::as_u64).unwrap_or(16384)});
    if !system.is_empty() {
        output["system"] = system.into();
    }
    if !tools.is_empty() {
        output["tools"] = tools.into();
    }
    copy_field(body, &mut output, "temperature", "temperature");
    copy_field(body, &mut output, "top_p", "top_p");
    Ok(output)
}

fn input_items(body: &Value) -> Result<Vec<Value>> {
    match body.get("input") {
        Some(Value::Array(items)) => Ok(items.clone()),
        Some(Value::String(text)) => Ok(vec![json!({
            "type":"message", "role":"user", "content":[{"type":"input_text", "text":text}]
        })]),
        _ => bail!("Responses request input must be a string or array"),
    }
}

fn content_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| part.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => value.to_string(),
    }
}

fn parse_arguments(value: &Value) -> Value {
    value
        .as_str()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| value.clone())
}

fn copy_field(source: &Value, target: &mut Value, from: &str, to: &str) {
    if let Some(value) = source.get(from) {
        target[to] = value.clone();
    }
}

fn chat_tool_choice(value: &Value) -> Value {
    if value.get("type").and_then(Value::as_str) == Some("function") {
        json!({"type":"function", "function":{"name":value["name"]}})
    } else {
        value.clone()
    }
}

fn push_anthropic_message(messages: &mut Vec<Value>, role: &str, block: Value) {
    if let Some(last) = messages.last_mut().filter(|x| x["role"] == role) {
        last["content"].as_array_mut().unwrap().push(block);
    } else {
        messages.push(json!({"role":role, "content":[block]}));
    }
}

fn sse(value: Value) -> Bytes {
    Bytes::from(format!("data: {}\n\n", value))
}
fn response_shell(id: &str, model: &str, status: &str, output: Value, usage: Value) -> Value {
    json!({"id":id,"object":"response","created_at":chrono::Utc::now().timestamp(),"status":status,
        "model":model,"output":output,"usage":usage})
}

#[derive(Default)]
struct ChatState {
    text: String,
    text_started: bool,
    reasoning: String,
    tools: Map<String, Value>,
    usage: Value,
}

struct StreamTraceGuard {
    trace_id: String,
    completed: bool,
}

impl Drop for StreamTraceGuard {
    fn drop(&mut self) {
        if !self.completed {
            warn!(trace_id = %self.trace_id, "downstream stream dropped before completion");
        }
    }
}

fn chat_stream(
    upstream: reqwest::Response,
    model: String,
    trace_id: String,
    tool_map: ToolMap,
) -> Body {
    let stream = try_stream! {
        if false { Err::<(), anyhow::Error>(anyhow::anyhow!("unreachable"))?; }
        let mut stream_guard = StreamTraceGuard { trace_id: trace_id.clone(), completed: false };
        let response_id = format!("resp_{}", Uuid::new_v4().simple());
        let message_id = format!("msg_{}", Uuid::new_v4().simple());
        let empty_usage = json!({"input_tokens":0,"output_tokens":0,"total_tokens":0});
        yield sse(json!({"type":"response.created","response":response_shell(&response_id,&model,"in_progress",json!([]),empty_usage)}));
        let mut state = ChatState::default();
        let mut events = upstream.bytes_stream().eventsource();
        let mut chunk_index = 0_u64;
        while let Some(event) = events.next().await {
            chunk_index += 1;
            let event = event.map_err(|error| {
                warn!(%trace_id, chunk_index, %error, "upstream SSE transport error");
                anyhow::Error::from(error)
            })?;
            if event.data == "[DONE]" {
                info!(%trace_id, chunk_index, "upstream SSE done marker");
                break;
            }
            let chunk: Value = serde_json::from_str(&event.data).map_err(|error| {
                warn!(%trace_id, chunk_index, %error, bytes = event.data.len(), "invalid upstream SSE JSON");
                anyhow::Error::from(error).context("invalid Chat Completions SSE")
            })?;
            info!(%trace_id, chunk_index, payload = %redacted_json(&chunk), "full upstream chat chunk");
            if let Some(usage) = chunk.get("usage").filter(|x| !x.is_null()) { state.usage = normalize_chat_usage(usage); }
            let Some(delta) = chunk.pointer("/choices/0/delta") else { continue };
            info!(
                %trace_id,
                chunk_index,
                finish_reason = chunk.pointer("/choices/0/finish_reason").and_then(|value| value.as_str()).unwrap_or("unset"),
                content_bytes = delta.get("content").and_then(|value| value.as_str()).map_or(0, str::len),
                reasoning_bytes = delta.get("reasoning_content").and_then(|value| value.as_str()).map_or(0, str::len),
                tool_deltas = delta.get("tool_calls").and_then(|value| value.as_array()).map_or(0, Vec::len),
                "upstream chat chunk"
            );
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    if !state.text_started {
                        state.text_started = true;
                        yield sse(json!({"type":"response.output_item.added","output_index":0,"item":{"id":message_id,"type":"message","role":"assistant","status":"in_progress","content":[]}}));
                        yield sse(json!({"type":"response.content_part.added","item_id":message_id,"output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}));
                    }
                    state.text.push_str(text);
                    yield sse(json!({"type":"response.output_text.delta","item_id":message_id,"output_index":0,"content_index":0,"delta":text}));
                }
            }
            if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
                state.reasoning.push_str(reasoning);
            }
            if let Some(tools) = delta.get("tool_calls").and_then(Value::as_array) {
                for tool in tools {
                    let index = tool.get("index").and_then(Value::as_u64).unwrap_or(0).to_string();
                    let entry = state.tools.entry(index.clone()).or_insert_with(|| json!({"id":"","name":"","arguments":""}));
                    append_string(entry, "id", tool.get("id").and_then(Value::as_str));
                    append_string(entry, "name", tool.pointer("/function/name").and_then(Value::as_str));
                    append_string(entry, "arguments", tool.pointer("/function/arguments").and_then(Value::as_str));
                    info!(
                        %trace_id,
                        chunk_index,
                        tool_index = %index,
                        call_id = entry.get("id").and_then(|value| value.as_str()).unwrap_or("unset"),
                        name = entry.get("name").and_then(|value| value.as_str()).unwrap_or("unset"),
                        argument_bytes = entry.get("arguments").and_then(|value| value.as_str()).map_or(0, str::len),
                        "accumulated tool call"
                    );
                }
            }
        }
        let mut output = Vec::new();
        if state.text_started {
            yield sse(json!({"type":"response.output_text.done","item_id":message_id,"output_index":0,"content_index":0,"text":state.text}));
            let item = json!({"id":message_id,"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":state.text,"annotations":[]}]});
            yield sse(json!({"type":"response.output_item.done","output_index":0,"item":item})); output.push(item);
        }
        if !state.reasoning.is_empty() {
            use base64::Engine as _;
            let index = output.len();
            let encrypted_content = format!(
                "flatline:v1:{}",
                base64::engine::general_purpose::STANDARD.encode(state.reasoning.as_bytes())
            );
            let item = json!({
                "id":format!("rs_{}",Uuid::new_v4().simple()),
                "type":"reasoning",
                "status":"completed",
                "summary":[],
                "encrypted_content":encrypted_content
            });
            yield sse(json!({"type":"response.output_item.added","output_index":index,"item":{
                "id":item["id"],"type":"reasoning","status":"in_progress","summary":[]
            }}));
            yield sse(json!({"type":"response.output_item.done","output_index":index,"item":item}));
            output.push(item);
        }
        for (_, tool) in state.tools {
            let index = output.len();
            info!(
                %trace_id,
                output_index = index,
                call_id = tool.get("id").and_then(|value| value.as_str()).unwrap_or("unset"),
                name = tool.get("name").and_then(|value| value.as_str()).unwrap_or("unset"),
                argument_bytes = tool.get("arguments").and_then(|value| value.as_str()).map_or(0, str::len),
                "emitting Responses function call"
            );
            let upstream_name = tool.get("name").and_then(Value::as_str).unwrap_or_default();
            let spec = tool_map.get(upstream_name);
            let (events, item) = chat_tool_events(&tool, index, spec);
            for event in events {
                info!(%trace_id, event_type = event["type"].as_str().unwrap_or("unknown"), output_index = index, payload = %redacted_json(&event), "downstream Responses event");
                yield sse(event);
            }
            output.push(item);
        }
        let usage = if state.usage.is_null() { json!({"input_tokens":0,"output_tokens":0,"total_tokens":0}) } else { state.usage };
        let completed = json!({"type":"response.completed","response":response_shell(&response_id,&model,"completed",Value::Array(output),usage)});
        info!(%trace_id, output_items = completed.pointer("/response/output").and_then(|value| value.as_array()).map_or(0, Vec::len), payload = %redacted_json(&completed), "emitting response.completed");
        yield sse(completed);
        yield Bytes::from_static(b"data: [DONE]\n\n");
        stream_guard.completed = true;
        info!(%trace_id, "downstream stream finished");
    };
    Body::from_stream(stream.map(|item: anyhow::Result<Bytes>| item))
}

fn chat_tool_events(
    tool: &Value,
    output_index: usize,
    spec: Option<&ToolSpec>,
) -> (Vec<Value>, Value) {
    let item_id = format!("fc_{}", Uuid::new_v4().simple());
    let call_id = tool.get("id").cloned().unwrap_or_default();
    // Codex registers tools by the (namespace, name) pair.  Keep the leaf name
    // separate here: a dotted name such as `functions.exec` is treated as an
    // un-namespaced tool and therefore cannot be found by the dispatcher.
    let name = spec
        .map(|spec| {
            Value::String(
                spec.original_name
                    .rsplit('.')
                    .next()
                    .unwrap_or(&spec.original_name)
                    .to_owned(),
            )
        })
        .unwrap_or_else(|| tool.get("name").cloned().unwrap_or_default());
    let arguments = tool
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    if spec.is_some_and(|spec| spec.kind == ToolKind::Custom) {
        let input = arguments
            .as_str()
            .and_then(|arguments| serde_json::from_str::<Value>(arguments).ok())
            .and_then(|arguments| arguments.get("input").cloned())
            .unwrap_or_else(|| arguments.clone());
        let recipient = spec
            .map(|spec| Value::String(spec.original_name.clone()))
            .unwrap_or_else(|| name.clone());
        let mut added = json!({"type":"response.output_item.added","output_index":output_index,"item":{
            "id":item_id,"type":"custom_tool_call","status":"in_progress","call_id":call_id,"name":name,"recipient":recipient,"input":""
        }});
        let delta = json!({"type":"response.custom_tool_call_input.delta","item_id":item_id,"output_index":output_index,"delta":input});
        let input_done = json!({"type":"response.custom_tool_call_input.done","item_id":item_id,"output_index":output_index,"input":input});
        let mut item = json!({"id":item_id,"type":"custom_tool_call","status":"completed","call_id":call_id,"name":name,"recipient":recipient,"input":input});
        if let Some(namespace) = spec.and_then(|spec| spec.namespace.as_deref()) {
            added["item"]["namespace"] = namespace.into();
            item["namespace"] = namespace.into();
        }
        let item_done =
            json!({"type":"response.output_item.done","output_index":output_index,"item":item});
        return (vec![added, delta, input_done, item_done], item);
    }
    let mut added = json!({
        "type":"response.output_item.added",
        "output_index":output_index,
        "item":{
            "id":item_id,
            "type":"function_call",
            "status":"in_progress",
            "call_id":call_id,
            "name":name,
            "arguments":""
        }
    });
    let delta = json!({
        "type":"response.function_call_arguments.delta",
        "item_id":item_id,
        "output_index":output_index,
        "delta":arguments
    });
    let arguments_done = json!({
        "type":"response.function_call_arguments.done",
        "item_id":item_id,
        "output_index":output_index,
        "arguments":arguments
    });
    let mut item = json!({
        "id":item_id,
        "type":"function_call",
        "status":"completed",
        "call_id":call_id,
        "name":name,
        "arguments":arguments
    });
    if let Some(namespace) = spec.and_then(|spec| spec.namespace.as_deref()) {
        added["item"]["namespace"] = namespace.into();
        item["namespace"] = namespace.into();
    }
    let item_done = json!({
        "type":"response.output_item.done",
        "output_index":output_index,
        "item":item
    });
    (vec![added, delta, arguments_done, item_done], item)
}

fn anthropic_stream(
    upstream: reqwest::Response,
    model: String,
    trace_id: String,
    tool_map: ToolMap,
) -> Body {
    let stream = try_stream! {
        if false { Err::<(), anyhow::Error>(anyhow::anyhow!("unreachable"))?; }
        let response_id = format!("resp_{}", Uuid::new_v4().simple());
        yield sse(json!({"type":"response.created","response":response_shell(&response_id,&model,"in_progress",json!([]),json!({"input_tokens":0,"output_tokens":0,"total_tokens":0}))}));
        let mut output: Vec<Value> = Vec::new(); let mut blocks: Map<String,Value> = Map::new(); let mut usage = json!({"input_tokens":0,"output_tokens":0,"total_tokens":0});
        let mut events = upstream.bytes_stream().eventsource();
        while let Some(event) = events.next().await {
            let event = event.map_err(anyhow::Error::from)?; let value: Value = serde_json::from_str(&event.data).context("invalid Anthropic SSE")?;
            match value.get("type").and_then(Value::as_str) {
                Some("message_start") => { usage["input_tokens"] = value.pointer("/message/usage/input_tokens").cloned().unwrap_or(json!(0)); }
                Some("content_block_start") => { let index=value["index"].as_u64().unwrap_or(0).to_string(); blocks.insert(index, value["content_block"].clone()); }
                Some("content_block_delta") => { let index=value["index"].as_u64().unwrap_or(0).to_string(); if let Some(block)=blocks.get_mut(&index) {
                    if value.pointer("/delta/type").and_then(Value::as_str)==Some("text_delta") { let text=value.pointer("/delta/text").and_then(Value::as_str).unwrap_or(""); append_string(block,"text",Some(text)); yield sse(json!({"type":"response.output_text.delta","delta":text})); }
                    if value.pointer("/delta/type").and_then(Value::as_str)==Some("input_json_delta") { append_string(block,"partial_json",value.pointer("/delta/partial_json").and_then(Value::as_str)); }
                }}
                Some("content_block_stop") => { let index=value["index"].as_u64().unwrap_or(0).to_string(); if let Some(block)=blocks.remove(&index) {
                    let i=output.len();
                    if block["type"]=="tool_use" {
                        let tool = json!({
                            "id":block["id"],
                            "name":block["name"],
                            "arguments":block.get("partial_json").cloned().unwrap_or_else(|| block["input"].to_string().into())
                        });
                        let spec = block["name"].as_str().and_then(|name| tool_map.get(name));
                        let (events, item) = chat_tool_events(&tool, i, spec);
                        for event in events {
                            info!(%trace_id, event_type = event["type"].as_str().unwrap_or("unknown"), output_index = i, "emitting Anthropic Responses tool event");
                            yield sse(event);
                        }
                        output.push(item);
                    } else {
                        let item=json!({"id":format!("msg_{}",Uuid::new_v4().simple()),"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":block["text"],"annotations":[]}]});
                        yield sse(json!({"type":"response.output_item.done","output_index":i,"item":item}));
                        output.push(item);
                    }
                }}
                Some("message_delta") => { usage["output_tokens"] = value.pointer("/usage/output_tokens").cloned().unwrap_or(json!(0)); }
                _ => {}
            }
        }
        usage["total_tokens"] = json!(usage["input_tokens"].as_u64().unwrap_or(0)+usage["output_tokens"].as_u64().unwrap_or(0));
        yield sse(json!({"type":"response.completed","response":response_shell(&response_id,&model,"completed",Value::Array(output),usage)})); yield Bytes::from_static(b"data: [DONE]\n\n");
    };
    Body::from_stream(stream.map(|item: anyhow::Result<Bytes>| item))
}

fn append_string(target: &mut Value, key: &str, part: Option<&str>) {
    if let Some(part) = part {
        let current = target.get(key).and_then(Value::as_str).unwrap_or("");
        target[key] = format!("{current}{part}").into();
    }
}
fn normalize_chat_usage(value: &Value) -> Value {
    json!({"input_tokens":value.get("prompt_tokens").cloned().unwrap_or(json!(0)),"output_tokens":value.get("completion_tokens").cloned().unwrap_or(json!(0)),"total_tokens":value.get("total_tokens").cloned().unwrap_or(json!(0)),"input_tokens_details":{"cached_tokens":value.pointer("/prompt_tokens_details/cached_tokens").cloned().unwrap_or(json!(0))}})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn converts_codex_tools_to_chat_tools() {
        let source = json!({"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[{"type":"function","name":"shell","description":"run","parameters":{"type":"object"}}]});
        let out = responses_to_chat(&source, "deepseek-chat").unwrap();
        assert_eq!(out["messages"][0]["content"], "hi");
        assert_eq!(out["tools"][0]["function"]["name"], "shell");
    }
    #[test]
    fn groups_parallel_tool_calls_before_their_results() {
        let source = json!({
            "input": [
                {"type":"message","role":"assistant","content":"Checking three things."},
                {"type":"function_call","call_id":"call_1","name":"one","arguments":"{}"},
                {"type":"function_call","call_id":"call_2","name":"two","arguments":"{}"},
                {"type":"function_call","call_id":"call_3","name":"three","arguments":"{}"},
                {"type":"function_call_output","call_id":"call_1","output":"first"},
                {"type":"function_call_output","call_id":"call_2","output":"second"},
                {"type":"function_call_output","call_id":"call_3","output":"third"}
            ]
        });
        let out = responses_to_chat(&source, "deepseek-v4-flash").unwrap();
        let messages = out["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["tool_calls"].as_array().unwrap().len(), 3);
        assert_eq!(messages[2]["tool_call_id"], "call_1");
        assert_eq!(messages[3]["tool_call_id"], "call_2");
        assert_eq!(messages[4]["tool_call_id"], "call_3");
    }
    #[test]
    fn replays_reasoning_content_with_tool_calls() {
        use base64::Engine as _;
        let encrypted = format!(
            "flatline:v1:{}",
            base64::engine::general_purpose::STANDARD.encode("provider reasoning token")
        );
        let source = json!({
            "input": [
                {"type":"reasoning","id":"rs_1","status":"completed","summary":[],
                 "encrypted_content":encrypted},
                {"type":"message","role":"assistant","content":"I will check."},
                {"type":"function_call","call_id":"call_1","name":"check","arguments":"{}"},
                {"type":"function_call_output","call_id":"call_1","output":"done"}
            ]
        });
        let out = responses_to_chat(&source, "deepseek-v4-flash").unwrap();
        let messages = out["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "I will check.");
        assert_eq!(messages[0]["reasoning_content"], "provider reasoning token");
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_1");
    }
    #[test]
    fn coalesces_text_then_reasoning_then_tool_call() {
        use base64::Engine as _;
        let encrypted = format!(
            "flatline:v1:{}",
            base64::engine::general_purpose::STANDARD.encode("required provider reasoning")
        );
        let source = json!({
            "input": [
                {"type":"message","role":"assistant","content":"I will run it."},
                {"type":"reasoning","id":"rs_1","status":"completed","summary":[],
                 "encrypted_content":encrypted},
                {"type":"custom_tool_call","call_id":"call_1","name":"exec",
                 "input":"await tools.exec_command({cmd: \"echo hello\"})"},
                {"type":"custom_tool_call_output","call_id":"call_1","output":"hello\n"}
            ]
        });
        let out = responses_to_chat(&source, "deepseek-v4-flash").unwrap();
        let messages = out["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["content"], "I will run it.");
        assert_eq!(
            messages[0]["reasoning_content"],
            "required provider reasoning"
        );
        assert_eq!(messages[0]["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(messages[1]["role"], "tool");
    }
    #[test]
    fn maps_codex_effort_for_glm_53() {
        for (codex, glm) in [
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "max"),
        ] {
            let source = json!({"input":"hi", "reasoning":{"effort":codex}});
            let out = responses_to_chat(&source, "glm-5.3").unwrap();
            assert_eq!(out["thinking"]["type"], "enabled");
            assert_eq!(out["reasoning_effort"], glm);
        }
    }
    #[test]
    fn maps_glm_53_effort_for_native_responses() {
        let source = json!({"model":"glm-5.3", "input":"hi", "reasoning":{"effort":"xhigh"}});
        let out = request(Protocol::Responses, &source, "glm-5.3").unwrap();
        assert_eq!(out["reasoning"]["effort"], "high");
    }
    #[test]
    fn glm_53_always_enables_thinking() {
        let out = responses_to_chat(&json!({"input":"hi"}), "glm-5.3").unwrap();
        assert_eq!(out["thinking"]["type"], "enabled");
        assert_eq!(out["reasoning_effort"], "low");
    }
    #[test]
    fn maps_codex_effort_for_deepseek_v4() {
        for model in ["deepseek-v4-pro", "deepseek-v4-flash"] {
            for (codex, deepseek) in [
                ("low", "high"),
                ("medium", "high"),
                ("high", "high"),
                ("xhigh", "max"),
                ("ultra", "max"),
            ] {
                let source = json!({"input":"hi", "reasoning":{"effort":codex}});
                let out = responses_to_chat(&source, model).unwrap();
                assert_eq!(out["thinking"]["type"], "enabled");
                assert_eq!(out["reasoning_effort"], deepseek);
            }
        }
    }
    #[test]
    fn emits_complete_responses_tool_call_lifecycle() {
        let tool = json!({"id":"call_123", "name":"shell", "arguments":"{\"cmd\":\"pwd\"}"});
        let (events, item) = chat_tool_events(&tool, 1, None);
        assert_eq!(events[0]["type"], "response.output_item.added");
        assert_eq!(events[0]["item"]["status"], "in_progress");
        assert_eq!(events[1]["type"], "response.function_call_arguments.delta");
        assert_eq!(events[2]["type"], "response.function_call_arguments.done");
        assert_eq!(events[3]["type"], "response.output_item.done");
        assert_eq!(item["status"], "completed");
        assert_eq!(item["call_id"], "call_123");
    }
    #[test]
    fn redacts_secrets_from_logged_json() {
        let source = json!({
            "authorization":"Bearer should-never-appear",
            "message":"api key: should-never-appear",
            "nested":{"access_token":"should-never-appear"},
            "safe":"hello"
        });
        let redacted = redacted_json(&source).to_string();
        assert!(!redacted.contains("should-never-appear"));
        assert!(redacted.contains("hello"));
    }
    #[test]
    fn flattens_codex_namespaced_and_custom_tools() {
        let source = json!({"input":[{"type":"message","role":"developer","tools":[
            {"type":"namespace","name":"functions","tools":[
                {"type":"custom","name":"exec","description":"run code"},
                {"type":"function","name":"wait","parameters":{"type":"object"}}
            ]}
        ]},{"type":"message","role":"user","content":"use a tool"}]});
        let out = responses_to_chat(&source, "glm-5.3").unwrap();
        assert_eq!(out["tools"][0]["function"]["name"], "functions__exec");
        assert_eq!(out["tools"][1]["function"]["name"], "functions__wait");
        assert_eq!(out["messages"].as_array().unwrap().len(), 1);
        let map = tool_map(&source);
        let tool =
            json!({"id":"call_1","name":"functions__exec","arguments":"{\"input\":\"text(1)\"}"});
        let (events, item) = chat_tool_events(&tool, 0, map.get("functions__exec"));
        assert_eq!(events[0]["item"]["type"], "custom_tool_call");
        assert_eq!(events[0]["item"]["recipient"], "functions.exec");
        assert_eq!(events[0]["item"]["namespace"], "functions");
        assert_eq!(events[0]["item"]["name"], "exec");
        assert_eq!(events[1]["type"], "response.custom_tool_call_input.delta");
        assert_eq!(item["namespace"], "functions");
        assert_eq!(item["name"], "exec");
        assert_eq!(item["recipient"], "functions.exec");
        assert_eq!(item["input"], "text(1)");
    }
    #[test]
    fn converts_tool_results_to_anthropic() {
        let source =
            json!({"input":[{"type":"function_call_output","call_id":"c1","output":"ok"}]});
        let out = responses_to_anthropic(&source, "claude").unwrap();
        assert_eq!(out["messages"][0]["content"][0]["type"], "tool_result");
    }
}
