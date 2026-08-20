mod adapters;
mod config;
mod routing;

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::{
    collections::{HashMap, VecDeque},
    fs::OpenOptions,
    io::{self, Write},
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{RawQuery, State},
    http::{header, HeaderMap, HeaderValue, Response, StatusCode},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use clap::Parser;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::RwLock;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use uuid::Uuid;

use config::{Auth, Config, ConfigStore, Protocol};
use routing::{candidates, prompt_fingerprint, CacheTracker};

#[derive(Parser)]
struct Args {
    #[arg(long, env = "FLATLINE_LISTEN", default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
    #[arg(long, env = "FLATLINE_CONFIG", default_value = "flatline.json")]
    config: PathBuf,
    #[arg(long, env = "FLATLINE_LOG_FILE", default_value = "flatline.log")]
    log_file: PathBuf,
}

struct TeeWriter {
    file: Arc<Mutex<std::fs::File>>,
}

impl Write for TeeWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        io::stderr().write_all(buffer)?;
        self.file
            .lock()
            .map_err(|_| io::Error::other("log file lock poisoned"))?
            .write_all(buffer)?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()?;
        self.file
            .lock()
            .map_err(|_| io::Error::other("log file lock poisoned"))?
            .flush()
    }
}

#[derive(Clone)]
struct AppState {
    config: Arc<RwLock<Config>>,
    store: ConfigStore,
    client: reqwest::Client,
    cache: Arc<CacheTracker>,
    usage: Arc<RwLock<VecDeque<UsageEvent>>>,
    cooldowns: Arc<RwLock<HashMap<String, Instant>>>,
    models_etag: Arc<RwLock<Option<String>>>,
}

#[derive(Clone, Serialize)]
struct UsageEvent {
    id: Uuid,
    at: DateTime<Utc>,
    requested_model: String,
    upstream_model: String,
    provider: String,
    cache_resident: bool,
    status: Option<u16>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(parent) = args
        .log_file
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).context("create log directory")?;
    }
    let mut log_options = OpenOptions::new();
    log_options.create(true).append(true);
    #[cfg(unix)]
    log_options.mode(0o600);
    let log_handle = log_options.open(&args.log_file).context("open log file")?;
    #[cfg(unix)]
    log_handle
        .set_permissions(std::fs::Permissions::from_mode(0o600))
        .context("secure log file permissions")?;
    let log_file = Arc::new(Mutex::new(log_handle));
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "flatline_proxy=info,tower_http=info".into()),
        )
        .with_writer(move || TeeWriter {
            file: Arc::clone(&log_file),
        })
        .init();
    let store = ConfigStore::new(args.config);
    let config = store.load().await?;
    let state = AppState {
        config: Arc::new(RwLock::new(config)),
        store,
        client: reqwest::Client::builder()
            .user_agent("flatline-proxy/0.1")
            .build()?,
        cache: Arc::new(CacheTracker::default()),
        usage: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
        cooldowns: Arc::new(RwLock::new(HashMap::new())),
        models_etag: Arc::new(RwLock::new(None)),
    };
    let app = Router::new()
        .route("/v1/responses", post(responses))
        .route("/v1/models", get(models))
        .route("/api/config", get(get_config).put(put_config))
        .route("/api/usage", get(get_usage))
        .route("/health", get(|| async { Json(json!({"ok": true})) }))
        .route("/", get(dashboard))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    info!(listen = %args.listen, log_file = %args.log_file.display(), "Flatline Proxy listening");
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Response<Body> {
    match forward_response(&state, &headers, &mut body).await {
        Ok(response) => response,
        Err(error) => {
            warn!(%error, "request failed");
            let payload =
                json!({"error":{"message":error.to_string(),"type":"flatline_proxy_error"}});
            (StatusCode::BAD_GATEWAY, Json(payload)).into_response()
        }
    }
}

async fn models(
    State(state): State<AppState>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
) -> Response<Body> {
    match fetch_models(&state, &headers, query.as_deref()).await {
        Ok((catalog, etag)) => {
            *state.models_etag.write().await = Some(etag.clone());
            let mut response = Json(catalog).into_response();
            if let Ok(value) = HeaderValue::from_str(&etag) {
                response.headers_mut().insert("x-models-etag", value);
            }
            if let Ok(value) = HeaderValue::from_str(&format!("\"{etag}\"")) {
                response.headers_mut().insert(header::ETAG, value);
            }
            response
        }
        Err(error) => {
            warn!(%error, "model catalog request failed");
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": error.to_string()})),
            )
                .into_response()
        }
    }
}

async fn fetch_models(
    state: &AppState,
    headers: &HeaderMap,
    query: Option<&str>,
) -> Result<(Value, String)> {
    let mut url = "https://chatgpt.com/backend-api/codex/models".to_string();
    if let Some(query) = query.filter(|query| !query.is_empty()) {
        url.push('?');
        url.push_str(query);
    }
    let mut request = state.client.get(url);
    for name in [
        header::AUTHORIZATION.as_str(),
        "chatgpt-account-id",
        "x-openai-fedramp",
        "originator",
    ] {
        if let Some(value) = headers.get(name) {
            request = request.header(name, value);
        }
    }
    let upstream = request.send().await?;
    anyhow::ensure!(
        upstream.status().is_success(),
        "upstream models returned HTTP {}",
        upstream.status()
    );
    let mut catalog: Value = upstream.json().await?;
    let configured = state.config.read().await.catalog_models.clone();
    let models = catalog
        .get_mut("models")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| anyhow::anyhow!("upstream catalog has no models array"))?;
    let template = models
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("upstream catalog is empty"))?;
    for (offset, entry) in configured.iter().enumerate() {
        if models
            .iter()
            .any(|model| model.get("slug").and_then(Value::as_str) == Some(&entry.id))
        {
            continue;
        }
        let mut model = template.clone();
        model["slug"] = entry.id.clone().into();
        model["display_name"] = entry.display_name.clone().into();
        model["description"] = entry
            .description
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null);
        model["priority"] = json!(-1000 - offset as i64);
        model["supported_in_api"] = Value::Bool(true);
        model["visibility"] = "list".into();
        model["upgrade"] = Value::Null;
        model["availability_nux"] = Value::Null;
        for (key, value) in &entry.metadata {
            model[key] = value.clone();
        }
        models.push(model);
    }
    let encoded = serde_json::to_vec(&catalog)?;
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in encoded {
        hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
    }
    Ok((catalog, format!("flatline-{hash:016x}")))
}

async fn forward_response(
    state: &AppState,
    incoming_headers: &HeaderMap,
    body: &mut Value,
) -> Result<Response<Body>> {
    let trace_id = Uuid::new_v4().simple().to_string();
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let input_items = body
        .get("input")
        .and_then(Value::as_array)
        .map_or(usize::from(body.get("input").is_some()), Vec::len);
    let tool_outputs = body
        .get("input")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|item| {
                    item.get("type").and_then(Value::as_str) == Some("function_call_output")
                })
                .count()
        })
        .unwrap_or(0);
    info!(
        %trace_id,
        %requested_model,
        input_items,
        tool_outputs,
        tools = body.get("tools").and_then(|value| value.as_array()).map_or(0, Vec::len),
        effort = body.pointer("/reasoning/effort").and_then(|value| value.as_str()).unwrap_or("unset"),
        "responses request received"
    );
    info!(%trace_id, headers = %redacted_headers(incoming_headers), payload = %adapters::redacted_json(body), "full inbound request");
    let config = state.config.read().await;
    let selections = candidates(&config, &requested_model, body, &state.cache).await?;
    let fingerprint = prompt_fingerprint(body);
    drop(config);
    let mut failures = Vec::new();
    for selection in selections {
        let provider_id = selection.provider.id.clone();
        info!(
            %trace_id,
            %provider_id,
            upstream_model = %selection.route.upstream_model,
            protocol = ?selection.provider.protocol,
            cache_resident = selection.cache_resident,
            "attempting route"
        );
        if let Some(until) = state.cooldowns.read().await.get(&provider_id).copied() {
            let now = Instant::now();
            if until > now {
                let delay = until - now;
                info!(%trace_id, %provider_id, ?delay, "waiting for provider cooldown");
                tokio::time::sleep(delay).await;
            }
        }
        let url = format!(
            "{}{}",
            selection.provider.base_url.trim_end_matches('/'),
            selection.provider.path.as_str()
        );
        let tool_map = adapters::tool_map(body);
        let adapted = adapters::request(
            selection.provider.protocol,
            body,
            &selection.route.upstream_model,
        )?;
        info!(%trace_id, %provider_id, url = %url, payload = %adapters::redacted_json(&adapted), "full adapted upstream request");
        let mut request = state.client.post(url).json(&adapted);
        request = match selection.provider.auth {
            Auth::Bearer => request.bearer_auth(provider_key(&selection.provider).await?),
            Auth::AnthropicKey => request
                .header("x-api-key", provider_key(&selection.provider).await?)
                .header("anthropic-version", "2023-06-01"),
            Auth::IncomingOpenAi => {
                let Some(authorization) = incoming_headers.get(header::AUTHORIZATION) else {
                    failures.push(format!(
                        "{provider_id}: Codex did not send ChatGPT authentication"
                    ));
                    continue;
                };
                let mut authenticated = request.header(header::AUTHORIZATION, authorization);
                for name in [
                    "chatgpt-account-id",
                    "x-openai-fedramp",
                    "originator",
                    "x-client-request-id",
                    "session_id",
                    "x-session-id",
                    "x-openai-subagent",
                ] {
                    if let Some(value) = incoming_headers.get(name) {
                        authenticated = authenticated.header(name, value);
                    }
                }
                authenticated
            }
        };
        if selection.provider.protocol == Protocol::Responses {
            // A Responses-compatible provider may use these headers to select
            // its Codex transport behavior. Preserve the direct-client shape
            // while deliberately excluding credentials, cookies, attestation,
            // account identifiers, Host, and framing headers.
            for name in [
                header::ACCEPT.as_str(),
                header::USER_AGENT.as_str(),
                "originator",
                "session-id",
                "thread-id",
                "x-client-request-id",
                "x-codex-beta-features",
                "x-codex-turn-metadata",
                "x-codex-window-id",
                "x-openai-internal-codex-responses-lite",
                "openai-organization",
                "openai-project",
            ] {
                if let Some(value) = incoming_headers.get(name) {
                    request = request.header(name, value);
                }
            }
        }
        match send_with_backoff(request, &provider_id, &trace_id).await {
            Ok(upstream) => {
                let status = upstream.status();
                info!(%trace_id, %provider_id, %status, headers = %redacted_headers(upstream.headers()), "upstream response received");
                record_usage(
                    &state.usage,
                    UsageEvent {
                        id: Uuid::new_v4(),
                        at: Utc::now(),
                        requested_model: requested_model.clone(),
                        upstream_model: selection.route.upstream_model.clone(),
                        provider: provider_id.clone(),
                        cache_resident: selection.cache_resident,
                        status: Some(status.as_u16()),
                    },
                )
                .await;
                if !status.is_success() && retryable(status) {
                    let delay = upstream
                        .headers()
                        .get(header::RETRY_AFTER)
                        .and_then(|value| value.to_str().ok())
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(Duration::from_secs)
                        .unwrap_or(Duration::from_secs(5));
                    state
                        .cooldowns
                        .write()
                        .await
                        .insert(provider_id.clone(), Instant::now() + delay);
                    failures.push(format!("{provider_id}: HTTP {}", status.as_u16()));
                    continue;
                }
                if status.is_success() {
                    state.cache.mark(&provider_id, &fingerprint).await;
                }
                let translate =
                    status.is_success() && selection.provider.protocol != Protocol::Responses;
                let content_type = if !translate {
                    upstream
                        .headers()
                        .get(header::CONTENT_TYPE)
                        .cloned()
                        .unwrap_or_else(|| HeaderValue::from_static("application/json"))
                } else {
                    HeaderValue::from_static("text/event-stream")
                };
                let response_body = if translate {
                    adapters::response_body(
                        selection.provider.protocol,
                        upstream,
                        requested_model.clone(),
                        trace_id.clone(),
                        tool_map.clone(),
                    )
                } else {
                    adapters::response_body(
                        Protocol::Responses,
                        upstream,
                        requested_model.clone(),
                        trace_id.clone(),
                        tool_map.clone(),
                    )
                };
                let mut response = Response::builder().status(status).body(response_body)?;
                response
                    .headers_mut()
                    .insert(header::CONTENT_TYPE, content_type);
                response
                    .headers_mut()
                    .insert("x-flatline-provider", HeaderValue::from_str(&provider_id)?);
                if let Some(etag) = state.models_etag.read().await.as_deref() {
                    response
                        .headers_mut()
                        .insert("x-models-etag", HeaderValue::from_str(etag)?);
                }
                return Ok(response);
            }
            Err(error) => {
                state
                    .cooldowns
                    .write()
                    .await
                    .insert(provider_id.clone(), Instant::now() + Duration::from_secs(5));
                failures.push(format!("{provider_id}: {error}"));
            }
        }
    }
    anyhow::bail!("all routes failed: {}", failures.join("; "))
}

async fn provider_key(provider: &config::Provider) -> Result<String> {
    if let Some(name) = provider.api_key_env.as_deref() {
        if let Ok(value) = std::env::var(name) {
            if !value.is_empty() {
                return Ok(value);
            }
        }
    }

    if let Some(service) = provider.keychain_service.as_deref() {
        let account = provider
            .keychain_account
            .as_deref()
            .unwrap_or("FlatlineProxy");
        let output = tokio::process::Command::new("/usr/bin/security")
            .args(["find-generic-password", "-w", "-s", service, "-a", account])
            .output()
            .await
            .with_context(|| format!("read Keychain credential for provider {}", provider.id))?;
        if output.status.success() {
            let value = String::from_utf8(output.stdout)?.trim().to_owned();
            if !value.is_empty() {
                return Ok(value);
            }
        }
        anyhow::bail!(
            "Keychain credential unavailable for provider {}",
            provider.id
        );
    }

    anyhow::bail!("provider {} has no available credential", provider.id)
}

fn retryable(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 401 | 402 | 408 | 409 | 429) || status.is_server_error()
}

fn backoff_retryable(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 409 | 429) || status.is_server_error()
}

async fn send_with_backoff(
    template: reqwest::RequestBuilder,
    provider_id: &str,
    trace_id: &str,
) -> Result<reqwest::Response> {
    const MAX_ATTEMPTS: usize = 4;

    for attempt in 0..MAX_ATTEMPTS {
        let request = template
            .try_clone()
            .context("upstream request body cannot be retried")?;
        match request.send().await {
            Ok(response) if backoff_retryable(response.status()) && attempt + 1 < MAX_ATTEMPTS => {
                let delay = response
                    .headers()
                    .get(header::RETRY_AFTER)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok())
                    .map(Duration::from_secs)
                    .unwrap_or_else(|| Duration::from_secs(1_u64 << attempt))
                    .min(Duration::from_secs(30));
                warn!(
                    %trace_id,
                    %provider_id,
                    attempt = attempt + 1,
                    status = %response.status(),
                    ?delay,
                    "retrying transient upstream response"
                );
                tokio::time::sleep(delay).await;
            }
            Ok(response) => return Ok(response),
            Err(error) if attempt + 1 < MAX_ATTEMPTS => {
                let delay = Duration::from_secs(1_u64 << attempt);
                warn!(
                    %trace_id,
                    %provider_id,
                    attempt = attempt + 1,
                    timeout = error.is_timeout(),
                    connect = error.is_connect(),
                    ?delay,
                    error = ?error,
                    "retrying upstream transport failure"
                );
                tokio::time::sleep(delay).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
    unreachable!("retry loop always returns on its final attempt")
}

fn redacted_headers(headers: &HeaderMap) -> Value {
    let mut output = serde_json::Map::new();
    for (name, value) in headers {
        let key = name.as_str();
        let sensitive = matches!(
            key,
            "authorization"
                | "proxy-authorization"
                | "x-api-key"
                | "cookie"
                | "set-cookie"
                | "x-oai-attestation"
                | "x-codex-turn-state"
        );
        output.insert(
            key.to_owned(),
            if sensitive {
                Value::String("[REDACTED]".to_owned())
            } else {
                Value::String(value.to_str().unwrap_or("[NON-UTF8]").to_owned())
            },
        );
    }
    Value::Object(output)
}

async fn record_usage(log: &RwLock<VecDeque<UsageEvent>>, event: UsageEvent) {
    let mut log = log.write().await;
    if log.len() == 1000 {
        log.pop_front();
    }
    log.push_back(event);
}

async fn get_config(State(state): State<AppState>) -> Json<Config> {
    Json(state.config.read().await.clone())
}

async fn put_config(
    State(state): State<AppState>,
    Json(config): Json<Config>,
) -> impl IntoResponse {
    if let Err(error) = validate_config(&config) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": error})));
    }
    if let Err(error) = state.store.save(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": error.to_string()})),
        );
    }
    *state.config.write().await = config;
    (StatusCode::OK, Json(json!({"ok": true})))
}

fn validate_config(config: &Config) -> Result<(), String> {
    let mut ids = std::collections::HashSet::new();
    for provider in &config.providers {
        if !ids.insert(&provider.id) {
            return Err(format!("duplicate provider id: {}", provider.id));
        }
        if !(provider.base_url.starts_with("https://")
            || provider.base_url.starts_with("http://127.0.0.1")
            || provider.base_url.starts_with("http://localhost"))
        {
            return Err(format!(
                "provider {} must use HTTPS (loopback HTTP is allowed)",
                provider.id
            ));
        }
        if provider.keychain_account.is_some() && provider.keychain_service.is_none() {
            return Err(format!(
                "provider {} has keychain_account without keychain_service",
                provider.id
            ));
        }
    }
    for (model, policy) in &config.models {
        if policy.routes.is_empty() {
            return Err(format!("model {model} has no routes"));
        }
        for route in &policy.routes {
            if !ids.contains(&route.provider) {
                return Err(format!(
                    "model {model} references missing provider {}",
                    route.provider
                ));
            }
            if let Some(provider) = config.providers.iter().find(|p| p.id == route.provider) {
                if !provider.allowed_upstream_model_prefixes.is_empty()
                    && !provider
                        .allowed_upstream_model_prefixes
                        .iter()
                        .any(|prefix| route.upstream_model.starts_with(prefix))
                {
                    return Err(format!(
                        "provider {} does not allow upstream model {}",
                        provider.id, route.upstream_model
                    ));
                }
            }
        }
    }
    Ok(())
}

async fn get_usage(State(state): State<AppState>) -> Json<Vec<UsageEvent>> {
    Json(state.usage.read().await.iter().rev().cloned().collect())
}

async fn dashboard() -> Html<&'static str> {
    Html(include_str!("dashboard.html"))
}
