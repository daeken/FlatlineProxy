use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::fs;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub providers: Vec<Provider>,
    #[serde(default)]
    pub models: HashMap<String, ModelPolicy>,
    #[serde(default)]
    pub catalog_models: Vec<CatalogModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub display_name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub keychain_service: Option<String>,
    #[serde(default)]
    pub keychain_account: Option<String>,
    pub path: String,
    #[serde(default)]
    pub protocol: Protocol,
    #[serde(default)]
    pub auth: Auth,
    #[serde(default)]
    pub allowed_upstream_model_prefixes: Vec<String>,
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    #[default]
    Responses,
    ChatCompletions,
    AnthropicMessages,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Auth {
    #[default]
    Bearer,
    AnthropicKey,
    IncomingOpenAi,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelPolicy {
    pub routes: Vec<Route>,
    #[serde(default = "default_cache_window")]
    pub cache_residency_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Route {
    pub provider: String,
    pub upstream_model: String,
    #[serde(default)]
    pub input_cost_per_million: f64,
    #[serde(default)]
    pub cached_input_cost_per_million: Option<f64>,
    #[serde(default)]
    pub output_cost_per_million: f64,
    #[serde(default)]
    pub priority: i32,
}

fn default_cache_window() -> u64 {
    300
}

#[derive(Clone)]
pub struct ConfigStore {
    path: PathBuf,
}

impl ConfigStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub async fn load(&self) -> Result<Config> {
        if !self.path.exists() {
            let config = Config::default();
            self.save(&config).await?;
            return Ok(config);
        }
        let bytes = fs::read(&self.path).await.context("read config")?;
        serde_json::from_slice(&bytes).context("parse config")
    }

    pub async fn save(&self, config: &Config) -> Result<()> {
        if let Some(parent) = self.path.parent().filter(|p| *p != Path::new("")) {
            fs::create_dir_all(parent)
                .await
                .context("create config directory")?;
        }
        let bytes = serde_json::to_vec_pretty(config)?;
        let temporary = self.path.with_extension("json.tmp");
        fs::write(&temporary, bytes)
            .await
            .context("write temporary config")?;
        fs::rename(temporary, &self.path)
            .await
            .context("replace config")
    }
}
