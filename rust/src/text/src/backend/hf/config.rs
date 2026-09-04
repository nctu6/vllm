// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;

use crate::error::{Error, Result};
use crate::normalize_top_k;

/// Minimal subset of `tokenizer_config.json` needed by chat/EOS handling.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct HfTokenizerConfig {
    #[serde(flatten)]
    pub special_tokens: HfSpecialTokens,
    pub chat_template: Option<String>,
    /// The `tokenizer_class` field from HuggingFace tokenizer configs. Some
    /// tiktoken-based models (e.g. DeepSeek, Kimi K2) set this to a value
    /// containing "Tiktoken" which can be used as a hint for backend
    /// selection.
    pub tokenizer_class: Option<String>,
}

/// Hugging Face named special tokens may be serialized as a string or an
/// object carrying the token content.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum NamedSpecialToken {
    Text(String),
    WithContent { content: String },
}

impl Serialize for NamedSpecialToken {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl From<NamedSpecialToken> for String {
    fn from(value: NamedSpecialToken) -> Self {
        match value {
            NamedSpecialToken::Text(string) => string,
            NamedSpecialToken::WithContent { content } => content,
        }
    }
}

impl NamedSpecialToken {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(value) => value,
            Self::WithContent { content } => content,
        }
    }
}

/// Minimal set of special-token entries needed by chat/EOS handling.
#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct HfSpecialTokens {
    pub bos_token: Option<NamedSpecialToken>,
    pub eos_token: Option<NamedSpecialToken>,
    pub unk_token: Option<NamedSpecialToken>,
    pub pad_token: Option<NamedSpecialToken>,
}

impl HfSpecialTokens {
    /// Returns true if we don't discover any special tokens in the config.
    pub fn is_empty(&self) -> bool {
        self.bos_token.is_none()
            && self.eos_token.is_none()
            && self.unk_token.is_none()
            && self.pad_token.is_none()
    }
}

/// Minimal subset of `config.json` (the model's main HF config).
///
/// This intentionally supports only the two layouts we currently care about in
/// the Rust frontend:
/// - pure text models that keep text metadata at the top level
/// - composite models that expose a single nested `text_config`
///
/// We do not support additional entry points such as `decoder`, `generator`, or
/// `text_encoder`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    model_type: Option<String>,
    vocab_size: Option<u32>,
    eos_token_id: Option<OneOrManyTokenIds>,
    num_experts: Option<OneOrManyExpertCount>,
    moe_num_experts: Option<OneOrManyExpertCount>,
    n_routed_experts: Option<OneOrManyExpertCount>,
    num_local_experts: Option<OneOrManyExpertCount>,
    block_configs: Vec<BlockConfig>,
    text_config: Option<Box<ModelConfig>>,
}

/// Minimal subset of `generation_config.json`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(super) struct GenerationConfig {
    pub eos_token_id: Option<OneOrManyTokenIds>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(deserialize_with = "deserialize_top_k")]
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub max_new_tokens: Option<u32>,
}

/// Deserialize a generation-config `top_k` into the text sampling model.
///
/// Null, `-1`, and `0` all disable top-k sampling; positive limits are
/// preserved.
fn deserialize_top_k<'de, D>(deserializer: D) -> std::result::Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<i64>::deserialize(deserializer)?
        .map(normalize_top_k)
        .transpose()
        .map(Option::flatten)
        .map_err(serde::de::Error::custom)
}

/// HF generation configs allow either one EOS id or a list of EOS ids.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrManyTokenIds {
    pub(super) fn as_slice(&self) -> &[u32] {
        match self {
            Self::One(id) => std::slice::from_ref(id),
            Self::Many(ids) => ids.as_slice(),
        }
    }

    pub(super) fn into_set(self) -> BTreeSet<u32> {
        self.as_slice().iter().copied().collect()
    }
}

/// Hugging Face configs may expose the expert count either as one integer or
/// as a list of repeated integers.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(super) enum OneOrManyExpertCount {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrManyExpertCount {
    fn first_value(&self) -> u32 {
        match self {
            Self::One(value) => *value,
            // Python currently takes the first value for list[int] expert
            // counts in remote-code configs.
            Self::Many(values) => values.first().copied().unwrap_or(0),
        }
    }
}

/// Heterogeneous block-level MoE metadata used as a fallback when no top-level
/// expert-count field is available.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(super) struct BlockConfig {
    pub block_type: String,
    pub n_routed_experts: u32,
}

impl ModelConfig {
    /// Return the config that the Rust frontend treats as the text/LLM config.
    ///
    /// This is deliberately narrower than Python/transformers: we only support
    /// either the top-level config itself or a single nested `text_config`.
    fn effective_text_config(&self) -> &Self {
        self.text_config.as_deref().unwrap_or(self)
    }

    /// Return the effective Hugging Face `model_type` used by the Rust
    /// frontend.
    ///
    /// This follows the same simplified text-config selection as the rest of
    /// this type: the top-level config wins, otherwise a single nested
    /// `text_config` may provide the value.
    pub fn model_type(&self) -> Option<&str> {
        self.model_type.as_deref().or_else(|| self.text_config.as_deref()?.model_type())
    }

    /// Return the effective model vocabulary size, following the same
    /// simplified text-config selection as `model_type`.
    ///
    /// Composite models (e.g. `Gemma3ForConditionalGeneration`) keep
    /// `vocab_size` in a nested `text_config` rather than at the top level, so
    /// we walk into it when the top-level value is absent. Returns `None` when
    /// no `vocab_size` can be resolved; callers may fall back to the tokenizer
    /// vocabulary size in that case.
    pub fn vocab_size(&self) -> Option<u32> {
        self.vocab_size.or_else(|| self.text_config.as_deref()?.vocab_size())
    }

    /// Return the effective model-side EOS token ids, following the same
    /// simplified text-config selection as `vocab_size`.
    pub(super) fn eos_token_ids(&self) -> &[u32] {
        if let Some(eos_token_id) = self.eos_token_id.as_ref() {
            eos_token_id.as_slice()
        } else if let Some(text_config) = self.text_config.as_deref() {
            text_config.eos_token_ids()
        } else {
            &[]
        }
    }

    /// Match Python's current expert-count priority on the selected text
    /// config.
    ///
    /// The only intentional simplification here is how we pick the text config:
    /// Rust only looks at the top level or `text_config`, not the broader
    /// transformers composite-config surface.
    fn num_experts_from_block_configs(&self) -> u32 {
        self.effective_text_config()
            .block_configs
            .iter()
            .filter(|block| block.block_type == "moe")
            .map(|block| block.n_routed_experts)
            .max()
            .unwrap_or(0)
    }

    pub(super) fn num_experts(&self) -> u32 {
        let config = self.effective_text_config();
        let direct = [
            config.num_experts.as_ref(),
            config.moe_num_experts.as_ref(),
            config.n_routed_experts.as_ref(),
            config.num_local_experts.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(OneOrManyExpertCount::first_value)
        .next()
        .unwrap_or(0);

        if direct > 0 {
            direct
        } else {
            self.num_experts_from_block_configs()
        }
    }

    pub(super) fn is_moe(&self) -> bool {
        self.num_experts() > 0
    }
}

/// Load the tokenizer-side EOS metadata if a config file is present.
pub fn load_tokenizer_config(path: Option<&Path>) -> Result<HfTokenizerConfig> {
    read_json_file(path)
}

/// Load the generation-side EOS metadata if a config file is present.
pub(super) fn load_generation_config(path: Option<&Path>) -> Result<GenerationConfig> {
    read_json_file(path)
}

/// Load the model-side config (`config.json`) if present.
pub fn load_model_config(path: Option<&Path>) -> Result<ModelConfig> {
    read_json_file(path)
}

fn read_json_file<T>(path: Option<&Path>) -> Result<T>
where
    T: for<'de> Deserialize<'de> + Default,
{
    let Some(path) = path else {
        return Ok(T::default());
    };
    let content = fs::read_to_string(path).map_err(|error| {
        Error::Tokenizer(format!(
            "failed to read {}: {}",
            path.display(),
            error.as_report()
        ))
    })?;
    serde_json::from_str(&content).map_err(|error| {
        Error::Tokenizer(format!(
            "failed to parse {}: {}",
            path.display(),
            error.as_report()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{GenerationConfig, ModelConfig};

    #[test]
    fn generation_config_normalizes_vllm_top_k_values() {
        for (input, expected) in [
            (r#"{"top_k":null}"#, None),
            (r#"{"top_k":-1}"#, None),
            (r#"{"top_k":0}"#, None),
            (r#"{"top_k":20}"#, Some(20)),
        ] {
            let config: GenerationConfig = serde_json::from_str(input).unwrap();
            assert_eq!(config.top_k, expected, "input={input}");
        }

        for input in [r#"{"top_k":-2}"#, r#"{"top_k":4294967296}"#] {
            assert!(serde_json::from_str::<GenerationConfig>(input).is_err());
        }
    }

    #[test]
    fn model_config_detects_moe_from_named_expert_fields() {
        let field_names = [
            "num_experts",
            "moe_num_experts",
            "n_routed_experts",
            "num_local_experts",
        ];

        for field_name in field_names {
            let config: ModelConfig =
                serde_json::from_str(&format!(r#"{{"{field_name}": 64}}"#)).unwrap();
            assert_eq!(config.num_experts(), 64, "field_name={field_name}");
            assert!(config.is_moe(), "field_name={field_name}");
        }
    }

    #[test]
    fn model_config_uses_first_value_for_list_expert_counts() {
        let config: ModelConfig = serde_json::from_str(r#"{"num_experts":[16,16]}"#).unwrap();

        assert_eq!(config.num_experts(), 16);
        assert!(config.is_moe());
    }

    #[test]
    fn model_config_falls_back_to_block_configs_maximum() {
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "block_configs": [
                    {"block_type":"attention","n_routed_experts":9},
                    {"block_type":"moe","n_routed_experts":32},
                    {"block_type":"moe","n_routed_experts":64}
                ]
            }"#,
        )
        .unwrap();

        assert_eq!(config.num_experts(), 64);
        assert!(config.is_moe());
    }

    #[test]
    fn model_config_prefers_nested_text_config_like_python_hf_text_config() {
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "model_type": "top_level",
                "num_experts": 64,
                "text_config": {
                    "model_type": "nested",
                    "num_local_experts": 8
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.num_experts(), 8);
        assert_eq!(config.model_type(), Some("top_level"));
        assert!(config.is_moe());
    }

    #[test]
    fn model_config_uses_nested_vocab_size_when_top_level_is_absent() {
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "text_config": {
                    "vocab_size": 151936
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.vocab_size(), Some(151936));
    }

    #[test]
    fn model_config_reads_nested_vocab_size_for_gemma3_composite_config() {
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "architectures": ["Gemma3ForConditionalGeneration"],
                "model_type": "gemma3",
                "eos_token_id": [1, 106],
                "text_config": {
                    "model_type": "gemma3_text",
                    "num_hidden_layers": 62,
                    "vocab_size": 262208
                },
                "vision_config": {
                    "model_type": "siglip_vision_model"
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.vocab_size(), Some(262208));
    }

    #[test]
    fn model_config_reads_top_level_eos_token_ids() {
        let single: ModelConfig = serde_json::from_str(r#"{"eos_token_id":151645}"#).unwrap();
        let many: ModelConfig =
            serde_json::from_str(r#"{"eos_token_id":[128001,128008,128009]}"#).unwrap();
        let null: ModelConfig = serde_json::from_str(r#"{"eos_token_id":null}"#).unwrap();

        assert_eq!(single.eos_token_ids(), &[151645]);
        assert_eq!(many.eos_token_ids(), &[128001, 128008, 128009]);
        assert!(null.eos_token_ids().is_empty());
    }

    #[test]
    fn model_config_uses_nested_eos_token_ids_when_top_level_is_absent() {
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "text_config": {
                    "eos_token_id": [59246, 59253, 59255]
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.eos_token_ids(), &[59246, 59253, 59255]);
    }

    #[test]
    fn model_config_returns_none_when_vocab_size_is_missing() {
        let config: ModelConfig = serde_json::from_str(r#"{}"#).unwrap();

        assert_eq!(config.vocab_size(), None);
    }

    #[test]
    fn model_config_returns_none_when_nested_text_config_omits_vocab_size() {
        // Trimmed Gemma-3 export whose `text_config` drops `vocab_size`; Python
        // fills it from model-class defaults while the Rust frontend must fall
        // back to the tokenizer vocabulary size.
        let config: ModelConfig = serde_json::from_str(
            r#"{
                "model_type": "gemma3",
                "text_config": {
                    "model_type": "gemma3_text",
                    "num_hidden_layers": 62
                }
            }"#,
        )
        .unwrap();

        assert_eq!(config.vocab_size(), None);
    }

    #[test]
    fn model_config_defaults_to_non_moe_when_no_expert_metadata_exists() {
        let config: ModelConfig = serde_json::from_str(r#"{}"#).unwrap();

        assert_eq!(config.num_experts(), 0);
        assert!(!config.is_moe());
    }
}
