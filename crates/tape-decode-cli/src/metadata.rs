//! TBC sidecar metadata: the structures serialized into the `.tbc.json`
//! document and the conversion from a decoder's per-run metadata.

use serde::{Deserialize, Serialize};
use tape_decode::{DecoderMetadata, FieldInfoEntry};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TbcMetadata {
    pcm_audio_parameters: PcmAudioParameters,
    video_parameters: VideoParameters,
}

/// Full metadata, including the per-field array, as read back from a sidecar.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TbcMetadataFull {
    pub(crate) fields: Vec<FieldInfoEntry>,
    pub(crate) pcm_audio_parameters: PcmAudioParameters,
    pub(crate) video_parameters: VideoParameters,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PcmAudioParameters {
    pub(crate) bits: usize,
    pub(crate) is_little_endian: bool,
    pub(crate) is_signed: bool,
    pub(crate) sample_rate: usize,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct VideoParameters {
    pub(crate) number_of_sequential_fields: usize,
    pub(crate) os_info: String,
    pub(crate) git_branch: String,
    pub(crate) git_commit: String,
    pub(crate) system: String,
    pub(crate) field_width: usize,
    pub(crate) sample_rate: f64,
    pub(crate) black_16b_ire: f64,
    pub(crate) white_16b_ire: f64,
    pub(crate) field_height: usize,
    pub(crate) colour_burst_start: i64,
    pub(crate) colour_burst_end: i64,
    pub(crate) active_video_start: i64,
    pub(crate) active_video_end: i64,
    pub(crate) tape_format: String,
}

pub(crate) fn metadata_to_tbc(metadata: &DecoderMetadata, field_count: usize) -> TbcMetadata {
    TbcMetadata {
        pcm_audio_parameters: PcmAudioParameters {
            bits: 16,
            is_little_endian: true,
            is_signed: true,
            sample_rate: 0,
        },
        video_parameters: VideoParameters {
            number_of_sequential_fields: field_count,
            os_info: String::new(),
            git_branch: "UNKNOWN".to_string(),
            git_commit: "UNKNOWN".to_string(),
            system: metadata.system.to_string(),
            field_width: metadata.field_width,
            sample_rate: metadata.sample_rate,
            black_16b_ire: metadata.black_16b_ire,
            white_16b_ire: metadata.white_16b_ire,
            field_height: metadata.field_height,
            colour_burst_start: metadata.colour_burst_start,
            colour_burst_end: metadata.colour_burst_end,
            active_video_start: metadata.active_video_start,
            active_video_end: metadata.active_video_end,
            tape_format: "TAPE".to_string(),
        },
    }
}
