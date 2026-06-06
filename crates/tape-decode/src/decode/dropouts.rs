use super::*;

fn find_dropouts_rf(
    env: &[f32],
    start_rf: usize,
    end_rf: usize,
    threshold: f32,
    hysteresis: f32,
    merge_threshold: isize,
) -> Vec<(usize, isize)> {
    let down_thresh = threshold;
    let up_thresh = threshold * hysteresis;
    // Each dropout is (start, end); end == -1 marks one still open. Only the
    // most recent dropout is ever extended, so the open one is always the last.
    let mut dropouts: Vec<(usize, isize)> = Vec::new();

    for i in start_rf..end_rf {
        let value = env[i];

        if value <= down_thresh {
            let should_start = match dropouts.last() {
                None => true,
                Some(&(_, end)) => end != -1 && (i as isize - end) > merge_threshold,
            };
            if should_start {
                dropouts.push((i, -1));
            }
        } else if value >= up_thresh {
            if let Some(last) = dropouts.last_mut() {
                if last.1 == -1 {
                    last.1 = i as isize;
                }
            }
        }
    }

    if let Some(last) = dropouts.last_mut() {
        if last.1 == -1 {
            last.1 = end_rf as isize;
        }
    }

    dropouts
}

fn map_dropouts_rf_to_tbc(
    errlist: &[(usize, isize)],
    start_line_idx: usize,
    end_line_idx: usize,
    linelocs: &[f64],
    outlinelen: usize,
    lineoffset: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut rv_lines = Vec::new();
    let mut rv_starts = Vec::new();
    let mut rv_ends = Vec::new();

    let mut line_idx = start_line_idx;
    let mut line_start_rf = linelocs[line_idx];
    let mut line_end_rf = linelocs[line_idx + 1];

    for &(start_rf, end_rf) in errlist {
        let start_rf = start_rf as f64;
        let end_rf = end_rf as f64;

        while line_idx < end_line_idx {
            let line_len = line_end_rf - line_start_rf;
            if (start_rf >= line_start_rf || line_idx == start_line_idx)
                && start_rf < line_end_rf
                && line_len > 0.0
            {
                rv_lines.push(line_idx - lineoffset);

                let start_rf_linepos = start_rf - line_start_rf;
                let start_linepos =
                    ((start_rf_linepos / line_len) * outlinelen as f64).floor() as usize;
                rv_starts.push(start_linepos);
                break;
            }

            line_idx += 1;
            line_start_rf = linelocs[line_idx];
            line_end_rf = linelocs[line_idx + 1];
        }

        while line_idx < end_line_idx {
            let line_len = line_end_rf - line_start_rf;
            if end_rf < line_end_rf && line_len > 0.0 {
                let end_rf_linepos = end_rf - line_start_rf;
                let end_linepos = ((end_rf_linepos / line_len) * outlinelen as f64).ceil() as usize;
                rv_ends.push(outlinelen.min(end_linepos));
                break;
            }

            rv_ends.push(outlinelen);
            line_idx += 1;

            if line_idx < end_line_idx {
                line_start_rf = linelocs[line_idx];
                line_end_rf = linelocs[line_idx + 1];

                rv_starts.push(0);
                rv_lines.push(line_idx - lineoffset);
            }
        }
    }

    (rv_lines, rv_starts, rv_ends)
}

pub(crate) fn detect_dropouts_rf(
    spec: &DecoderSpec,
    field: &DecodedField,
    merge_threshold: isize,
    min_length: isize,
) -> Result<(f64, Vec<usize>, Vec<usize>, Vec<usize>)> {
    let env: &[f32] = &field.data.video.envelope;
    let linecount = field.linecount.unwrap_or(0);
    let linelocs = field.linelocs.as_ref().context("missing linelocs")?;
    let field_average = mean_slice(env);
    let threshold = spec
        .dod_threshold_a
        .unwrap_or(field_average as f32 * spec.dod_threshold_p);

    let start_line = field.lineoffset + 1;
    let end_line = (linelocs.len() - 1).min(linecount + start_line + 1);

    let start_rf = linelocs[start_line].floor() as usize;
    let end_rf = env.len().min(linelocs[end_line].ceil() as usize);

    let mut dropouts_rf = find_dropouts_rf(
        env,
        start_rf,
        end_rf,
        threshold,
        spec.dod_hysteresis,
        merge_threshold,
    );
    dropouts_rf.retain(|(start_rf, end_rf)| *end_rf - *start_rf as isize > min_length);

    let (rv_lines, rv_starts, rv_ends) = map_dropouts_rf_to_tbc(
        &dropouts_rf,
        start_line,
        end_line,
        linelocs,
        field.outlinelen,
        field.lineoffset,
    );

    Ok((field_average, rv_lines, rv_starts, rv_ends))
}
