"""Notebook-compatible data loader with stricter trial and timing handling."""

from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


RETRO_CONDITIONS = {3, 4, 5, 6}

TRIAL_FIELDS = [
    "rawTrialNo",
    "Condition",
    "StopCondition",
    "START_TRIAL",
    "FIXATE_ON",
    "FIXATE_ACQUIRED",
    "SAMPLES_ON",
    "DELAY2_START",
    "CUE2_ON",
    "DELAY3_START",
    "WHEEL_ON",
    "TARGET_FIX",
    "FEEDBACK_ON",
    "END_TRIAL",
    "CORRECT_TRIAL",
    "ColorID",
    "distractor_index",
    "ColorTarget",
    "ColorDist",
    "LABthetaTarget",
    "LABthetaDist",
    "IsUpperSample",
    "UpperSampleLocation",
    "LowerSampleLocation",
    "TargetLocation",
    "TargetTheta",
    "DistTheta",
]

_EXTRA_TRIAL_FIELDS = [
    "is_one_sample_displayed",
    "SAMPLES_ON_diode",
    "CUE2_ON_diode",
    "WHEEL_ON_diode",
]

_EPOCH_ALIASES = {
    "SAMPLES_ON": ("SAMPLES_ON_diode", "SAMPLES_ON"),
    "CUE2_ON": ("CUE2_ON_diode", "CUE2_ON"),
    "WHEEL_ON": ("WHEEL_ON_diode", "WHEEL_ON"),
}


class ChannelUnitId(str):
    """String-like unit id that still exposes its channel to old notebook code."""

    def __new__(cls, channel, sorted_unit_id=None):
        channel = int(channel)
        if sorted_unit_id is None:
            text = f"chan{channel}"
        else:
            text = f"chan{channel}_u{int(sorted_unit_id)}"
        obj = super().__new__(cls, text)
        obj.channel = channel
        obj.sorted_unit_id = None if sorted_unit_id is None else int(sorted_unit_id)
        return obj

    def replace(self, old, new, count=-1):
        if old == "chan" and new == "" and count == -1:
            return str(self.channel)
        return super().replace(old, new, count)


def load_session(session, data_dir="Data"):
    """Load correct retro two-item trials and sorted single-unit spike times."""
    session = str(session)
    session_dir = Path(data_dir) / session

    trials = _load_correct_retro_trials(session_dir)
    units = _load_units(session_dir)

    return {
        "session": session,
        "session_dir": session_dir,
        "trials": trials,
        "units": units,
        "missing": {
            "session": not session_dir.exists(),
            "behavior": not _behavior_file(session_dir).exists(),
            "spikes": not _spikes_dir(session_dir).exists(),
        },
    }


def align_spikes(session_data, epoch, window=(-0.5, 1.0)):
    """Align each unit's spike times to one trial event."""
    trials = session_data["trials"]
    units = session_data["units"]
    window = tuple(float(x) for x in window)
    epoch_field = None if not trials else _resolve_epoch_field(trials, epoch)

    aligned_units = {}
    for unit in units:
        unit_trials = []
        for trial in trials:
            event_time = float(trial.get(epoch_field, np.nan))
            unit_trials.append(_align_one_trial(unit["spike_times"], event_time, window))
        aligned_units[unit["unit_id"]] = unit_trials

    return {
        "session": session_data.get("session"),
        "epoch": epoch,
        "resolved_epoch": epoch_field,
        "window": window,
        "unit_ids": [unit["unit_id"] for unit in units],
        "trial_ids": [trial["trial_index"] for trial in trials],
        "units": aligned_units,
    }


def get_spike_counts(aligned, window_size=0.100, step=0.025):
    """Count aligned spikes in sliding half-open windows."""
    window_size = float(window_size)
    step = float(step)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    start, end = aligned["window"]
    bin_starts = _make_bin_starts(start, end, window_size, step)
    unit_ids = list(aligned["unit_ids"])
    trial_ids = list(aligned["trial_ids"])

    counts = np.zeros((len(unit_ids), len(trial_ids), len(bin_starts)), dtype=int)
    for unit_index, unit_id in enumerate(unit_ids):
        for trial_index, spikes in enumerate(aligned["units"][unit_id]):
            counts[unit_index, trial_index, :] = _count_one_trial(spikes, bin_starts, window_size)

    return {
        "counts": counts,
        "bin_starts": bin_starts,
        "bin_centers": bin_starts + window_size / 2,
        "unit_ids": unit_ids,
        "trial_ids": trial_ids,
    }


def _load_correct_retro_trials(session_dir):
    bhv = _load_behavior(session_dir)
    if bhv is None:
        return []
    matlab_trials = np.atleast_1d(bhv.Trials)

    trials = []
    for matlab_index, matlab_trial in enumerate(matlab_trials, start=1):
        if _is_correct_retro_two_item_trial(matlab_trial):
            trial = _trial_to_dict(matlab_trial)
            trial["trial_index"] = matlab_index
            trials.append(trial)
    return trials


def _load_behavior(session_dir):
    behavior_file = _behavior_file(session_dir)
    if not behavior_file.exists():
        return None
    return loadmat(behavior_file, struct_as_record=False, squeeze_me=True)["bhv"]


def _is_correct_retro_two_item_trial(trial):
    is_correct = int(trial.StopCondition) == 1
    is_retro = int(trial.Condition) in RETRO_CONDITIONS
    is_two_item = _is_explicit_zero(getattr(trial, "is_one_sample_displayed", np.nan))
    return is_correct and is_retro and is_two_item


def _trial_to_dict(matlab_trial):
    trial = {}
    for field in TRIAL_FIELDS + _EXTRA_TRIAL_FIELDS:
        if hasattr(matlab_trial, field):
            trial[field] = _as_python_value(getattr(matlab_trial, field))
    return trial


def _load_units(session_dir):
    units = []
    for spike_file in _find_spike_files(session_dir):
        channel = _channel_from_file(spike_file)
        unit_spike_times = _split_sorted_units(spike_file)
        if len(unit_spike_times) == 1:
            sorted_unit_id, spike_times = unit_spike_times[0]
            units.append(_make_unit_dict(channel, sorted_unit_id, spike_file, spike_times, single_unit_file=True))
            continue
        for sorted_unit_id, spike_times in unit_spike_times:
            units.append(_make_unit_dict(channel, sorted_unit_id, spike_file, spike_times))
    return units


def _make_unit_dict(channel, sorted_unit_id, spike_file, spike_times, single_unit_file=False):
    unit_id = ChannelUnitId(channel, None if single_unit_file else sorted_unit_id)
    return {
        "unit_id": unit_id,
        "channel": int(channel),
        "sorted_unit_id": int(sorted_unit_id),
        "file": spike_file,
        "spike_times": spike_times,
    }


def _find_spike_files(session_dir):
    spikes_dir = _spikes_dir(session_dir)
    if not spikes_dir.exists():
        return []

    files = []
    for spike_file in spikes_dir.glob("selWM_001_chan*_4sd-srt.mat"):
        if "-mua" not in spike_file.name:
            files.append(spike_file)
    return sorted(files, key=_channel_from_file)


def _behavior_file(session_dir):
    return session_dir / "bhv" / "bhv.mat"


def _spikes_dir(session_dir):
    return session_dir / "spikes"


def _channel_from_file(spike_file):
    match = re.search(r"_chan(\d+)_", spike_file.name)
    if not match:
        raise ValueError(f"Could not read channel from {spike_file.name}")
    return int(match.group(1))


def _split_sorted_units(spike_file):
    mat = loadmat(spike_file, squeeze_me=True, struct_as_record=False)
    all_spike_times = np.ravel(mat["ts"]).astype(float)
    if all_spike_times.size == 0:
        return []

    if "id" in mat and np.ravel(mat["id"]).size:
        all_unit_ids = np.ravel(mat["id"]).astype(int)
    else:
        all_unit_ids = np.ones(all_spike_times.shape[0], dtype=int)

    units = []
    for sorted_unit_id in sorted(np.unique(all_unit_ids)):
        spike_times = np.sort(all_spike_times[all_unit_ids == sorted_unit_id])
        if spike_times.size == 0:
            continue
        units.append((int(sorted_unit_id), spike_times))
    return units


def _resolve_epoch_field(trials, epoch):
    if not trials:
        raise ValueError("Cannot align spikes because there are no retained trials")

    candidate_fields = _EPOCH_ALIASES.get(epoch, (epoch,))
    known_fields = {field for trial in trials for field in trial}
    available_fields = [field for field in candidate_fields if field in known_fields]
    if not available_fields:
        raise ValueError(f"Unknown epoch: {epoch}")

    for field in available_fields:
        if any(np.isfinite(float(trial.get(field, np.nan))) for trial in trials):
            return field

    choices = ", ".join(available_fields)
    raise ValueError(f"Epoch {epoch} has no finite timestamps in retained trials ({choices})")


def _align_one_trial(spike_times, event_time, window):
    if not np.isfinite(event_time):
        return np.array([], dtype=float)

    relative_times = spike_times - event_time
    keep = (relative_times >= window[0]) & (relative_times < window[1])
    return relative_times[keep]


def _make_bin_starts(start, end, window_size, step):
    last_start = end - window_size
    if last_start < start:
        return np.array([], dtype=float)
    n_bins = int(np.floor((last_start - start) / step + 1e-9)) + 1
    return start + np.arange(n_bins, dtype=float) * step


def _count_one_trial(spikes, bin_starts, window_size):
    if spikes.size == 0 or bin_starts.size == 0:
        return np.zeros(bin_starts.shape[0], dtype=int)
    left = np.searchsorted(spikes, bin_starts, side="left")
    right = np.searchsorted(spikes, bin_starts + window_size, side="left")
    return (right - left).astype(int)


def _as_python_value(value):
    array = np.asarray(value)
    if array.shape == ():
        value = array.item()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    return np.asarray(value)


def _is_explicit_zero(value):
    array = np.asarray(value, dtype=float)
    if array.shape != ():
        return False
    scalar = float(array.item())
    return np.isfinite(scalar) and scalar == 0.0


def color_ids_to_groups(color_ids, n_groups=4, n_colors=64):
    """Convert color ids on a 1..64 wheel into discrete class labels."""
    n_groups = int(n_groups)
    n_colors = int(n_colors)
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if n_groups > n_colors:
        raise ValueError("n_groups cannot exceed n_colors")

    color_ids = np.asarray(color_ids, dtype=int)
    if np.any(color_ids < 1) or np.any(color_ids > n_colors):
        raise ValueError(f"color_ids must be between 1 and {n_colors}")

    return ((color_ids - 1) * n_groups // n_colors).astype(int)

def selected_condition_masks(trials,N_COLOR_GROUPS):
    selected_ids = np.asarray([trial["ColorID"] for trial in trials], dtype=int)
    selected_groups = color_ids_to_groups(selected_ids, n_groups=N_COLOR_GROUPS)
    is_selected_upper = np.asarray([trial["IsUpperSample"] == 1 for trial in trials], dtype=bool)

    masks = []
    for group_id in range(N_COLOR_GROUPS):
        masks.append(is_selected_upper & (selected_groups == group_id))
    for group_id in range(N_COLOR_GROUPS):
        masks.append((~is_selected_upper) & (selected_groups == group_id))
    return masks

def load_region_channels(session_id, region, data_dir):
    label_file = data_dir / session_id / "labels" / f"{region}.txt"
    if not label_file.exists():
        return []
    text = label_file.read_text().strip()
    return [] if text == "" else [int(x) for x in text.split()]

