import base64
import io
import os
import random
import subprocess
import tempfile
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Suppress the torch.nn.utils.weight_norm deprecation warning from Dia's internal code
warnings.filterwarnings(
    "ignore",
    message=r".*weight_norm.*deprecated.*",
    category=FutureWarning,
)

# Set HF token from environment if available (silences "unauthenticated requests" warning)
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    os.environ["HF_TOKEN"] = _hf_token

from dia.model import Dia

# Import GCS voice manager
try:
    from gcs_voices import GCSVoiceManager, DEFAULT_VOICE_PRESETS
    _gcs_available = True
except ImportError:
    _gcs_available = False
    DEFAULT_VOICE_PRESETS = {}


class VoiceMode(str, Enum):
    """Voice selection mode for TTS generation."""
    PREDEFINED = "predefined"  # GCS-stored predefined voice presets
    SESSION = "session"        # Session-based (uploaded audio prompts)
    SEED = "seed"              # Deterministic seed-based (no voice cloning)


def _pick_device() -> torch.device:
    forced = os.getenv("DIA_DEVICE", "").strip()
    if forced:
        return torch.device(forced)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _pick_dtype(device: torch.device) -> str:
    if device.type == "cuda":
        return os.getenv("DIA_COMPUTE_DTYPE", "float16")
    # MPS/CPU tends to be more stable with float32
    return os.getenv("DIA_COMPUTE_DTYPE", "float32")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_speaker_tags(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    # If caller already includes Dia speaker tags, keep as-is.
    if "[S1]" in t or "[S2]" in t:
        return t

    # Minimal formatting: prefix with [S1].
    return f"[S1] {t}"


def _extract_speaker_tag(text: str) -> str:
    """Extract speaker tag from text (returns 'S1', 'S2', or 'S1' as default)."""
    stripped = (text or "").strip()
    if stripped.startswith("[S2]"):
        return "S2"
    return "S1"


def _wav_to_mp3_bytes(wav_bytes: bytes, sample_rate: int, speed: float) -> bytes:
    # Uses ffmpeg if available. This keeps the Go backend unchanged (expects MP3 bytes).
    # On macOS: `brew install ffmpeg`
    # In Docker: install ffmpeg via apt.
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "in.wav")
        mp3_path = os.path.join(tmp, "out.mp3")

        # Write wav bytes to disk
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            wav_path,
            "-map_metadata",
            "-1",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            os.getenv("DIA_MP3_BITRATE", "128k"),
            # Make MP3 concatenation safer by avoiding Xing/VBR headers.
            "-write_xing",
            "0",
            # Avoid writing ID3v1 tags.
            "-write_id3v1",
            "0",
            mp3_path,
        ]

        # Speed control via time-stretching.
        # atempo supports 0.5..2.0.
        if speed and abs(speed - 1.0) > 1e-6:
            pos = cmd.index("-ar")
            cmd.insert(pos, "-filter:a")
            cmd.insert(pos + 1, f"atempo={speed}")

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "ffmpeg is required to return mp3. Install it (macOS: brew install ffmpeg)."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg failed: {e}") from e

        with open(mp3_path, "rb") as f:
            return f.read()


class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field("default", description="Logical voice name (mapped to a seed)")
    seed: Optional[int] = Field(None, description="Seed for deterministic voice")
    format: str = Field("mp3", description="mp3|wav")

    # Playback speed / speaking pace.
    # Implemented via ffmpeg atempo during MP3 encoding.
    speed: float = Field(1.0, ge=0.5, le=2.0)

    # Optional Dia generation knobs (safe defaults)
    max_tokens: int = Field(3072, ge=256, le=4096)
    cfg_scale: float = Field(3.0, ge=1.0, le=6.0)
    temperature: float = Field(1.2, ge=0.7, le=2.5)
    top_p: float = Field(0.95, ge=0.5, le=1.0)
    cfg_filter_top_k: int = Field(45, ge=10, le=200)

    # Audio prompt for voice cloning (optional)
    # Base64-encoded audio file (WAV or MP3, 5-10 seconds recommended)
    audio_prompt: Optional[str] = Field(None, description="Base64-encoded reference audio for voice cloning")
    # Transcript of the audio prompt with speaker tags
    audio_prompt_text: Optional[str] = Field(None, description="Transcript of reference audio (e.g., '[S1] Hello world.')")

    # Session ID for cached audio prompts (use instead of audio_prompt for subsequent chunks)
    session_id: Optional[str] = Field(None, description="Session ID for cached audio prompts")

    # Hybrid API fields (unified server)
    voice_mode: Optional[VoiceMode] = Field(None, description="Voice selection mode: predefined, session, or seed")
    voice_preset: Optional[str] = Field(None, description="Predefined voice preset name (e.g., 'warm_duo_us_fm')")

    # Text splitting (for long-form content)
    split_text: Optional[bool] = Field(None, description="Enable text chunking for long content")
    chunk_size: Optional[int] = Field(None, ge=100, le=2000, description="Chunk size if splitting")


class SessionCreateRequest(BaseModel):
    """Request to create a TTS session with cached audio prompts."""
    # S1 audio prompt (required for session)
    s1_audio_prompt: str = Field(..., description="Base64-encoded reference audio for S1")
    s1_audio_prompt_text: str = Field(..., description="Transcript of S1 reference audio")
    # S2 audio prompt (optional, for dialogue)
    s2_audio_prompt: Optional[str] = Field(None, description="Base64-encoded reference audio for S2")
    s2_audio_prompt_text: Optional[str] = Field(None, description="Transcript of S2 reference audio")
    # Session TTL in seconds (default 1 hour)
    ttl_seconds: int = Field(3600, ge=60, le=86400)


class SessionResponse(BaseModel):
    """Response from session creation."""
    session_id: str
    expires_at: int  # Unix timestamp (seconds)


class PredefinedVoiceResponse(BaseModel):
    """Response for a predefined voice preset."""
    name: str
    description: Optional[str] = None
    style: Optional[str] = None
    speakers: int = 1


class PredefinedVoicesResponse(BaseModel):
    """Response listing all predefined voices."""
    voices: List[PredefinedVoiceResponse]
    count: int


@dataclass
class AudioPromptCache:
    """Cached audio prompt data for a session."""
    s1_audio: np.ndarray
    s1_text: str
    s2_audio: Optional[np.ndarray] = None
    s2_text: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 3600


@dataclass
class BootstrappedVoice:
    """Pre-generated voice prompt for consistent voice cloning."""
    voice_key: str
    audio_prompt: np.ndarray
    audio_prompt_text: str
    seed: int


@dataclass
class ModelState:
    device: torch.device
    dtype: str
    model: Dia


app = FastAPI(title="Dia TTS Unified Service", version="0.4.0")
_state: Optional[ModelState] = None
_load_error: Optional[str] = None
_loading = False
_load_lock = threading.Lock()

# Session cache for audio prompts (enables voice consistency across chunks)
_session_cache: Dict[str, AudioPromptCache] = {}
_session_lock = threading.Lock()

# Bootstrapped voices cache (pre-generated on startup for consistent voice cloning)
_bootstrapped_voices: Dict[str, BootstrappedVoice] = {}
_bootstrap_lock = threading.Lock()

# GCS Voice Manager (for predefined GCS-backed voices)
_gcs_voice_manager: Optional["GCSVoiceManager"] = None


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions from cache."""
    now = time.time()
    with _session_lock:
        expired = [
            sid for sid, cache in _session_cache.items()
            if now > cache.created_at + cache.ttl_seconds
        ]
        for sid in expired:
            del _session_cache[sid]


def _decode_audio_prompt(b64_audio: str) -> np.ndarray:
    """Decode base64 audio to numpy array."""
    audio_bytes = base64.b64decode(b64_audio)
    
    # Try to read as audio file
    try:
        with io.BytesIO(audio_bytes) as buf:
            audio_data, sr = sf.read(buf)
            # Resample to 44100 if needed
            if sr != 44100:
                # Simple resampling (for production, use librosa or scipy)
                ratio = 44100 / sr
                new_length = int(len(audio_data) * ratio)
                indices = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
            return audio_data.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to decode audio prompt: {e}")


def _init_gcs_voice_manager() -> None:
    """Initialize GCS voice manager if available."""
    global _gcs_voice_manager
    
    if not _gcs_available:
        print("GCS voices not available (gcs_voices module not found)")
        return
    
    bucket_name = os.getenv("GCS_BUCKET", "ai-podcast-voice-presets")
    try:
        _gcs_voice_manager = GCSVoiceManager(bucket_name=bucket_name)
        print(f"GCS voice manager initialized (bucket: {bucket_name})")
        # List available voices
        voices = _gcs_voice_manager.list_available_voices()
        print(f"  Available GCS voices: {voices}")
    except Exception as e:
        print(f"Failed to initialize GCS voice manager: {e}")
        _gcs_voice_manager = None


def _load_model() -> None:
    global _state, _load_error, _loading

    device = _pick_device()
    dtype = _pick_dtype(device)
    # Model weights are still pulled from Hugging Face.
    # The Python package is a fork with MPS support; the default model ID remains the same.
    model_id = os.getenv("DIA_MODEL", "nari-labs/Dia-1.6B-0626")

    try:
        model = Dia.from_pretrained(model_id, compute_dtype=dtype, device=device)
        _state = ModelState(device=device, dtype=dtype, model=model)
        
        # Initialize GCS voice manager
        _init_gcs_voice_manager()
        
        # Bootstrap default voice prompts after model load (legacy support)
        if os.getenv("DIA_BOOTSTRAP_VOICES", "true").lower() == "true":
            _bootstrap_default_voices()
    except Exception as e:
        _load_error = f"Failed to load Dia model {model_id}: {e}"
    finally:
        _loading = False


def _bootstrap_default_voices() -> None:
    """Pre-generate voice prompts for consistent voice cloning across chunks.
    
    This is legacy support for bootstrapped voices. For new deployments,
    use GCS predefined voices via /v1/voices/predefined endpoint.
    """
    global _bootstrapped_voices
    
    if _state is None:
        return
    
    if not DEFAULT_VOICE_PRESETS:
        print("No voice presets configured for bootstrapping")
        return
    
    print("Bootstrapping default voice prompts...")
    
    for voice_key, config in DEFAULT_VOICE_PRESETS.items():
        try:
            seed = config.get("seed", 12345)
            text = config.get("text", "[S1] Hello, this is a test.")
            
            _set_seed(seed)
            
            with torch.inference_mode():
                audio_np = _state.model.generate(
                    text=text,
                    max_tokens=2048,  # Shorter for prompt
                    cfg_scale=3.0,
                    temperature=1.2,
                    top_p=0.95,
                    cfg_filter_top_k=45,
                    use_torch_compile=False,
                    verbose=False,
                )
            
            if audio_np is not None and len(audio_np) > 0:
                with _bootstrap_lock:
                    _bootstrapped_voices[voice_key] = BootstrappedVoice(
                        voice_key=voice_key,
                        audio_prompt=np.asarray(audio_np, dtype=np.float32),
                        audio_prompt_text=text,
                        seed=seed,
                    )
                print(f"  ✓ Bootstrapped voice: {voice_key}")
            else:
                print(f"  ✗ Failed to bootstrap voice: {voice_key} (empty audio)")
        except Exception as e:
            print(f"  ✗ Failed to bootstrap voice {voice_key}: {e}")
    
    print(f"Voice bootstrapping complete: {len(_bootstrapped_voices)}/{len(DEFAULT_VOICE_PRESETS)}")


def _get_bootstrapped_voice(voice_key: str) -> Optional[BootstrappedVoice]:
    """Get a bootstrapped voice prompt by key."""
    with _bootstrap_lock:
        return _bootstrapped_voices.get(voice_key)


def _get_predefined_voice_audio(preset_name: str, speaker: str = "S1") -> Optional[tuple]:
    """Get audio prompt from GCS predefined voice.
    
    Returns (audio_np, audio_text) or None if not found.
    """
    if _gcs_voice_manager is None:
        return None
    
    try:
        audio_np = _gcs_voice_manager.get_voice_audio_prompt(preset_name, speaker)
        if audio_np is None:
            return None
        
        # Get the text prompt from preset config
        preset_config = DEFAULT_VOICE_PRESETS.get(preset_name, {})
        if speaker == "S2":
            text = preset_config.get("s2_text", "[S2] I'm the second speaker.")
        else:
            text = preset_config.get("s1_text", "[S1] I'm the first speaker.")
        
        return (audio_np, text)
    except Exception as e:
        print(f"Error loading predefined voice {preset_name}/{speaker}: {e}")
        return None


@app.on_event("startup")
def _startup() -> None:
    # Lazy-load in a background thread so the web server becomes responsive
    # immediately (model download/load can take minutes).
    if os.getenv("DIA_AUTO_LOAD", "true").lower() != "true":
        return

    global _loading
    with _load_lock:
        if _state is not None or _loading:
            return
        _loading = True
        threading.Thread(target=_load_model, daemon=True).start()


@app.get("/health")
def health() -> dict:
    # Cleanup expired sessions periodically
    _cleanup_expired_sessions()
    
    return {
        "ok": True,
        "model_loaded": _state is not None,
        "loading": _loading,
        "error": _load_error,
        "active_sessions": len(_session_cache),
        "gcs_available": _gcs_voice_manager is not None,
    }


@app.get("/v1/voices/predefined", response_model=PredefinedVoicesResponse)
def list_predefined_voices() -> PredefinedVoicesResponse:
    """List all available GCS-backed predefined voice presets.
    
    These voices are stored in GCS and loaded on-demand for consistent
    voice cloning across TTS generation requests.
    """
    voices = []
    
    # Add GCS-backed voices
    if _gcs_voice_manager:
        available = _gcs_voice_manager.list_available_voices()
        for name in available:
            config = DEFAULT_VOICE_PRESETS.get(name, {})
            voices.append(PredefinedVoiceResponse(
                name=name,
                description=config.get("description", ""),
                style=config.get("style", ""),
                speakers=config.get("speakers", 1),
            ))
    
    # Fallback to DEFAULT_VOICE_PRESETS if no GCS
    if not voices:
        for name, config in DEFAULT_VOICE_PRESETS.items():
            voices.append(PredefinedVoiceResponse(
                name=name,
                description=config.get("description", ""),
                style=config.get("style", ""),
                speakers=config.get("speakers", 1),
            ))
    
    return PredefinedVoicesResponse(voices=voices, count=len(voices))


@app.post("/v1/audio/session", response_model=SessionResponse)
def create_session(req: SessionCreateRequest) -> SessionResponse:
    """
    Create a voice session with cached audio prompts.
    
    Audio prompts enable voice cloning - the model will generate audio
    that matches the voice characteristics of the provided reference audio.
    Sessions persist for ttl_seconds (default 1 hour) and can be used
    across multiple speech requests for consistent voice.
    """
    try:
        s1_audio = _decode_audio_prompt(req.s1_audio_prompt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid s1_audio_prompt: {e}")
    
    s2_audio = None
    if req.s2_audio_prompt:
        try:
            s2_audio = _decode_audio_prompt(req.s2_audio_prompt)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid s2_audio_prompt: {e}")
    
    session_id = str(uuid.uuid4())
    cache = AudioPromptCache(
        s1_audio=s1_audio,
        s1_text=req.s1_audio_prompt_text,
        s2_audio=s2_audio,
        s2_text=req.s2_audio_prompt_text,
        created_at=time.time(),
        ttl_seconds=req.ttl_seconds,
    )
    
    with _session_lock:
        _session_cache[session_id] = cache
    
    return SessionResponse(
        session_id=session_id,
        expires_at=int(cache.created_at + cache.ttl_seconds),
    )


class SessionFromBootstrapRequest(BaseModel):
    """Request to create a session from bootstrapped voices."""
    s1_voice_key: str = Field(..., description="Bootstrapped voice key for S1 (e.g., 'warm_duo_us_fm_s1')")
    s2_voice_key: Optional[str] = Field(None, description="Bootstrapped voice key for S2 (optional)")
    ttl_seconds: int = Field(3600, ge=60, le=86400)


@app.post("/v1/audio/session/bootstrap", response_model=SessionResponse)
def create_session_from_bootstrap(req: SessionFromBootstrapRequest) -> SessionResponse:
    """
    Create a voice session using pre-bootstrapped voice prompts.
    
    This enables voice consistency across chunks without requiring the client
    to upload audio prompts. Bootstrapped voices are generated on server startup.
    """
    s1_voice = _get_bootstrapped_voice(req.s1_voice_key)
    if not s1_voice:
        raise HTTPException(
            status_code=404, 
            detail=f"Bootstrapped voice not found: {req.s1_voice_key}. "
                   f"Available: {list(_bootstrapped_voices.keys())}"
        )
    
    s2_audio = None
    s2_text = None
    if req.s2_voice_key:
        s2_voice = _get_bootstrapped_voice(req.s2_voice_key)
        if not s2_voice:
            raise HTTPException(
                status_code=404,
                detail=f"Bootstrapped voice not found: {req.s2_voice_key}"
            )
        s2_audio = s2_voice.audio_prompt
        s2_text = s2_voice.audio_prompt_text
    
    session_id = str(uuid.uuid4())
    cache = AudioPromptCache(
        s1_audio=s1_voice.audio_prompt,
        s1_text=s1_voice.audio_prompt_text,
        s2_audio=s2_audio,
        s2_text=s2_text,
        created_at=time.time(),
        ttl_seconds=req.ttl_seconds,
    )
    
    with _session_lock:
        _session_cache[session_id] = cache
    
    return SessionResponse(
        session_id=session_id,
        expires_at=int(cache.created_at + cache.ttl_seconds),
    )


@app.get("/v1/audio/voices/bootstrapped")
def list_bootstrapped_voices() -> dict:
    """List all available bootstrapped voice prompts (legacy).
    
    For new integrations, use /v1/voices/predefined instead.
    """
    with _bootstrap_lock:
        return {
            "voices": list(_bootstrapped_voices.keys()),
            "count": len(_bootstrapped_voices),
        }


@app.delete("/v1/audio/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Delete a voice session and free its cached audio prompts."""
    with _session_lock:
        if session_id in _session_cache:
            del _session_cache[session_id]
            return {"deleted": True, "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/v1/audio/session/{session_id}")
def get_session(session_id: str) -> dict:
    """Check if a session exists and get its metadata."""
    with _session_lock:
        if session_id in _session_cache:
            cache = _session_cache[session_id]
            return {
                "session_id": session_id,
                "has_s1": True,
                "has_s2": cache.s2_audio is not None,
                "expires_at": int(cache.created_at + cache.ttl_seconds),
                "remaining_seconds": int(cache.created_at + cache.ttl_seconds - time.time()),
            }
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    if _state is None:
        if _load_error:
            raise HTTPException(status_code=503, detail=_load_error)
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = _ensure_speaker_tags(req.input)
    if not text:
        raise HTTPException(status_code=400, detail="input is required")

    # Determine voice mode
    voice_mode = req.voice_mode
    if voice_mode is None:
        # Auto-detect based on provided fields
        if req.session_id:
            voice_mode = VoiceMode.SESSION
        elif req.voice_preset:
            voice_mode = VoiceMode.PREDEFINED
        elif req.audio_prompt:
            voice_mode = VoiceMode.SESSION  # Inline audio prompt is session-like
        else:
            voice_mode = VoiceMode.SEED

    # Resolve audio prompt based on voice mode
    audio_prompt = None
    audio_prompt_text = None
    
    if voice_mode == VoiceMode.SESSION:
        if req.session_id:
            # Load audio prompt from session cache
            with _session_lock:
                if req.session_id not in _session_cache:
                    raise HTTPException(status_code=404, detail="Session not found or expired")
                cache = _session_cache[req.session_id]
                
                # Determine which speaker's audio prompt to use based on text
                speaker = _extract_speaker_tag(text)
                if speaker == "S2" and cache.s2_audio is not None:
                    audio_prompt = cache.s2_audio
                    audio_prompt_text = cache.s2_text
                else:
                    audio_prompt = cache.s1_audio
                    audio_prompt_text = cache.s1_text
        elif req.audio_prompt:
            # Decode audio prompt from request
            try:
                audio_prompt = _decode_audio_prompt(req.audio_prompt)
                audio_prompt_text = req.audio_prompt_text
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid audio_prompt: {e}")
    
    elif voice_mode == VoiceMode.PREDEFINED:
        if not req.voice_preset:
            raise HTTPException(status_code=400, detail="voice_preset required for predefined mode")
        
        # Load from GCS predefined voices
        speaker = _extract_speaker_tag(text)
        result = _get_predefined_voice_audio(req.voice_preset, speaker)
        if result:
            audio_prompt, audio_prompt_text = result
        else:
            # Fallback to bootstrapped voices (legacy)
            voice_key = f"{req.voice_preset}_{speaker.lower()}"
            bootstrapped = _get_bootstrapped_voice(voice_key)
            if bootstrapped:
                audio_prompt = bootstrapped.audio_prompt
                audio_prompt_text = bootstrapped.audio_prompt_text
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Predefined voice not found: {req.voice_preset}"
                )
    
    elif voice_mode == VoiceMode.SEED:
        # Check for bootstrapped voice prompts based on voice key (legacy support)
        voice_key = req.voice
        if voice_key and voice_key != "default":
            bootstrapped = _get_bootstrapped_voice(voice_key)
            if bootstrapped:
                audio_prompt = bootstrapped.audio_prompt
                audio_prompt_text = bootstrapped.audio_prompt_text
            else:
                # Try to match preset pattern (e.g., "warm_duo_us_fm" -> "warm_duo_us_fm_s1")
                speaker = _extract_speaker_tag(text)
                voice_key_with_speaker = f"{voice_key}_{speaker.lower()}"
                bootstrapped = _get_bootstrapped_voice(voice_key_with_speaker)
                if bootstrapped:
                    audio_prompt = bootstrapped.audio_prompt
                    audio_prompt_text = bootstrapped.audio_prompt_text

    # Seed-based voice consistency (fallback when no audio prompt):
    # If seed is provided use it; else derive from voice string for stable outputs.
    seed = req.seed
    if seed is None:
        # Deterministic mapping from voice string -> seed
        # Keep in 32-bit range.
        seed = (abs(hash(req.voice)) % (2**31 - 1))

    _set_seed(int(seed))

    use_torch_compile = os.getenv("DIA_USE_TORCH_COMPILE", "false").lower() == "true"
    # torch.compile is generally not supported on macOS; default false.
    if _state.device.type in {"mps", "cpu"}:
        use_torch_compile = False

    try:
        with torch.inference_mode():
            # Build generation kwargs
            gen_kwargs = {
                "text": text,
                "max_tokens": req.max_tokens,
                "cfg_scale": req.cfg_scale,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "cfg_filter_top_k": req.cfg_filter_top_k,
                "use_torch_compile": use_torch_compile,
                "verbose": False,
            }
            
            # Add audio prompt if available (for voice cloning)
            if audio_prompt is not None:
                gen_kwargs["audio_prompt"] = audio_prompt
                if audio_prompt_text:
                    gen_kwargs["audio_prompt_text"] = audio_prompt_text
            
            audio_np = _state.model.generate(**gen_kwargs)

        if audio_np is None or len(audio_np) == 0:
            raise RuntimeError("Dia returned empty audio")

        # Dia default sample rate is 44100
        sample_rate = 44100

        # Encode to WAV in-memory first
        wav_buf = io.BytesIO()
        sf.write(wav_buf, np.asarray(audio_np, dtype=np.float32), sample_rate, format="WAV")
        wav_bytes = wav_buf.getvalue()

        if req.format.lower() == "wav":
            return Response(content=wav_bytes, media_type="audio/wav")

        if req.format.lower() != "mp3":
            raise HTTPException(status_code=400, detail="format must be mp3 or wav")

        mp3_bytes = _wav_to_mp3_bytes(wav_bytes, sample_rate, float(req.speed))
        return Response(content=mp3_bytes, media_type="audio/mpeg")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
