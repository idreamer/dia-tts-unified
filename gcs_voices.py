"""GCS-based predefined voices module for Dia TTS.

This module provides functionality to fetch and cache voice prompts
from Google Cloud Storage for consistent voice cloning across requests.
"""

import io
import logging
import os
import timerom dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Try to import GCS client, fall back gracefully if not available
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage not installed. GCS voices will be unavailable.")


@dataclass
class PredefinedVoice:
    """A predefined voice loaded from GCS."""
    name: str
    display_name: str
    description: str
    speaker_mode: str  # "dialogue" or "monologue"
    s1_audio: Optional[np.ndarray] = None
    s1_text: Optional[str] = None
    s2_audio: Optional[np.ndarray] = None
    s2_text: Optional[str] = None
    gcs_prefix: Optional[str] = None
    loaded_at: float = field(default_factory=time.time)


# Default voice presets with their metadata
DEFAULT_VOICE_PRESETS = {
    "warm_duo_us_fm": {
        "display_name": "Warm Duo (F/M)",
        "description": "Warm, conversational female and male hosts",
        "speaker_mode": "dialogue",
        "s1_text": "[S1] Welcome to today's episode! I'm so excited to share some incredible stories with you. Let's dive right in and explore what's happening in the world of technology and innovation.",
        "s2_text": "[S2] That's a great point! I've been thinking about this topic all week and there's so much we can unpack together. Let me share some insights that might surprise you.",
    },
    "warm_duo_us_ff": {
        "display_name": "Warm Duo (F/F)",
        "description": "Two warm, conversational female hosts",
        "speaker_mode": "dialogue",
        "s1_text": "[S1] Hey everyone, welcome back to another fantastic episode! Today we have some amazing content lined up that I know you're going to love. Let's get started!",
        "s2_text": "[S2] Absolutely! I've been looking forward to this discussion all week. There are so many interesting angles we can explore on this topic together.",
    },
    "warm_duo_us_mm": {
        "display_name": "Warm Duo (M/M)",
        "description": "Two warm, conversational male hosts",
        "speaker_mode": "dialogue",
        "s1_text": "[S1] What's up everyone, welcome to the show! We've got a packed episode for you today with some really fascinating topics to dig into. Let's jump right in!",
        "s2_text": "[S2] Couldn't agree more! This is exactly the kind of topic I love diving deep into. There's a lot of nuance here that we should definitely explore.",
    },
    "warm_duo_us_mf": {
        "display_name": "Warm Duo (M/F)",
        "description": "Warm, conversational male and female hosts",
        "speaker_mode": "dialogue",
        "s1_text": "[S1] Good morning everyone and welcome to today's episode! I'm really excited about what we have lined up for you. There's so much to cover, so let's get right into it!",
        "s2_text": "[S2] Thanks for having me! I've been doing a lot of research on this topic and I think our listeners are going to find this really valuable and insightful.",
    },
    "professional_duo_fm": {
        "display_name": "Professional Duo (F/M)",
        "description": "Professional, news-style female and male hosts",
        "speaker_mode": "dialogue",
        "s1_text": "[S1] Good evening. Tonight we bring you comprehensive coverage of the day's most significant developments across technology, business, and innovation sectors worldwide.",
        "s2_text": "[S2] The analysis suggests several key factors at play here. Let me walk you through the implications of these findings and what they mean for the industry.",
    },
    "narrative_solo_f": {
        "display_name": "Narrative Solo (Female)",
        "description": "Single female narrator for storytelling",
        "speaker_mode": "monologue",
        "s1_text": "[S1] In this episode, we journey through the latest innovations shaping our world. From breakthrough discoveries to emerging trends, here's everything you need to know about what's happening today.",
        "s2_text": None,
    },
    "narrative_solo_m": {
        "display_name": "Narrative Solo (Male)",
        "description": "Single male narrator for storytelling",
        "speaker_mode": "monologue",
        "s1_text": "[S1] Today's stories take us across continents and industries. We'll examine the forces driving change, the people behind the innovations, and what it all means for the future.",
        "s2_text": None,
    },
}


class GCSVoiceManager:
    """Manages predefined voices from Google Cloud Storage."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        prefix: str = "voice-presets",
        cache_ttl_seconds: int = 3600,
    ):
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET")
        self.prefix = prefix
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, PredefinedVoice] = {}
        self._client: Optional["storage.Client"] = None

    def _get_client(self) -> Optional["storage.Client"]:
        """Get or create GCS client (auto-detects credentials)."""
        if not GCS_AVAILABLE:
            return None
        if self._client is None:
            try:
                self._client = storage.Client()
            except Exception as e:
                logger.error(f"Failed to create GCS client: {e}")
                return None
        return self._client

    def list_available_voices(self) -> List[dict]:
        """List all available predefined voices with metadata."""
        voices = []
        for name, meta in DEFAULT_VOICE_PRESETS.items():
            voices.append({
                "name": name,
                "display_name": meta["display_name"],
                "description": meta["description"],
                "speaker_mode": meta["speaker_mode"],
                "s1_key": f"{name}_s1",
                "s2_key": f"{name}_s2" if meta["speaker_mode"] == "dialogue" else None,
            })
        return voices

    def _load_audio_from_gcs(self, blob_path: str) -> Optional[np.ndarray]:
        """Load audio file from GCS and return as numpy array."""
        client = self._get_client()
        if not client or not self.bucket_name:
            return None

        try:
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                logger.warning(f"GCS blob not found: {blob_path}")
                return None

            audio_bytes = blob.download_as_bytes()
            
            # Decode audio to numpy array
            with io.BytesIO(audio_bytes) as buf:
                audio_data, sr = sf.read(buf)
                # Resample to 44100 if needed
                if sr != 44100:
                    ratio = 44100 / sr
                    new_length = int(len(audio_data) * ratio)
                    indices = np.linspace(0, len(audio_data) - 1, new_length)
                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
                return audio_data.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to load audio from GCS {blob_path}: {e}")
            return None

    def get_voice(self, preset_name: str) -> Optional[PredefinedVoice]:
        """Get a predefined voice, loading from GCS if not cached."""
        # Check cache first
        if preset_name in self._cache:
            cached = self._cache[preset_name]
            if time.time() - cached.loaded_at < self.cache_ttl_seconds:
                return cached

        # Get metadata
        if preset_name not in DEFAULT_VOICE_PRESETS:
            return None

        meta = DEFAULT_VOICE_PRESETS[preset_name]
        
        # Try to load from GCS
        s1_path = f"{self.prefix}/{preset_name}/s1_prompt.mp3"
        s1_audio = self._load_audio_from_gcs(s1_path)
        
        s2_audio = None
        if meta["speaker_mode"] == "dialogue":
            s2_path = f"{self.prefix}/{preset_name}/s2_prompt.mp3"
            s2_audio = self._load_audio_from_gcs(s2_path)

        voice = PredefinedVoice(
            name=preset_name,
            display_name=meta["display_name"],
            description=meta["description"],
            speaker_mode=meta["speaker_mode"],
            s1_audio=s1_audio,
            s1_text=meta["s1_text"],
            s2_audio=s2_audio,
            s2_text=meta.get("s2_text"),
            gcs_prefix=f"{self.prefix}/{preset_name}",
        )

        # Cache even if audio is None (to avoid repeated GCS calls)
        self._cache[preset_name] = voice
        return voice

    def get_voice_audio_prompt(
        self, voice_key: str
    ) -> Optional[tuple[np.ndarray, str]]:
        """Get audio prompt for a specific voice key (e.g., 'warm_duo_us_fm_s1').
        
        Returns:
            Tuple of (audio_array, prompt_text) or None if not found.
        """
        # Parse voice key: "preset_name_s1" or "preset_name_s2"
        if voice_key.endswith("_s1"):
            preset_name = voice_key[:-3]
            speaker = "s1"
        elif voice_key.endswith("_s2"):
            preset_name = voice_key[:-3]
            speaker = "s2"
        else:
            # Assume it's a preset name, default to S1
            preset_name = voice_key
            speaker = "s1"

        voice = self.get_voice(preset_name)
        if not voice:
            return None

        if speaker == "s1":
            if voice.s1_audio is not None:
                return (voice.s1_audio, voice.s1_text or "")
        elif speaker == "s2":
            if voice.s2_audio is not None:
                return (voice.s2_audio, voice.s2_text or "")

        return None

    def clear_cache(self):
        """Clear the voice cache."""
        self._cache.clear()


# Global instance
_voice_manager: Optional[GCSVoiceManager] = None


def get_voice_manager() -> GCSVoiceManager:
    """Get or create the global voice manager."""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = GCSVoiceManager()
    return _voice_manager
