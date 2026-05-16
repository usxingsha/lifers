"""
lifers/core/voice_manager.py
────────────────────────────────────
Female voice configuration for all major countries/locales.
Supports: Windows SAPI (pyttsx3), Edge TTS (edge-tts), custom engines.
Fully customizable — voices can be added/edited via voice_config.json.
"""
from __future__ import annotations
import json, logging, subprocess, sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "voice_config.json"


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class VoiceProfile:
    id:           str            # unique key e.g. "zh-CN-XiaoxiaoNeural"
    name:         str            # display name
    country:      str            # e.g. "China"
    locale:       str            # BCP-47 e.g. "zh-CN"
    language:     str            # e.g. "Chinese (Simplified)"
    gender:       str            = "female"
    engine:       str            = "edge-tts"   # edge-tts | sapi | custom
    rate:         str            = "+0%"        # speech rate
    volume:       str            = "+0%"
    pitch:        str            = "+0Hz"
    custom_cmd:   Optional[str]  = None         # shell cmd for custom engine
    enabled:      bool           = True
    notes:        str            = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ── Built-in female voice database ───────────────────────────────────────────

BUILTIN_VOICES: list[VoiceProfile] = [
    # ── East Asia ─────────────────────────────────────────────────────────────
    VoiceProfile("zh-CN-XiaoxiaoNeural",   "晓晓",        "China",       "zh-CN", "Chinese (Simplified)", notes="温柔、自然，微软默认"),
    VoiceProfile("zh-CN-XiaoyiNeural",     "晓伊",        "China",       "zh-CN", "Chinese (Simplified)", notes="活泼"),
    VoiceProfile("zh-CN-XiaohanNeural",    "晓涵",        "China",       "zh-CN", "Chinese (Simplified)", notes="成熟"),
    VoiceProfile("zh-CN-XiaomengNeural",   "晓梦",        "China",       "zh-CN", "Chinese (Simplified)", notes="甜美"),
    VoiceProfile("zh-CN-XiaoruiNeural",    "晓睿",        "China",       "zh-CN", "Chinese (Simplified)", notes="知性"),
    VoiceProfile("zh-TW-HsiaoChenNeural",  "曉臻",        "Taiwan",      "zh-TW", "Chinese (Traditional)"),
    VoiceProfile("zh-HK-HiuMaanNeural",    "曉曼",        "Hong Kong",   "zh-HK", "Chinese (Cantonese)"),
    VoiceProfile("ja-JP-NanamiNeural",     "七海",         "Japan",       "ja-JP", "Japanese"),
    VoiceProfile("ja-JP-AoiNeural",        "葵",           "Japan",       "ja-JP", "Japanese", notes="明るい"),
    VoiceProfile("ko-KR-SunHiNeural",      "선희",         "Korea",       "ko-KR", "Korean"),
    VoiceProfile("ko-KR-YuJinNeural",      "유진",         "Korea",       "ko-KR", "Korean", notes="활발한"),

    # ── Southeast Asia ────────────────────────────────────────────────────────
    VoiceProfile("vi-VN-HoaiMyNeural",     "Hoài My",     "Vietnam",     "vi-VN", "Vietnamese"),
    VoiceProfile("th-TH-PremwadeeNeural",  "Premwadee",   "Thailand",    "th-TH", "Thai"),
    VoiceProfile("id-ID-GadisNeural",      "Gadis",       "Indonesia",   "id-ID", "Indonesian"),
    VoiceProfile("ms-MY-YasminNeural",     "Yasmin",      "Malaysia",    "ms-MY", "Malay"),
    VoiceProfile("fil-PH-BlessicaNeural",  "Blessica",    "Philippines", "fil-PH","Filipino"),

    # ── South Asia ────────────────────────────────────────────────────────────
    VoiceProfile("hi-IN-SwaraNeural",      "Swara",       "India",       "hi-IN", "Hindi"),
    VoiceProfile("bn-IN-TanishaaNeural",   "Tanishaa",    "India",       "bn-IN", "Bengali (India)"),
    VoiceProfile("ta-IN-PallaviNeural",    "Pallavi",     "India",       "ta-IN", "Tamil"),
    VoiceProfile("te-IN-ShrutiNeural",     "Shruti",      "India",       "te-IN", "Telugu"),
    VoiceProfile("ur-PK-UzmaNeural",       "Uzma",        "Pakistan",    "ur-PK", "Urdu"),
    VoiceProfile("si-LK-ThiliniNeural",    "Thilini",     "Sri Lanka",   "si-LK", "Sinhala"),
    VoiceProfile("ne-NP-HemkalaNeural",    "Hemkala",     "Nepal",       "ne-NP", "Nepali"),

    # ── Middle East / Central Asia ────────────────────────────────────────────
    VoiceProfile("ar-SA-ZariyahNeural",    "Zariyah",     "Saudi Arabia","ar-SA", "Arabic (Saudi)"),
    VoiceProfile("ar-EG-SalmaNeural",      "Salma",       "Egypt",       "ar-EG", "Arabic (Egypt)"),
    VoiceProfile("fa-IR-DilaraNeural",     "Dilara",      "Iran",        "fa-IR", "Persian"),
    VoiceProfile("tr-TR-EmelNeural",       "Emel",        "Turkey",      "tr-TR", "Turkish"),
    VoiceProfile("he-IL-HilaNeural",       "Hila",        "Israel",      "he-IL", "Hebrew"),
    VoiceProfile("kk-KZ-AigulNeural",      "Aigul",       "Kazakhstan",  "kk-KZ", "Kazakh"),

    # ── Europe — Western ──────────────────────────────────────────────────────
    VoiceProfile("en-US-JennyNeural",      "Jenny",       "USA",         "en-US", "English (US)", notes="标准美式，最常用"),
    VoiceProfile("en-US-AriaNeural",       "Aria",        "USA",         "en-US", "English (US)"),
    VoiceProfile("en-US-SaraNeural",       "Sara",        "USA",         "en-US", "English (US)"),
    VoiceProfile("en-GB-LibbyNeural",      "Libby",       "UK",          "en-GB", "English (UK)"),
    VoiceProfile("en-GB-MiaNeural",        "Mia",         "UK",          "en-GB", "English (UK)"),
    VoiceProfile("en-AU-NatashaNeural",    "Natasha",     "Australia",   "en-AU", "English (AU)"),
    VoiceProfile("en-CA-ClaraNeural",      "Clara",       "Canada",      "en-CA", "English (CA)"),
    VoiceProfile("en-IE-EmilyNeural",      "Emily",       "Ireland",     "en-IE", "English (IE)"),
    VoiceProfile("en-NZ-MollyNeural",      "Molly",       "New Zealand", "en-NZ", "English (NZ)"),
    VoiceProfile("en-ZA-LeahNeural",       "Leah",        "South Africa","en-ZA", "English (ZA)"),
    VoiceProfile("fr-FR-DeniseNeural",     "Denise",      "France",      "fr-FR", "French"),
    VoiceProfile("fr-BE-CharlineNeural",   "Charline",    "Belgium",     "fr-BE", "French (BE)"),
    VoiceProfile("fr-CA-SylvieNeural",     "Sylvie",      "Canada",      "fr-CA", "French (CA)"),
    VoiceProfile("de-DE-KatjaNeural",      "Katja",       "Germany",     "de-DE", "German"),
    VoiceProfile("de-AT-IngridNeural",     "Ingrid",      "Austria",     "de-AT", "German (AT)"),
    VoiceProfile("de-CH-LeniNeural",       "Leni",        "Switzerland", "de-CH", "German (CH)"),
    VoiceProfile("es-ES-ElviraNeural",     "Elvira",      "Spain",       "es-ES", "Spanish (ES)"),
    VoiceProfile("es-MX-DaliaNeural",      "Dalia",       "Mexico",      "es-MX", "Spanish (MX)"),
    VoiceProfile("es-AR-ElenaNeural",      "Elena",       "Argentina",   "es-AR", "Spanish (AR)"),
    VoiceProfile("it-IT-ElsaNeural",       "Elsa",        "Italy",       "it-IT", "Italian"),
    VoiceProfile("pt-BR-FranciscaNeural",  "Francisca",   "Brazil",      "pt-BR", "Portuguese (BR)"),
    VoiceProfile("pt-PT-RaquelNeural",     "Raquel",      "Portugal",    "pt-PT", "Portuguese (PT)"),
    VoiceProfile("nl-NL-FennaNeural",      "Fenna",       "Netherlands", "nl-NL", "Dutch"),
    VoiceProfile("nl-BE-DenaNeural",       "Dena",        "Belgium",     "nl-BE", "Dutch (BE)"),

    # ── Europe — Northern ─────────────────────────────────────────────────────
    VoiceProfile("sv-SE-SofieNeural",      "Sofie",       "Sweden",      "sv-SE", "Swedish"),
    VoiceProfile("nb-NO-PernilleNeural",   "Pernille",    "Norway",      "nb-NO", "Norwegian"),
    VoiceProfile("da-DK-ChristelNeural",   "Christel",    "Denmark",     "da-DK", "Danish"),
    VoiceProfile("fi-FI-NooraNeural",      "Noora",       "Finland",     "fi-FI", "Finnish"),
    VoiceProfile("is-IS-GudrunNeural",     "Gudrun",      "Iceland",     "is-IS", "Icelandic"),

    # ── Europe — Eastern ──────────────────────────────────────────────────────
    VoiceProfile("ru-RU-SvetlanaNeural",   "Светлана",    "Russia",      "ru-RU", "Russian"),
    VoiceProfile("pl-PL-ZofiaNeural",      "Zofia",       "Poland",      "pl-PL", "Polish"),
    VoiceProfile("cs-CZ-VlastaNeural",     "Vlasta",      "Czechia",     "cs-CZ", "Czech"),
    VoiceProfile("sk-SK-ViktoriaNeural",   "Viktória",    "Slovakia",    "sk-SK", "Slovak"),
    VoiceProfile("uk-UA-PolinaNeural",     "Polina",      "Ukraine",     "uk-UA", "Ukrainian"),
    VoiceProfile("ro-RO-AlinaNeural",      "Alina",       "Romania",     "ro-RO", "Romanian"),
    VoiceProfile("hu-HU-NoemiNeural",      "Noémi",       "Hungary",     "hu-HU", "Hungarian"),
    VoiceProfile("bg-BG-KalinaNeural",     "Kalina",      "Bulgaria",    "bg-BG", "Bulgarian"),
    VoiceProfile("hr-HR-GabrijelaNeural",  "Gabrijela",   "Croatia",     "hr-HR", "Croatian"),
    VoiceProfile("sr-RS-SophieNeural",     "Sophie",      "Serbia",      "sr-RS", "Serbian"),
    VoiceProfile("sl-SI-PetraNeural",      "Petra",       "Slovenia",    "sl-SI", "Slovenian"),
    VoiceProfile("et-EE-AnuNeural",        "Anu",         "Estonia",     "et-EE", "Estonian"),
    VoiceProfile("lv-LV-EveritaNeural",    "Everita",     "Latvia",      "lv-LV", "Latvian"),
    VoiceProfile("lt-LT-OnaNeural",        "Ona",         "Lithuania",   "lt-LT", "Lithuanian"),
    VoiceProfile("mk-MK-MarijaNeural",     "Marija",      "Macedonia",   "mk-MK", "Macedonian"),
    VoiceProfile("sq-AL-AnilaNeural",      "Anila",       "Albania",     "sq-AL", "Albanian"),
    VoiceProfile("el-GR-AthinaNeural",     "Αθηνά",       "Greece",      "el-GR", "Greek"),

    # ── Africa ────────────────────────────────────────────────────────────────
    VoiceProfile("sw-KE-ZuriNeural",       "Zuri",        "Kenya",       "sw-KE", "Swahili (KE)"),
    VoiceProfile("sw-TZ-RehemaNeural",     "Rehema",      "Tanzania",    "sw-TZ", "Swahili (TZ)"),
    VoiceProfile("am-ET-MekdesNeural",     "Mekdes",      "Ethiopia",    "am-ET", "Amharic"),
    VoiceProfile("zu-ZA-ThandoNeural",     "Thando",      "South Africa","zu-ZA", "Zulu"),
    VoiceProfile("af-ZA-AdriNeural",       "Adri",        "South Africa","af-ZA", "Afrikaans"),
    VoiceProfile("yo-NG-AdeNeural",        "Ade",         "Nigeria",     "yo-NG", "Yoruba"),

    # ── Latin America (extra) ─────────────────────────────────────────────────
    VoiceProfile("es-CO-SalomeNeural",     "Salomé",      "Colombia",    "es-CO", "Spanish (CO)"),
    VoiceProfile("es-CL-CatalinaNeural",   "Catalina",    "Chile",       "es-CL", "Spanish (CL)"),
    VoiceProfile("es-PE-CamilaNeural",     "Camila",      "Peru",        "es-PE", "Spanish (PE)"),
    VoiceProfile("es-VE-PaolaNeural",      "Paola",       "Venezuela",   "es-VE", "Spanish (VE)"),
]


# ── Manager ───────────────────────────────────────────────────────────────────

class VoiceManager:
    """
    Manages female voice profiles for all countries.

    Usage
    -----
    vm = VoiceManager()
    voices = vm.list_by_country("China")
    vm.speak("你好世界", voice_id="zh-CN-XiaoxiaoNeural")
    vm.add_custom(VoiceProfile(...))
    vm.save()
    """

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        self._path = config_path
        self._voices: dict[str, VoiceProfile] = {v.id: v for v in BUILTIN_VOICES}
        self._load_custom()

    # ── Query ──────────────────────────────────────────────────────────────────

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        return self._voices.get(voice_id)

    def list_all(self) -> list[VoiceProfile]:
        return [v for v in self._voices.values() if v.enabled]

    def list_by_country(self, country: str) -> list[VoiceProfile]:
        c = country.lower()
        return [v for v in self._voices.values()
                if v.country.lower() == c and v.enabled]

    def list_by_locale(self, locale: str) -> list[VoiceProfile]:
        return [v for v in self._voices.values()
                if v.locale.lower() == locale.lower() and v.enabled]

    def list_countries(self) -> list[str]:
        return sorted({v.country for v in self._voices.values()})

    def search(self, query: str) -> list[VoiceProfile]:
        q = query.lower()
        return [v for v in self._voices.values()
                if q in v.country.lower() or q in v.locale.lower()
                or q in v.name.lower() or q in v.language.lower()]

    # ── Custom voice CRUD ─────────────────────────────────────────────────────

    def add_custom(self, profile: VoiceProfile) -> None:
        """Add or overwrite a voice profile."""
        self._voices[profile.id] = profile
        log.info("Added voice: %s (%s)", profile.id, profile.country)

    def add_custom_from_dict(self, data: dict) -> VoiceProfile:
        """Convenience: build VoiceProfile from dict and add it."""
        p = VoiceProfile(**data)
        self.add_custom(p)
        return p

    def remove(self, voice_id: str) -> bool:
        if voice_id in self._voices:
            del self._voices[voice_id]
            return True
        return False

    def disable(self, voice_id: str) -> None:
        if v := self._voices.get(voice_id):
            v.enabled = False

    def enable(self, voice_id: str) -> None:
        if v := self._voices.get(voice_id):
            v.enabled = True

    # ── Playback ──────────────────────────────────────────────────────────────

    def speak(
        self,
        text: str,
        voice_id: Optional[str] = None,
        locale: Optional[str]   = None,
        output_file: Optional[Path] = None,
    ) -> bool:
        """
        Speak text using selected voice.
        Tries edge-tts first (async), falls back to pyttsx3 (sync SAPI).
        If output_file is given, saves audio instead of playing.
        """
        voice = self._pick(voice_id, locale)
        if voice is None:
            log.error("No voice found for id=%s locale=%s", voice_id, locale)
            return False

        if voice.engine == "custom" and voice.custom_cmd:
            return self._speak_custom(text, voice, output_file)
        if voice.engine == "edge-tts":
            return self._speak_edge(text, voice, output_file)
        if voice.engine == "sapi":
            return self._speak_sapi(text, voice)
        log.warning("Unknown engine: %s", voice.engine)
        return False

    def _pick(self, voice_id: Optional[str], locale: Optional[str]) -> Optional[VoiceProfile]:
        if voice_id:
            return self._voices.get(voice_id)
        if locale:
            matches = self.list_by_locale(locale)
            return matches[0] if matches else None
        # Default: first zh-CN voice
        for v in self._voices.values():
            if v.locale == "zh-CN" and v.enabled:
                return v
        return next(iter(self._voices.values()), None)

    def _speak_edge(self, text: str, voice: VoiceProfile,
                    out: Optional[Path]) -> bool:
        try:
            import asyncio
            async def _run():
                import edge_tts  # pip install edge-tts
                c = edge_tts.Communicate(
                    text, voice.id,
                    rate=voice.rate, volume=voice.volume, pitch=voice.pitch)
                target = str(out) if out else "lifers_tts_output.mp3"
                await c.save(target)
            asyncio.run(_run())
            if not out:
                self._play("lifers_tts_output.mp3")
            return True
        except ImportError:
            log.warning("edge-tts not installed. Run: pip install edge-tts")
            return self._speak_sapi(text, voice)
        except Exception as e:
            log.error("edge-tts error: %s", e)
            return False

    def _speak_sapi(self, text: str, voice: VoiceProfile) -> bool:
        try:
            import pyttsx3  # pip install pyttsx3
            eng = pyttsx3.init()
            voices = eng.getProperty("voices")
            # Try to match locale
            for v in voices:
                if voice.locale.replace("-", "_").lower() in (v.id or "").lower():
                    eng.setProperty("voice", v.id)
                    break
            eng.say(text)
            eng.runAndWait()
            return True
        except ImportError:
            log.warning("pyttsx3 not installed. Run: pip install pyttsx3")
            return False
        except Exception as e:
            log.error("pyttsx3 error: %s", e)
            return False

    def _speak_custom(self, text: str, voice: VoiceProfile,
                      out: Optional[Path]) -> bool:
        cmd = voice.custom_cmd.format(
            text=text,
            output=str(out) if out else "output.wav")
        try:
            subprocess.run(cmd, shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            log.error("Custom voice cmd failed: %s", e)
            return False

    @staticmethod
    def _play(path: str) -> None:
        if sys.platform == "win32":
            subprocess.run(["start", "/wait", "", path], shell=True)
        elif sys.platform == "darwin":
            subprocess.run(["afplay", path])
        else:
            subprocess.run(["aplay", path], capture_output=True)

    # ── Persist custom voices ─────────────────────────────────────────────────

    def save(self) -> None:
        """Persist all non-builtin (custom) voices to voice_config.json."""
        builtin_ids = {v.id for v in BUILTIN_VOICES}
        custom = [v.to_dict() for v in self._voices.values()
                  if v.id not in builtin_ids]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump({"custom_voices": custom}, f, ensure_ascii=False, indent=2)
        log.info("Saved %d custom voice(s)", len(custom))

    def _load_custom(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for d in data.get("custom_voices", []):
                p = VoiceProfile(**d)
                self._voices[p.id] = p
                log.info("Loaded custom voice: %s", p.id)
        except Exception as e:
            log.warning("voice_config.json load error: %s", e)

    def export_all(self) -> list[dict]:
        return [v.to_dict() for v in self._voices.values()]
