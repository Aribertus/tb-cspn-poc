from typing import Optional, Dict, Any
from pathlib import Path

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Literal

Style = Literal["simple", "table", "pretty"]
DebugLevel = Literal["info","debug"]

class AppSettings(BaseModel):
    style: Style = Field(default="simple", description="Formato dell'output")
    folder: str = Field(default="runs", description="Cartella di destinazione")
    filename: Optional[str] = Field(default=None, description="Nome file; se assente usa timestamp")
    llm_call: bool = Field(default=False, description="Uso di llm nei nodi di supervisor")
    n_items: int = Field(default=50, description="Su quant oggetti eseguire la run. Se il numero è maggiore degli oggetti presenti, verrò usato il numero degli oggetti presenti")
    log_level: DebugLevel = Field(default="info", description="Regola la verbosità dei log")
    model: str = Field(default="gpt-4o-mini", description="Il modello usato per le chiamate agli LLM")
    temperature: float = Field(default=0.5, description="La temperatura per la creatività dei modelli")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT.parent / "config.yaml"

DEFAULT_CONFIG_CONTENT = {
    "style": "simple",
    "folder": "runs",
    "filename": None,
    "llm_call": False,
    "n_items": 2,
    "log_level": "info",
    "model": "gpt-4o-mini",
    "temperature": 0.5,
}

def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> AppSettings:
    if not path.exists():
        logger.debug(f"{path.name} non trovato. Creo un file di default.")
        path.write_text(yaml.safe_dump(DEFAULT_CONFIG_CONTENT, sort_keys=False, allow_unicode=True), encoding="utf-8")
    try:
        data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        settings = AppSettings.model_validate(data)
        logger.debug(f"Config caricata da {path}: {settings.model_dump()}")
        return settings
    except ValidationError as e:
        logger.error(f"config.yaml non valida:\n{e}")
        raise

def log_settings(settings: AppSettings) -> None:
    dumped = yaml.safe_dump(
        settings.model_dump(),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False
    )
    logger.info("\n=== App Settings ===\n" + dumped + "====================")
