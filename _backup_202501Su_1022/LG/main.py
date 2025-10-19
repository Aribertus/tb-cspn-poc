import sys
from typing import List, Dict

from loguru import logger

from assets.helper.logging import print_summary, log_processing
from assets.custom_obj import BaseLog
from assets.helper.config_helper import load_settings, log_settings
from assets.run import process_input


logs: List[Dict[str, Dict[str, List[BaseLog]]]] = []
# Struttura dell'oggetto di log
# Analisi dall'esterno verso l'interno
# List[Dict[.....]] -> ogni dizionario della lista ha chiave IncX e raccoglie tutte le info per l'incident
# ...Dict[str, Dict[str, .... -> Ogni dizionario con chiave IncX ha al suo interno 3 dizionari, con chiavi Supervisor, Consultant, worker
# ...Dict[str,List[BaseLog]]... -> Ogni chiave relativa al ruolo contiene una lista di log specifici del ruolo, tutti estensioni di BaseLog

def main():
    settings = load_settings()
    log_settings(settings)
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level.upper())
    print_summary(
        log_processing(
            process_input(
                settings.llm_call,
                settings.n_items
            )
        ),
        settings)


if __name__=="__main__":
    main()
