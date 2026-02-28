import os
import sys
import threading
import time

from colorama import Fore, Style, init
from lib.app import AppConfig, App
from lib.rag import llm_query
from dotenv import load_dotenv

from lib.request_logger import RequestLogger

# from huggingface_hub import whoami

init(autoreset=True)

# EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# CHROMA_PATH = "chroma_db_final"
# COLLECTION_NAME = "matrix_bge-base"

def spinner(stop_event, prefix="Thinking"):
    frames = ["|", "/", "-", "\\"]
    i = 0
    while not stop_event.is_set():
        msg = f"{Fore.BLUE}{prefix} {frames[i % len(frames)]}"
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)

    # Clear the spinner line
    sys.stdout.write("\r" + " " * (len(prefix)+2) + "\r")
    sys.stdout.flush()

def main():
    load_dotenv(verbose=True)

    cfg = AppConfig(
        chroma_path=os.environ.get('CHROMA_PATH', 'chroma_db_final'),
        collection_name=os.environ.get('COLLECTION_NAME', 'matrix_bge-base'),
        emb_model_name=os.environ.get('EMB_MODEL_NAME', 'BAAI/bge-base-en-v1.5'),
        rerank_model_name=os.environ.get('RERANK_MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
        openai_base_url="https://api.deepseek.com",
        openai_api_key=os.environ['OPENAI_API_KEY_DEEPSEEK'],
        safety=os.environ.get('SAFETY', 'true').strip().lower() == 'true'
    )
    app = App(cfg)
    req_logger = RequestLogger()

    print(f"{Fore.BLUE}Welcome to RAG Chat of QuantumForge Software")
    print(f"{Fore.BLUE}Feel free to ask")
    print(f"{Fore.BLUE}Type 'exit' or press Ctrl+C to quit\n")

    try:
        while True:
            question = input(f"{Fore.CYAN}> ")

            if question.strip().lower() in {"exit", "quit"}:
                print(f"{Fore.RED}Bye 👋")
                break

            stop = threading.Event()
            t = threading.Thread(target=spinner, args=(stop,), daemon=True)
            t.start()
            try:
                logger, path = req_logger.create_request_log()
                logger.info(f"safety: {app.cfg.safety}")
                logger.info(f"question: {question}")
                answer = llm_query(app, question, logger)
            finally:
                stop.set()
                t.join()

            # print(f"{Fore.YELLOW}Q: {question}")
            # print(f"{Fore.GREEN}A1: {ANSWERS[0]}")
            # print(f"{Fore.MAGENTA}A2: {ANSWERS[1]}")
            print(f"{Fore.GREEN}{answer}")
            print()

    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Interrupted. Bye 👋")


if __name__ == "__main__":
    main()
