import logging
import warnings
from rag.reranker import _get_reranker
from rag.language import _get_detector

_get_reranker()
_get_detector()

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)

warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings", category=UserWarning
)

from rag.graph import app

if __name__ == "__main__":
    #app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("Hello AI Assistant!")
    print(
        app.invoke(
            {
                "question": "Šta su poslovni sistemi i koje su njihove ključne komponente?"
            }
        )
    )