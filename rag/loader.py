from markitdown import MarkItDown 
import pymupdf4llm




def load_document(file_path):
    """
    Converts a supported document into Markdown

    """
    if file_path.suffix.lower() == ".pdf":
        return pymupdf4llm.to_markdown(str(file_path))
    md=MarkItDown()
    return md.convert(str(file_path)).text_content