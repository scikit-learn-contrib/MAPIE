import sys
from pathlib import Path

from nbclient import NotebookClient
from nbformat import read

NOTEBOOKS_DIR = Path(__file__).parent
notebooks = NOTEBOOKS_DIR.rglob("*.ipynb")

if __name__ == "__main__":
    success = True

    for nb_path in notebooks:
        print(f"Running {nb_path} ...")
        try:
            with nb_path.open() as f:
                nb = read(f, as_version=4)
            client = NotebookClient(nb, timeout=60)
            client.execute()
        except Exception as e:
            print(f"Notebook {nb_path} failed:\n{e}")
            success = False

    if success:
        print("\nAll notebooks executed successfully. 100% OK.\n")
        sys.exit(0)
    else:
        print("\nSome notebooks failed.")
        sys.exit(1)
