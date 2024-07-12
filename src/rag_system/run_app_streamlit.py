"""Allows to run the streamlit app from within the python package."""
import sys

import streamlit.web.cli as stcli


def main():
    """Run the streamlit program from within a Python script."""
    sys.argv = ["streamlit", "run", "rag_system/app_streamlit.py",
                "--server.port", "1028"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
