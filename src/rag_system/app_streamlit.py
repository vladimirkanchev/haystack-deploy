"""An entrypoint file streamlit gui of seven wonders app."""
from typing import Tuple
import os
import streamlit as st
print(os.getcwd())
from rag_system.responds import get_respond_streamlit
from rag_system.utils import create_gt_data
from rag_system.utils import create_qui_question_data

NUM_COLS = 2
PARAMS = ["faithfulness: "]
VALS_STR = ['val1', 'val2']


def initialize() -> None:
    """Initialize streamlit guis areas."""
    if VALS_STR[0] not in st.session_state:
        st.session_state[VALS_STR[0]] = ""
    if VALS_STR[1] not in st.session_state:
        st.session_state[VALS_STR[1]] = ""
    if 'parm_text' not in st.session_state:
        st.session_state.parm_text = ""


def enter_wonder_question() -> None:
    """Generate and evauate AI answer for a question."""
    st.title("AI App for the Seven Ancient Wonders:")

    question_gui_data = create_qui_question_data()
    ground_truth_data = create_gt_data()
    initialize()

    left_column, right_column = st.columns(NUM_COLS)
    with right_column:
        st.text_area("AI generated answer",
                     value=st.session_state[VALS_STR[0]],
                     height=200)
        st.text_area("Ground truth answer",
                     value=st.session_state[VALS_STR[1]],
                     height=200)
        st.write(st.session_state.parm_text)

    with left_column:
        query = st.selectbox(
            "Select a question about your wonder...",
            question_gui_data, index=None,
            placeholder="Select a question...",
        )
        st.write("You selected:", query)
        if st.button("Ask AI"):
            # Update the two text areas and parameter value with content
            rag_answer, param_value = get_respond_streamlit(query)
            st.session_state[VALS_STR[0]] = rag_answer
            st.session_state[VALS_STR[1]] = ground_truth_data[query]
            st.session_state.parm_text = f"{PARAMS[0]}: {param_value}"
            st.rerun()


def run() -> None:
    """Run streamlit gui application for ai rag answering."""
    enter_wonder_question()


if __name__ == "__main__":
    run()
