"""Construct utility functions to generate useful data."""
import pickle
from typing import Dict, List, Tuple

from haystack import Pipeline


def create_gt_answer_data() -> List[str]:
    """Create utility ground truth data - answers for questions."""
    all_gt_answers = [
        'It was 115 m (377 ft) long and 46 m (151 ft) wide, '
        + 'supposedly the first Greek temple built of marble. '
        + 'Its peripteral columns stood some 13 m (40 ft) high, '
        + 'in double rows that formed a wide ceremonial passage '
        + 'around the cella that housed the goddess\'s cult image.',
        'The Great Pyramid of Giza was the tomb of pharaoh Khufu, and '
        + 'still contains his granite sarcophagus. It had, '
        + 'like other tombs of Egyptian elites, four main purposes: '
        + 'It housed the body of the deceased and kept it safe. '
        + 'It demonstrated the status of the deceased and his family.',
        'Legend has it that King Nebuchadnezzar of Babylon had the gardens '
        + 'built as a gift to his wife Semiramis, '
        + 'a Persian princess, to ease her homesickness for the green forests '
        + 'of her homeland.',
        'It was said to be 70 cubits (105 feet [32 metres]) tall, '
        + 'and it depicted the sun god Helios. Many representations '
        + 'of the statue depict the figure as nude or semi-nude, '
        + 'except for a cloak, '
        + 'and a representation in one relief suggests '
        + 'that the figure was shielding its eyes with one hand.',
        'For many centuries it was one of the tallest man-made structures '
        + 'in the world. The lighthouse was severely damaged by '
        + 'three earthquakes between 956 and 1323 AD and became '
        + 'an abandoned ruin.',
        'The monument was the tomb of Mausolus, ruler of Caria, '
        + 'in southwestern Asia Minor. It was built in his capital city, '
        + 'Halicarnassus, between about 353 and 351 bce by his sister '
        + 'and and widow, Artemisia II.',
        'The Statue of Zeus at Olympia was a giant seated figure, '
        + 'about 12.4 m (41 ft) tall, made by the Greek sculptor Phidias '
        + 'around 435 BC at the sanctuary of Olympia, Greece, and erected in '
        + 'the Temple of Zeus there. Zeus is the sky and thunder god '
        + 'in ancient Greek religion, '
        + 'who rules as king of the gods of Mount Olympus.'
    ]

    return all_gt_answers


def create_gt_data() -> Dict[str, str]:
    """Create a dict with a wonder question and its ground truth answer."""
    gt_answers = create_gt_answer_data()
    questions = create_question_data()
    quest_gt_answers = {}

    for quest, ans in zip(questions, gt_answers):
        quest_gt_answers[quest] = ans

    return quest_gt_answers


def create_question_data() -> List[str]:
    """Create utility question data for seven ancient world wonders."""
    all_questions = ['What does The Temple of Artemis at Ephesus look like?',
                     'What was known for the The Great Pyramid of Giza, '
                     + 'Egypt in the Antiquity?',
                     'What was the function of Hanging Gardens of Babylon?',
                     'What does Rhodes Statue look like?',
                     'When and how was destroyed the Lighthouse of '
                     + 'Alexandria?',
                     'When was built the Mausoleum at Halicarnassus?',
                     'What is known about the Statue of Zeus?'
                     ]
    return all_questions


def create_qui_question_data() -> Tuple:
    """Provide seven hard-coded questions user to select."""
    return tuple(create_question_data())


def save_eval_data(rag_answers: List[str], retrieved_docs: List[List[str]]) \
        -> None:
    """Save evaluation data of the rag algo/given pipeline."""
    file_path = 'data/eval_data.pkl'
    with open(file_path, "wb") as file_in:
        # Writing data to a file
        pickle.dump([rag_answers, retrieved_docs], file_in)


def load_eval_data() -> Tuple[List[str], List[List[str]]]:
    """Load evaluation data of the rag algo/given pipeline."""
    file_path = 'data/eval_data.pkl'
    with open(file_path, "rb") as file_out:
        # Writing data to a file
        [rag_answers, retrieved_docs] = pickle.load(file_out)

    return rag_answers, retrieved_docs


def extract_rag_answer(response_rag: Pipeline) -> str:
    """Extract rag answer from pipeline response."""
    return response_rag["answer_builder"]["answers"][0].data


def extract_retrieved_docs(response_rag: Pipeline) -> List[str]:
    """Extract retrieved documents from pipeline response."""
    retrieved_docs = []
    retrieved_raw_docs = response_rag["answer_builder"][
        "answers"][0].documents
    for docs in retrieved_raw_docs:
        retrieved_docs.append(docs.content)

    return retrieved_docs
