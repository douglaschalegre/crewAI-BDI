#!/usr/bin/env python
import sys
import warnings

from toy_sample.doc_automation.src.doc_automation.crew import DocAutomation

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run(code_function: str, code_function_name: str):
    """
    Run the crew.
    """
    inputs = {"code_function": code_function, "code_function_name": code_function_name}

    try:
        DocAutomation().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train(code_function: str, code_function_name: str):
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"code_function": code_function, "code_function_name": code_function_name}
    try:
        DocAutomation().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        DocAutomation().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test(code_function: str, code_function_name: str):
    """
    Test the crew execution and returns the results.
    """
    inputs = {"code_function": code_function, "code_function_name": code_function_name}
    try:
        DocAutomation().crew().test(
            n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
