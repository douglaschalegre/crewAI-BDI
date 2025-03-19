from typing import List, Optional

from pydantic import BaseModel, Field

from crewai.task import Task


class Plan(BaseModel):
    """
    A plan is a list of tasks that are executed in a sequence.
    """

    name: str = Field(
        description="Name of the plan.",
    )
    description: str = Field(
        description="Description of the plan.",
    )
    tasks: List[Task] = Field(
        default=[],
        description="List of tasks that are executed in a sequence.",
    )
    current_task: Optional[Task] = Field(
        default=None,
        description="Current task of the plan.",
    )
    current_task_index: Optional[int] = Field(
        default=0,
        description="Current task index of the plan. If the plan is not started, the current task index is 0. If the plan is finished, the current task index is None.",
    )

    def _update_current_task_index(self, index: int) -> None:
        """Update the current task index of the plan."""
        if index >= len(self.tasks):
            self.current_task_index = None
        else:
            self.current_task_index = index

    def find_next_task(self) -> Optional[Task] | None:
        """Find the next task of the plan."""
        if self.current_task_index is None:
            return None
        else:
            return self.tasks[self.current_task_index]

    def update_current_task(self, task: Task) -> None:
        """Update the current task of the plan."""
        assert task.name == self.tasks[self.current_task_index].name, (
            f"Task name {task.name} is not the current task."
        )
        # Increase one because plan execution is always sequential
        self._update_current_task_index(self.current_task_index + 1)
        self.current_task = self.find_next_task()

    def is_finished(self) -> bool:
        """Check if the plan is finished."""
        return self.current_task_index is None
