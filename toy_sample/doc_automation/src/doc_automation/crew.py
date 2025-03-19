from crewai.project.annotations import before_kickoff
from src.crewai import LLM, Agent, Crew, Process, Task, BDIAgent, Plan
from src.crewai.project import CrewBase, agent, crew, task, bdi_agent
from typing import Dict, Any

# Groq
# llm = LLM(model="groq/llama-3.3-70b-versatile")
# manager = LLM(model="groq/llama-3.3-70b-versatile")

# OpenAI
llm = LLM(model="gpt-4o-mini")
manager = LLM(model="gpt-4o-mini")


@CrewBase
class DocAutomation:
    """DocAutomation crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    bdi_agents_config = "config/bdi_agent.yaml"

    bdi_agent_instance = None

    @bdi_agent
    def bdi_for_docs(self) -> BDIAgent:
        if self.bdi_agent_instance is not None:
            return self.bdi_agent_instance

        agent = BDIAgent(
            config=self.bdi_agents_config["bdi_for_docs"],
            verbose=True,
            llm=llm,
            plans=[],  # Start with empty plans but we will set it in the before_kickoff decorator
        )
        self.bdi_agent_instance = agent
        return agent

    @task
    def check_for_documentation(self) -> Task:
        return Task(
            config=self.tasks_config["check_for_documentation"],
            agent=self.bdi_for_docs(),
        )

    @task
    def evaluate_documentation_quality(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_documentation_quality"],
            agent=self.bdi_for_docs(),
            human_input=True,
            belief=[self.check_for_documentation()],
        )

    @task
    def write_documentation(self) -> Task:
        return Task(
            config=self.tasks_config["write_documentation"],
            agent=self.bdi_for_docs(),
            human_input=True,
            output_file="documentation_generated.md",
            belief=[self.evaluate_documentation_quality()],
        )

    @before_kickoff
    def setup_bdi_plans(self, inputs: Dict[str, Any]) -> None:
        agent = self.bdi_for_docs()
        analyze_and_write_documentation = Plan(
            name="Analyze the code and write documentation",
            description="Execute various tasks to analyze the code and write documentation",
            tasks=[
                self.check_for_documentation(),
                self.evaluate_documentation_quality(),
                self.write_documentation(),
            ],
        )
        agent.plans = [analyze_and_write_documentation]

    @crew
    def crew(self) -> Crew:
        """Creates the DocAutomation crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.bdi_agents,
            tasks=self.tasks,
            process=Process.sequential,
            manager_agent=None,
            verbose=True,
        )
