from src.crewai import LLM, Agent, Crew, Process, Task
from src.crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Groq
# llm = LLM(model="groq/llama-3.3-70b-versatile")
# manager = LLM(model="groq/llama-3.3-70b-versatile")

# OpenAI
llm = LLM(model="gpt-4o-mini")
manager = LLM(model="gpt-4o-mini")


@CrewBase
class DocAutomation:
    """DocAutomation crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def software_documentation_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["software_documentation_engineer"],
            verbose=True,
            llm=llm,
        )

    @agent
    def software_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["software_engineer"], verbose=True, llm=llm
        )

    @agent
    def qa(self) -> Agent:
        return Agent(config=self.agents_config["qa"], verbose=True, llm=llm)

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def check_for_documentation(self) -> Task:
        return Task(
            config=self.tasks_config["check_for_documentation"],
            allow_delegation=True,
        )

    @task
    def evaluate_documentation_quality(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_documentation_quality"],
            human_input=True,
            belief=[self.check_for_documentation()],
        )

    @task
    def write_documentation(self) -> Task:
        return Task(
            config=self.tasks_config["write_documentation"],
            human_input=True,
            output_file="documentation_generated.md",
            belief=[self.evaluate_documentation_quality()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DocAutomation crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.hierarchical,
            manager_llm=manager,
            manager_agent=None,
            verbose=True,
        )
