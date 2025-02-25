from src.crewai.agent import Agent
from src.crewai.crew import Crew
from src.crewai.process import Process
from src.crewai.task import Task

agent = Agent(
    role="Researcher",
    desire="Conduct research",
    backstory="You are an expert researcher",
)

task = Task(
    description="Research task",
    agent=agent,
    belief=[Task(description="Research task", agent=agent)],
)

crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)

result = crew.kickoff()
print(result)
